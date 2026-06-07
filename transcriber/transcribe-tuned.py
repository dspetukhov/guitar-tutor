import sys
import json
import optuna
import hashlib
import numpy as np
import librosa
from pathlib import Path
from utility import logging, chord_templates
from utility.metrics import evaluate_harmony

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

audio_file_path = Path("Time On My Hands.wav")

if audio_file_path.is_file():
    waveform, sample_rate = librosa.load(
        audio_file_path.as_posix(),
        sr=None,
        mono=True  # downmix to mono
    )
    logging.info(f"Waveform: {waveform.shape} | Sample rate: {sample_rate:,}")
else:
    sys.exit(1)


T, _ = chord_templates()
logging.info(f"Template: {T.shape}")


def get_cqt_parameters(trial, n_chroma):
    return {
        "hop_length": trial.suggest_int("hop_length", 64, 65536, step=64),
        "n_octaves": trial.suggest_int("n_octaves", 4, 10, step=1),
        "bins_per_octave": trial.suggest_int("bins_per_octave", n_chroma, n_chroma * n_chroma, step=n_chroma),
        "fmin": trial.suggest_float("fmin", 1.1, 100.1, step=0.1),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
        "cqt_mode": trial.suggest_categorical("cqt_mode", ["full", "hybrid"]),
    }


def get_stft_parameters(trial):
    window_types = [
        "barthann", "bartlett",
        "boxcar", "cosine", "exponential",
        "flattop",
        "hamming", "hann", "nuttall", "parzen",
        "taylor"
    ]
    pad_modes = ["constant", "reflect", "edge", "wrap"]

    params = {
        "n_fft": trial.suggest_int("n_fft", 8, 65536 * 2, step=32),
        "hop_length": trial.suggest_int("hop_length", 64, 32768, step=64),
        "win_length": trial.suggest_int("hop_length", 8, 32768, step=64),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
        "center": trial.suggest_categorical("center", [True, False]),
        "base_c": trial.suggest_categorical("base_c", [True, False]),
        "window": trial.suggest_categorical("window", window_types),
        "ctroct": trial.suggest_float("ctroct", 2.0, 8.0, step=0.1),
        # "octwidth": trial.suggest_float("octwidth", 0.1, 4.9, step=0.1),
    }
    print(params["window"])
    if params["center"]:
        params["pad_mode"] = trial.suggest_categorical("pad_mode", pad_modes)
    params["hop_length"] = min(params["hop_length"], params["n_fft"] // 2)
    params["win_length"] = min(params["win_length"], params["n_fft"])
    return params


def harmony_objective(trial, waveform, sample_rate, metric, cache):
    """
    Optuna objective that tunes transformation parameters.

    To Do:
    - beat-synchronous chroma as an alternative to framewise (current)
    - [Optional] HPSS before Chroma to isolate harmonic content (reduces drum bleed) in live recordings
    - [Optional] temporal smoothing with median filter or HMM/Viterbi to reduce jitter
    """
    method = cache["method"]
    n_chroma = 12

    if method == "cqt":
        params = get_cqt_parameters(trial, n_chroma)
    elif method == "stft":
        params = get_stft_parameters(trial)
    else:
        return np.inf

    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    if key in cache:
        return cache[key]["evals"][metric]

    try:
        if method == "cqt":
            chromagram = librosa.feature.chroma_cqt(
                y=waveform, sr=sample_rate, n_chroma=n_chroma,
                **params
            )
        elif method == "stft":
            chromagram = librosa.feature.chroma_stft(
                y=waveform, sr=sample_rate, n_chroma=n_chroma,
                **params
            )
        else:
            return np.inf

    except librosa.util.exceptions.ParameterError:
        logging.warning("Trial raised ParameterError; scoring as +inf")
        return np.inf

    logging.info(f"Chromagram [{method}]: {chromagram.shape}")

    scores = T @ chromagram
    predictions = np.argmax(scores, axis=0)
    evals = evaluate_harmony(chromagram, T, predictions)
    cache[key] = {
        "params": params,
        "evals": evals
    }
    # Use only one metric as a first approximation
    return cache[key]["evals"][metric]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Harmony lane",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=2026)
    )

    cache = {"method": "stft"}
    study.optimize(
        lambda trial: harmony_objective(
            trial,
            waveform, sample_rate, "RMSE", cache
        ),
        n_trials=300
    )

    for trial in study.best_trials:
        print(f"Trial #{trial.number}")
        print(f"  Values: {trial.values}")
        print(f"  Params: {trial.params}")

    with Path(f"harmony-lane-{cache['method']}-tuned.json").open("w") as file:
        json.dump(cache, file, indent=4)
