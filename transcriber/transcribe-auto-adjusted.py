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


T = chord_templates()  # triad_indices_seq
logging.info(f"Template: {T.shape}")


def harmony_tuning(trial, waveform, sample_rate, cache):
    """
    Optuna objective that tunes transformation parameters.

    To Do:
    - add more parameters for CQT-chroma
    - use 'normalize' as an optimized parameter
    - HPCP as an alternative to CQT-chroma
    - beat-synchronous chroma as an alternative to framewise (current)
    - [Optional] HPSS before Chroma to isolate harmonic content (reduces drum bleed) in live recordings
    - [Optional] temporal smoothing with median filter or HMM/Viterbi to reduce jitter
    """
    # librosa.feature.chroma_stft(y=y, sr=sr)
    # mean_chroma = np.mean(chromagram, axis=1)
    n_chroma = 12  # default value
    params = {
        "hop_length": trial.suggest_int("hop_length", 64, 32768, step=64),
        "bins_per_octave": trial.suggest_int(
            "bins_per_octave",
            n_chroma,
            n_chroma * n_chroma,
            step=n_chroma)
    }

    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    if key in cache:
        return cache[key]

    chroma_features = librosa.feature.chroma_cqt(
        y=waveform, sr=sample_rate,
        hop_length=params["hop_length"],
        bins_per_octave=params["bins_per_octave"]
    )
    chroma_features = librosa.util.normalize(chroma_features, axis=0)
    logging.info(f"Chroma: {chroma_features.shape}")

    scores = T @ chroma_features
    predictions = np.argmax(scores, axis=0)
    evals = evaluate_harmony(chroma_features, T, predictions)
    cache[key] = {
        "params": params,
        "evals": evals
    }
    return cache[key]["evals"]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Chroma CQT",
        directions=["minimize", "maximize", "minimize"],
    )

    cache = {}
    study.optimize(
        lambda trial: transform_tuning(
            trial,
            waveform, sample_rate, cache
        ),
        n_trials=200
    )

    for trial in study.best_trials:
        print(f"Trial #{trial.number}")
        print(f"  Values: {trial.values}")
        print(f"  Params: {trial.params}")

    with Path("chroma-adjusted.json").open("w") as file:
        json.dump(cache, file, indent=4)
