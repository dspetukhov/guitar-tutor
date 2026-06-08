import sys
import json
import optuna
import hashlib
import numpy as np
import librosa
import psutil
from pathlib import Path
from threading import Thread
from multiprocessing import Process, Queue
from time import sleep
from utility import logging, chord_templates
from utility.metrics import evaluate_harmony

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

N_TRIALS = 260
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"

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


def _memory_watchdog(process, queue, limit_bytes):
    p = psutil.Process(process.pid)
    while process.is_alive():
        if p.memory_info().rss > limit_bytes:
            logging.warning("Terminating process: memory exceeded 1GB")
            process.terminate()
            break
        sleep(0.1)  # Poll every 0.5 seconds
    queue.put(None)


def get_params_key(params: dict) -> str:
    """MD5 hash of a parameter dict, used as a cache key."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_cqt_parameters(trial, n_chroma):
    return {
        "hop_length": trial.suggest_int("hop_length", 64, 65536, step=64),
        "n_octaves": trial.suggest_int("n_octaves", 4, 10, step=1),
        "bins_per_octave": trial.suggest_int("bins_per_octave", n_chroma, n_chroma * n_chroma, step=n_chroma),
        "fmin": trial.suggest_float("fmin", 1.1, 100.1, step=0.1),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
        "cqt_mode": trial.suggest_categorical("cqt_mode", ["full", "hybrid"]),
    }


def _stft_worker(queue, y, sr, n_chroma, **kwargs):
    chromagram = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=n_chroma,
        **kwargs
    )
    queue.put(chromagram)


def get_stft_parameters(trial):
    window_types = [
        "barthann", "bartlett", "blackman", "blackmanharris", "bohman",
        "boxcar", "cosine", "exponential",
        "flattop",
        "hamming", "hann", "lanczos", "nuttall", "parzen",
        "taylor", "triang", "tukey"
    ]
    pad_modes = ["constant", "reflect", "edge"]

    params = {
        "n_fft": trial.suggest_int("n_fft", 8, 65536 * 2, step=32),
        "hop_length": trial.suggest_int("hop_length", 64, 32768, step=64),
        "win_length": trial.suggest_int("win_length", 8, 32768, step=64),
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

    key = get_params_key(params)
    if key in cache:
        return cache[key]["evals"][metric]

    try:
        if method == "cqt":
            chromagram = librosa.feature.chroma_cqt(
                y=waveform, sr=sample_rate, n_chroma=n_chroma,
                **params
            )
        elif method == "stft":
            queue = Queue()
            process = Process(
                target=_stft_worker,
                args=(queue, waveform, sample_rate, n_chroma),
                kwargs=params
            )
            process.start()
            Thread(
                target=_memory_watchdog,
                args=(process, queue, 1024**3),
                daemon=True
            ).start()
            chromagram = queue.get()

            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()

            if chromagram is None:
                return np.inf
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
    return cache[key]["evals"][metric]


def run_study(method: str, criterion: str) -> dict:
    cache = {"method": method}
    study_name = f"harmony-{method}-{criterion}"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{LOGS_DIR}/{study_name}.log")
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=2026),
        load_if_exists=True
    )

    completed = 0
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        cache[get_params_key(trial.params)] = trial.value
        completed += 1

    remaining = N_TRIALS - completed
    if completed:
        logging.info(
            f"{completed}/{N_TRIALS} trials already done for {method}/{criterion},"
            f"running {remaining} more"
        )

    study.optimize(
        lambda trial: harmony_objective(
            trial,
            waveform, sample_rate, criterion, cache
        ),
        n_trials=remaining
    )

    best_trial = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params
    }

    return best_trial


if __name__ == "__main__":
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    criterion = "RMSE"
    stft = run_study("stft", criterion)
    logging.info(f"stft: {stft}")
    cqt = run_study("cqt", criterion)
    logging.info(f"cqt: {cqt}")
