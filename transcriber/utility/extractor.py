import logging
from multiprocessing import Process, Queue
from threading import Thread
from time import sleep

import librosa
import numpy as np
import psutil


def _memory_watchdog(process, queue, limit_bytes):
    p = psutil.Process(process.pid)
    while process.is_alive():
        if p.memory_info().rss > limit_bytes:
            logging.warning("Terminating process: memory exceeded 1GB")
            process.terminate()
            break
        sleep(0.1)  # Poll every 0.5 seconds
    queue.put(None)


def _stft_worker(queue, y, sr, n_chroma, **kwargs):
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, **kwargs)
    queue.put(chromagram)


def _get_cqt_parameters(trial, n_chroma):
    return {
        "hop_length": trial.suggest_int("hop_length", 64, 65536, step=64),
        "n_octaves": trial.suggest_int("n_octaves", 4, 10, step=1),
        "bins_per_octave": trial.suggest_int(
            "bins_per_octave", n_chroma, n_chroma * n_chroma, step=n_chroma
        ),
        "fmin": trial.suggest_float("fmin", 41.1, 211.1, step=0.1),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
        "cqt_mode": trial.suggest_categorical("cqt_mode", ["full", "hybrid"]),
    }


def _get_hcqt_parameters(trial):
    """Baseline needs testing."""
    return None


def _get_stft_parameters(trial):
    window_types = [
        "barthann",
        "bartlett",
        "blackman",
        "blackmanharris",
        "bohman",
        "boxcar",
        "cosine",
        "exponential",
        "flattop",
        "hamming",
        "hann",
        "lanczos",
        "nuttall",
        "parzen",
        "taylor",
        "triang",
        "tukey",
    ]
    params = {
        "n_fft": trial.suggest_int("n_fft", 128, 65536 * 2, step=32),
        "hop_length": trial.suggest_int("hop_length", 64, 32768, step=64),
        "win_length": trial.suggest_int("win_length", 8, 32768, step=64),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
        "center": trial.suggest_categorical("center", [True, False]),
        "base_c": trial.suggest_categorical("base_c", [True, False]),
        "window": trial.suggest_categorical("window", window_types),
        "ctroct": trial.suggest_float("ctroct", 2.0, 8.0, step=0.1),
        # "octwidth": trial.suggest_float("octwidth", 0.1, 4.9, step=0.1),
    }
    if params["center"]:
        params["pad_mode"] = trial.suggest_categorical(
            "pad_mode", ["constant", "reflect", "edge"]
        )
    params["hop_length"] = min(params["hop_length"], params["n_fft"] // 2)
    params["win_length"] = min(params["win_length"], params["n_fft"])
    return params


def get_parameters(trial, extractor, **kwargs):
    if extractor.lower() == "stft":
        return _get_stft_parameters(trial)
    elif extractor.lower() == "cqt":
        return _get_cqt_parameters(trial, **kwargs)
    elif extractor.lower() == "hcqt":
        return _get_hcqt_parameters(trial)
    else:
        logging.warning(f"There is no such feature extraction method: {extractor}")
        return {}


def extract_features(extractor, waveform, sample_rate, n_chroma, params):
    if extractor.lower() == "stft":
        queue = Queue()
        process = Process(
            target=_stft_worker,
            args=(queue, waveform, sample_rate, n_chroma),
            kwargs=params,
        )
        process.start()
        Thread(
            target=_memory_watchdog, args=(process, queue, 1024**3), daemon=True
        ).start()
        features = queue.get()

        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join()

        if features is None:
            return []

    elif extractor.lower() == "cqt":
        features = librosa.feature.chroma_cqt(
            y=waveform, sr=sample_rate, n_chroma=n_chroma, **params
        )
    elif extractor.lower() == "hcqt":
        return []
    else:
        logging.warning(f"There is no such feature extraction method: {extractor}")
        return []
