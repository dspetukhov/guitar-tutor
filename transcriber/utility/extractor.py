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


def _suggest_cqt_parameters(trial, n_chroma):
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


def _suggest_hcqt_parameters(trial, n_chroma):
    """Baseline needs testing."""
    return {
        "hop_length": trial.suggest_int("hop_length", 64, 65536, step=64),
        "n_bins": trial.suggest_int(
            "n_bins", n_chroma, n_chroma * n_chroma, step=n_chroma
        ),
        # "harmonics"
        "bins_per_octave": 36,
        # trial.suggest_int(
        #     "bins_per_octave", n_chroma, n_chroma * n_chroma, step=n_chroma
        # ),
        "fmin": trial.suggest_float("fmin", 41.1, 211.1, step=0.1),
        "norm": trial.suggest_categorical("norm", [np.inf, None]),
    }


def _suggest_stft_parameters(trial):
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


def suggest_parameters(trial, extractor, **kwargs):
    if extractor.lower() == "stft":
        return _suggest_stft_parameters(trial)
    elif extractor.lower() == "cqt":
        return _suggest_cqt_parameters(trial, **kwargs)
    elif extractor.lower() == "hcqt":
        return _suggest_hcqt_parameters(trial, **kwargs)
    else:
        logging.warning(f"There is no such feature extraction method: {extractor}")
        return {}


def _extract_stft_features(waveform, sample_rate, n_chroma, params):
    queue = Queue()
    process = Process(
        target=_stft_worker,
        args=(queue, waveform, sample_rate, n_chroma),
        kwargs=params,
    )
    process.start()
    Thread(target=_memory_watchdog, args=(process, queue, 1024**3), daemon=True).start()
    features = queue.get()

    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join()

    if features is None:
        return []


def _extract_hcqt_features(waveform, sample_rate, n_chroma, params):
    # --- Compute HCQT: stack CQTs at each harmonic multiple of fmin ---
    # Result shape: (H, F, T) where H=harmonics, F=frequency bins, T=time frames
    hcqt_layers = []
    for h in params["harmonics"]:
        cqt_magnitude = np.abs(
            librosa.cqt(
                waveform,
                sr=sample_rate,
                fmin=params["fmin"] * h,
                n_bins=params["n_bins"],
                bins_per_octave=params["bins_per_octave"],
                hop_length=params["hop_length"],
            )
        )
        hcqt_layers.append(cqt_magnitude)
        logging.info(f"CQT harmonic h={h}: shape {cqt_magnitude.shape}")

    hcqt = np.stack(hcqt_layers, axis=0)  # (H, F, T)
    logging.info(f"HCQT tensor: {hcqt.shape}")

    # --- Pitch salience map: average across harmonic axis -> (F, T) ---
    salience = np.mean(hcqt, axis=0)  # (F, T)

    # Normalize each time frame to [0, 1] to suppress level differences
    frame_max = salience.max(axis=0, keepdims=True)
    frame_max[frame_max == 0] = 1.0  # avoid division by zero
    salience = salience / frame_max
    logging.info(f"Salience map: {salience.shape}")
    n_frames = salience.shape[1]

    # --- Triad / harmony detection ---
    # Fold the F salience bins into 12-bin chroma by summing bins that share the same pitch class
    n_chroma = 12
    chroma = np.zeros((n_chroma, n_frames))
    for k in range(params["n_bins"]):
        pitch_class = k % n_chroma
        chroma[pitch_class, :] += salience[k, :]

    # Normalize chroma columns to unit L∞ norm (matches transcribe.py convention)
    chroma_max = chroma.max(axis=0, keepdims=True)
    chroma_max[chroma_max == 0] = 1.0
    chroma = chroma / chroma_max
    logging.info(f"Chroma (from HCQT): {chroma.shape}")
    return chroma


def extract_features(extractor, waveform, sample_rate, n_chroma, params):
    if extractor.lower() == "stft":
        return _extract_stft_features(waveform, sample_rate, n_chroma, params)
    elif extractor.lower() == "cqt":
        return librosa.feature.chroma_cqt(
            y=waveform, sr=sample_rate, n_chroma=n_chroma, **params
        )
    elif extractor.lower() == "hcqt":
        return _extract_hcqt_features(waveform, sample_rate, n_chroma, params)
    else:
        logging.warning(f"Not supported feature extraction method: {extractor}")
        return np.array([])
