import sys
import json
import optuna
import logging
import hashlib
import numpy as np
import librosa
from pathlib import Path

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
        mono=True
    )
    logging.info(f"Waveform: {waveform.shape} | Sample rate: {sample_rate:,}")
else:
    sys.exit(1)

PITCHES = "C C# D D# E F F# G G# A A# B".split()
CHORDS = [f"{p}:{q}" for q in ("maj", "min") for p in PITCHES]
logging.info(f"Chords: {CHORDS}")


def chord_templates():
    # 12-dim pitch-class templates for maj/min triads
    NP = len(PITCHES)
    I = np.arange(NP)
    T = []
    for root in range(NP):
        # major: {0,4,7}; minor: {0,3,7}
        c_maj = np.isin(
            I,
            [(root) % NP, (root+4) % NP, (root+7) % NP]
        ).astype(float)
        # c_maj /= np.linalg.norm(c_maj) + 1e-9
        c_min = np.isin(
            I,
            [(root) % NP, (root+3) % NP, (root+7) % NP]
        ).astype(float)
        # c_min /= np.linalg.norm(c_min) + 1e-9
        T.append(c_maj)
        T.append(c_min)
    return np.stack(T, axis=0)  # (24, 12)


T = chord_templates()  # triad_indices_seq
logging.info(f"Template: {T.shape}")


def get_metrics(chroma, T, predictions):
    """
    Lower RMSE and higher cosine similarity mean
    your triads better capture the harmonic content of the audio.
    """
    reconstructed = np.zeros_like(chroma)
    for idx in range(predictions.shape[0]):
        reconstructed[np.nonzero(T[predictions[idx]])[0], idx] = 1

    rmse = np.sqrt(np.mean((chroma - reconstructed) ** 2))
    chroma_norm = chroma / (np.linalg.norm(chroma) + 1e+8)
    reconstructed_norm = reconstructed / (np.linalg.norm(reconstructed) + 1e+8)
    cosine_similarity = np.array([
        np.dot(chroma_norm[:, idx], reconstructed_norm[:, idx])
        for idx in range(chroma.shape[1])
    ])
    l2_distance = np.linalg.norm(chroma - reconstructed)
    # cos_sims = np.array([
    #     np.dot(chroma[:, t], reconstructed[:, t]) / (np.linalg.norm(chroma[:, t]) * np.linalg.norm(reconstructed[:, t]) + 1e-8)
    #     for t in range(chroma.shape[1])
    # ])
    return float(rmse), float(np.mean(cosine_similarity)), float(l2_distance)


def objective_features(trial, waveform, sample_rate, cache):
    """
    Optuna objective that tunes audio transform parameters via per-feature AUC scoring.

    Evaluates each extracted feature column independently and returns the mean AUC
    across all columns. Duplicate parameter configurations are bypassed with a cache.
    Per-trial metadata (transform params, max, median) is stored as user attributes
    so it survives process interruption and can be recovered from the journal.

    Returns:
        float: Mean AUC across all feature columns for the trial's transform parameters.
    """
    params = {
        "hop_length": trial.suggest_int("hop_length", 64, 32768, step=64),
        "bins_per_octave": trial.suggest_int("bins_per_octave", 12, 120, step=12)
    }

    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    if key in cache:
        return cache[key]

    chroma = librosa.feature.chroma_cqt(
        y=waveform, sr=sample_rate,
        hop_length=params["hop_length"],
        bins_per_octave=params["bins_per_octave"]
    )
    chroma_features = librosa.util.normalize(chroma, axis=0)
    logging.info(f"Chroma: {chroma_features.shape}")

    scores = T @ chroma
    predictions = np.argmax(scores, axis=0)
    evals = get_metrics(chroma_features, T, predictions)
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
        lambda trial: objective_features(
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
