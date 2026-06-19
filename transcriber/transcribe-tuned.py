import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import optuna
from utility import logging, triads_template
from utility.extractor import extract_features, suggest_parameters
from utility.metrics import evaluate_harmony, evaluate_melody

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

N_TRIALS = 260
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"

HARMONY_CRITERIA = {
    "RMSE": "minimize",
    "Cosine similarity": "maximize",
    "L2 Distance": "maximize",
    "Energy coverage": "maximize",
}
HARMONY_FEATURES = {
    "STFT": "minimize",
    "CQT": "maximize",
    "HCQT": "maximize",
}

MELODY_CRITERIA = {
    "F1": "maximize",
    "Salience": "maximize",
}


def get_params_key(params: dict) -> str:
    """MD5 hash of a parameter dict, used as a cache key."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def harmony_objective(trial, waveform, sample_rate, extractor, metric, cache, TT):
    """
    Optuna objective that tunes transformation parameters.

    To Do:
    - beat-synchronous chroma as an alternative to framewise (current)
    - [Optional] HPSS before Chroma to isolate harmonic content (reduces drum bleed) in live recordings
    - [Optional] temporal smoothing with median filter or HMM/Viterbi to reduce jitter
    """
    n_chroma = 12

    params = suggest_parameters(trial, extractor, n_chroma=n_chroma)
    if params is None:
        return np.inf
    key = get_params_key(params)
    if key in cache:
        return cache[key]["evals"][metric]

    try:
        features = extract_features(
            extractor, waveform, sample_rate, n_chroma, **params
        )
    except librosa.util.exceptions.ParameterError:
        logging.warning("Trial raised ParameterError; scoring as +inf")
        return np.inf

    logging.info(f"Features [{extractor}]: {features.shape}")

    scores = TT @ features
    predictions = np.argmax(scores, axis=0)
    evals = evaluate_harmony(features, TT, predictions)
    cache[key] = {"params": params, "evals": evals}
    return cache[key]["evals"][metric]


def melody_objective(trial, waveform, sample_rate, metric, cache):
    """..."""
    n_chroma = 12
    frame_step = 32

    fmin = trial.suggest_float("fmin", 40.0, 115.0, step=0.1)
    fmax = trial.suggest_float("fmax", 2000.0, 2200.0, step=0.1)

    min_frame_length = int(sample_rate / fmin / frame_step + 1) * frame_step
    max_frame_length = (min(65536, len(waveform)) // frame_step) * frame_step
    logging.info(min_frame_length)
    frame_length = trial.suggest_int(
        "frame_length", min_frame_length, max_frame_length, step=frame_step
    )

    hop_divisor = trial.suggest_int("hop_divisor", 2, 8)
    hop_length = max(1, frame_length // hop_divisor)
    params = {
        "frame_length": frame_length,
        "hop_length": hop_length,
        "fmin": fmin,
        "fmax": fmax,
        "resolution": trial.suggest_float("resolution", 0.01, 0.99, step=0.01),
        "switch_prob": trial.suggest_float("switch_prob", 0.0, 1.0, step=0.01),
        "bins_per_octave": trial.suggest_int(
            "bins_per_octave", n_chroma, n_chroma * n_chroma, step=n_chroma
        ),
        "center": trial.suggest_categorical("center", [True, False]),
        "pad_mode": trial.suggest_categorical(
            "pad_mode", ["constant", "reflect", "edge"]
        ),
    }

    key = get_params_key(params)
    if key in cache:
        return cache[key]["evals"][metric]

    try:
        evals = evaluate_melody(waveform, sample_rate, params)
    except librosa.util.exceptions.ParameterError as e:
        logging.warning(f"Trial raised ParameterError: {e}; scoring as 0.0")
        return 0.0

    logging.info(evals)
    cache[key] = {"params": params, "evals": evals}
    return cache[key]["evals"][metric]


def run_study(study_type: str, method: str, criterion: str, direction: str, T) -> dict:
    if study_type == "melody":
        cache = {}
        study_name = f"{study_type}-{criterion}"
    else:
        cache = {"method": method}
        study_name = f"{study_type}-{method}-{criterion}"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{LOGS_DIR}/{study_name}.log")
    )
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=2026),
        load_if_exists=True,
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

    if study_type == "harmony":
        study.optimize(
            lambda trial: harmony_objective(
                trial, waveform, sample_rate, criterion, cache, T
            ),
            n_trials=remaining,
        )
    elif study_type == "melody":
        study.optimize(
            lambda trial: melody_objective(
                trial, waveform, sample_rate, criterion, cache
            ),
            n_trials=remaining,
        )
    else:
        logging.warning("Unknow study type")
        return {None: None}

    best_trial = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
    }

    return best_trial


if __name__ == "__main__":
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    TT, _ = triads_template()

    audio_file_path = Path("ll-slice.wav")
    if audio_file_path.is_file():
        waveform, sample_rate = librosa.load(
            audio_file_path.as_posix(),
            sr=None,
            mono=True,  # downmix to mono
        )
        logging.info(f"Waveform: {waveform.shape} | Sample rate: {sample_rate:,}")
    else:
        sys.exit(1)

    # harmony_criterion, direction = "RMSE", "minimize"
    # stft = run_study("harmony", "stft", harmony_criterion, direction, T)
    # logging.info(f"stft: {stft}")
    # cqt = run_study("harmony", "cqt", harmony_criterion, direction, TT)
    # logging.info(f"cqt: {cqt}")

    # melody_criterion, direction = "salience", "maximize"
    # melody = run_study("melody", "", melody_criterion, direction, T)
    # logging.info(f"melody: {melody}")

    # output = {
    #     "audio_file_path": str(audio_file_path),
    #     "harmony-criterion": harmony_criterion,
    #     "harmony-stft": stft,
    #     "harmony-cqt": cqt,
    # "melody-criterion": melody_criterion,
    # "melody": melody
    # }
    # with (Path(ARTIFACTS_DIR) / audio_file_path.stem).open("w") as file:
    #     json.dump(output, file, indent=4)
