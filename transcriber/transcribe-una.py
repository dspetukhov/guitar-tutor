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

N_TRIALS = 26 * 3
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"

CRITERIA_DIRECTION = {
    # Harmony
    "RMSE": "minimize",
    "Cosine similarity": "maximize",
    "L2 Distance": "maximize",
    "Energy coverage": "maximize",
    # Melody
    "F1": "maximize",
    "Salience": "maximize",
}
HARMONY_FEATURES = {
    "STFT": "minimize",
    "CQT": "maximize",
    "HCQT": "maximize",
}


def get_params_key(params: dict) -> str:
    """MD5 hash of a parameter dict, used as a cache key."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def harmony_objective(
    trial, waveform, sample_rate, extractor, criterion, cache, TT, default_return
):
    """
    Optuna objective that tunes extractor parameters.

    To Do:
    - beat-synchronous chroma as an alternative to frame-wise (current)
    - [Optional] HPSS before Chroma to isolate harmonic content (reduces drum bleed) in live recordings
    - [Optional] temporal smoothing with median filter or HMM/Viterbi to reduce jitter
    """
    n_chroma = 12

    params = suggest_parameters(trial, extractor, n_chroma=n_chroma)
    if not params:
        return default_return
    key = get_params_key(params)
    if key in cache:
        return cache[key]

    try:
        features = extract_features(extractor, waveform, sample_rate, n_chroma, params)
    except librosa.util.exceptions.ParameterError:
        logging.warning(f"Trial raised ParameterError; scoring as {default_return}")
        return default_return

    logging.info(f"Features [{extractor} | {criterion}]: {features.shape}")

    scores = TT @ features
    predictions = np.argmax(scores, axis=0)
    evals = evaluate_harmony(features, TT, predictions)
    cache[key] = evals[criterion]
    return cache[key]


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


def run_study(study_type: str, extractor: str, criterion: str, TT) -> dict:
    study_name = f"{study_type}-{extractor}-{criterion}".lower().replace(" ", "-")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{LOGS_DIR}/{study_name}.log")
    )
    direction = CRITERIA_DIRECTION[criterion]

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=2026),
        load_if_exists=True,
    )

    completed = 0
    cache = {}
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        cache[get_params_key(trial.params)] = trial.value
        completed += 1

    remaining = N_TRIALS - completed
    if completed:
        logging.info(
            f"{completed}/{N_TRIALS} trials already done for {extractor}/{criterion},"
            f"running {remaining} more"
        )
    default_return = float("inf") if direction == "minimize" else -float("inf")
    # if study_type == "harmony":
    study.optimize(
        lambda trial: harmony_objective(
            trial,
            waveform,
            sample_rate,
            extractor,
            criterion,
            cache,
            TT,
            default_return,
        ),
        n_trials=remaining,
    )
    # elif study_type == "melody":
    #     study.optimize(
    #         lambda trial: melody_objective(
    #             trial, waveform, sample_rate, criterion, cache
    #         ),
    #         n_trials=remaining,
    #     )
    # else:
    #     logging.warning("Unknow study type")
    #     return {None: None}

    best_trial = {
        # "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
    }
    # print(cache)
    return best_trial


def load_config(file_path):
    if file_path.is_file():
        with open(file_path, encoding="utf-8") as file:
            config = json.load(file)
            logging.info(f"Configuration loaded: {file_path}")
            return config
    else:
        raise SystemExit("Exit: configuration file wasn't found")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        config = load_config(Path(sys.argv[1]))
    else:
        raise SystemExit("Usage: python transcribe-una.py <config_file_path>")

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    TT, _ = triads_template()

    audio_file_path = Path(config["audio_file"])
    if audio_file_path.is_file():
        waveform, sample_rate = librosa.load(
            audio_file_path.as_posix(),
            sr=None,
            mono=True,  # downmix to mono
        )
        logging.info(f"Waveform: {waveform.shape} | Sample rate: {sample_rate:,}")
    else:
        raise SystemExit("Exit: audio file wasn't found")

    output = {"audio_file": config["audio_file"]}
    if config.get("harmony_lane"):
        output["harmony_lane"] = []
        for item in config["harmony_lane"]:
            extractor = item["extractor"]
            criterion = item["criterion"]
            result = run_study("harmony", extractor, criterion, TT)
            output["harmony_lane"].append(
                {
                    "extractor": extractor,
                    "criterion": criterion,
                    "value": result["value"],
                    "parameters": result["params"],
                    "hash": get_params_key(result["params"]),
                }
            )

    with (Path(ARTIFACTS_DIR) / audio_file_path.stem).open("w") as file:
        json.dump(output, file, indent=4)
