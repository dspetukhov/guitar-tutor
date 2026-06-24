"""
ROADMAP:

- implement better features:
    - spectral whitening, and per-frame normalization
    - key-awareness: boost chords consistent with a running key estimate (major/minor) via a key -> chord prior
- better temporal modeling
    - Hidden Markov Model: learn transition probabilities between chord classes and use Viterbi decoding
    - beat-synchronous features: average chroma/frames within beats to stabilize timing
- preprocessing for live recordings
  - HPSS is a must; optionally add noise gating and gentle EQ cuts around strong drum fundamentals
  - music source separation to isolate harmonic stems prior to chord estimation (Spleeter or similar OSS)
- metrics for no reference case (e.g., weighted chord symbol recall, overlap ratio)
- augment via pitch-shift/time-stretch to cover all keys and tempos
- if necessary iterate toward a CRNN-based deep model that ingests CQT or log-mel spectrograms and predicts chord classes per frame (might outperform templates on complex mixes)
- [optional] export to MusicXML or simple chord charts for use in notation editors
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
from utility import PITCHES, logging, triads_template
from utility.extractor import extract_features, synchronize_beats

ARTIFACTS_DIR = "artifacts"
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


def main(config):
    TT, L = triads_template()

    # Identify best extractor based on criterion
    by_criterion = {}
    for record in config["harmony_lane"]:
        by_criterion.setdefault(record["criterion"], []).append(record)

    best_per_criterion = {}
    for criterion, records in by_criterion.items():
        direction = CRITERIA_DIRECTION[criterion]
        # For "minimize" - ascending (smallest first); for "maximize" - descending
        records_sorted = sorted(
            records,
            key=lambda e: e["value"],
            reverse=(direction == "maximize"),
        )
        best_per_criterion[criterion] = records_sorted[0]  # first = best

    extractor, criterion, beat, parameters = None, None, None, {}
    for c, b in best_per_criterion.items():
        if extractor is None:
            extractor = b["extractor"]
            criterion = c
            beat = b["beat"]
            parameters = b["parameters"]
        print(
            f"{c} ({CRITERIA_DIRECTION[c]}): "
            f"extractor = {b['extractor']} | beat = {b['beat']} | value = {b['value']:.6f}"
        )
        print(f"\tparams = {b['parameters']}")

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

    percussive = None
    if parameters["hpss"]:
        waveform, percussive = librosa.effects.hpss(
            y=waveform, hop_length=parameters["hop_length"]
        )

    n_chroma = 12
    features = extract_features(
        extractor,
        waveform,
        sample_rate,
        n_chroma,
        parameters,
    )

    features = synchronize_beats(
        beat,
        percussive if parameters["hpss"] else waveform,
        sample_rate,
        features,
        parameters,
    )

    logging.info(f"Features: {extractor} / {criterion} / {beat}: {features.shape}")

    scores = TT @ features
    predictions = np.argmax(scores, axis=0)

    frame_time = (
        parameters["hop_length"] / sample_rate
    )  # Duration of one frame in seconds
    start_time = 0
    harmony_segments = []
    prediction = None

    triads_description = {}

    # Merge segments
    for p in range(len(predictions)):
        if predictions[p] != prediction:
            current_time = p * frame_time
            if prediction is not None:
                label = L[predictions[p]]
                if label not in triads_description:
                    triads_description[label] = [
                        PITCHES[p] for p in TT[predictions[p]].nonzero()[0]
                    ]
                harmony_segments.append([start_time, current_time, label])
            prediction = predictions[p]
            start_time = current_time

    output = {
        "audio_file": audio_file_path.as_posix(),
        "sample_rate": sample_rate,
        "duration": waveform.shape[0] / sample_rate,
        "harmony_segments": harmony_segments,
        "extractor": extractor,
        "criterion": criterion,
        "beat": beat,
        "parameters": parameters,
        "triads_description": triads_description,
    }

    artifact_file_path = Path(ARTIFACTS_DIR) / audio_file_path
    with artifact_file_path.with_name(
        artifact_file_path.stem + "-transcription.json"
    ).open("w") as file:
        json.dump(output, file, indent=4, ensure_ascii=False, sort_keys=True)


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
        main(config)
    else:
        raise SystemExit("Usage: python transcribe.py <config_file_path>")
