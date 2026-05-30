"""
ROADMAP:

To Dos:
- improve triads detection (harmony lane/extractor)
- implement note detection (melody lane/extractor) - time-stamped pitches for the dominant line (monophonic where possible)
    - extract a single dominant melody line from guitar-heavy recordings;
    - detect onsets, estimate pitch over time, and smooth;
    - a naive “highest-energy band” melody tracker on the harmonic component is a useful baseline.

How:
- consider alternative based on Chroma STFT
    https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
- implement better features:
    - harmonic CQT, spectral whitening, and per-frame normalization
    - key-awareness: boost chords consistent with a running key estimate (major/minor) via a key -> chord prior
- better temporal modeling
    - Hidden Markov Model: learn transition probabilities between chord classes and use Viterbi decoding
    - beat-synchronous features: average chroma/frames within beats to stabilize timing
- bigger vocabulary
    - add dim/aug, 7, maj7, min7, sus2/4, power (5) chords by extending templates and priors
    - back off to simpler labels when confidence is low (e.g., from G7 -> G:maj)
- preprocessing for live recordings
  - HPSS is a must; optionally add noise gating and gentle EQ cuts around strong drum fundamentals
  - consider music source separation to isolate harmonic stems prior to chord estimation (Spleeter or similar OSS)
- use metrics for no reference case (e.g., weighted chord symbol recall, overlap ratio)
- augment via pitch-shift/time-stretch to cover all keys and tempos
- if necessary iterate toward a CRNN-based deep model that ingests CQT or log-mel spectrograms and predicts chord classes per frame (might outperform templates on complex mixes)
- [optional] export to MusicXML or simple chord charts for use in notation editors
"""

import sys
import json
import logging
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

hop_length = 28480
chroma = librosa.feature.chroma_cqt(
    y=waveform, sr=sample_rate,
    hop_length=hop_length,
    bins_per_octave=84
)
chroma = librosa.util.normalize(chroma, axis=0)
logging.info(f"Chroma: {chroma.shape}")

PITCHES = "C C# D D# E F F# G G# A A# B".split()
CHORDS = [f"{p}:{q}" for q in ("maj", "min") for p in PITCHES]
logging.info(f"Chords: {CHORDS}")


def chord_templates():
    # 12-dim pitch-class templates for triads
    NP = len(PITCHES)
    I = np.arange(NP)
    T, L = [], []
    tol = 1e-9
    for root in range(NP):
        # major: 0,4,7; minor: 0,3,7
        c_maj = np.isin(
            I,
            [(root) % NP, (root+4) % NP, (root+7) % NP]
        ).astype(float)
        c_maj /= np.linalg.norm(c_maj) + tol
        c_min = np.isin(
            I,
            [(root) % NP, (root+3) % NP, (root+7) % NP]
        ).astype(float)
        c_min /= np.linalg.norm(c_min) + tol
        c_dim = np.isin(
            I,
            [(root) % NP, (root+3) % NP, (root+6) % NP]
        ).astype(float)
        c_dim /= np.linalg.norm(c_dim) + tol
        c_aug = np.isin(
            I,
            [(root) % NP, (root+4) % NP, (root+8) % NP]
        ).astype(float)
        c_aug /= np.linalg.norm(c_aug) + tol
        T.append(c_maj)
        L.append(f"{PITCHES[root]}:maj")
        T.append(c_min)
        L.append(f"{PITCHES[root]}:min")
        T.append(c_dim)
        L.append(f"{PITCHES[root]}:dim")
        T.append(c_aug)
        L.append(f"{PITCHES[root]}:aug")
    return np.stack(T, axis=0), L  # (48, 12)


T, L = chord_templates()
logging.info(f"Template: {T.shape} | Labels: {L}")

# Get harmony lane raw estimates
scores = T @ chroma  # -> (24, frames)


def prediction_to_triad(item):
    root = item // 2
    quality = "maj" if item % 2 == 0 else "min"
    return f"{PITCHES[root]}:{quality}"


# Smooth labels?
predictions = np.argmax(scores, axis=0)
frames = (np.arange(predictions.size) * hop_length).tolist()

segments = []
_p = None

# Merge segments
for f, p in zip(frames, predictions):
    if segments:
        if _p == p:
            continue
        else:
            _p = p
            segments[-1][1] = f
            segments.append([f, None, prediction_to_triad(p)])
    else:
        _p = p
        segments.append([f, None, prediction_to_triad(p)])

segments[-1][1] = f

output = {
    "file_path": audio_file_path.resolve().as_posix(),
    "sample_rate": sample_rate,
    "duration": waveform.shape[0] / sample_rate,
    "segments": segments
}

with open("test.json", "w") as file:
    json.dump(output, file, indent=4, sort_keys=True)
