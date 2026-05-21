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

hop_length = 512
chroma = librosa.feature.chroma_cqt(
    y=waveform, sr=sample_rate,
    hop_length=hop_length,
    bins_per_octave=36
)
chroma = librosa.util.normalize(chroma, axis=0)
logging.info(f"Chroma: {chroma.shape}")

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
        c_maj /= np.linalg.norm(c_maj) + 1e-9
        c_min = np.isin(
            I,
            [(root) % NP, (root+3) % NP, (root+7) % NP]
        ).astype(float)
        c_min /= np.linalg.norm(c_min) + 1e-9
        T.append(c_maj)
        T.append(c_min)
    return np.stack(T, axis=0)  # (24, 12)


T = chord_templates()
logging.info(f"Template: {T.shape}")

scores = T @ chroma  # (24, frames)


def prediction_to_triad(item):
    root = item // 2
    qual = "maj" if item % 2 == 0 else "min"
    return f"{PITCHES[root]}:{qual}"


vfunc = np.vectorize(prediction_to_triad, otypes=[str])
predictions = np.argmax(scores, axis=0)
_frames = (np.arange(predictions.size) * hop_length).tolist()

frames, triads = [], []

for s, p in zip(_frames, predictions):
    if frames and triads and p == triads[-1]:
        continue
    frames.append(s)
    triads.append(p)

output = {
    "file_path": audio_file_path.resolve().as_posix(),
    "frame": frames,
    "triad": vfunc(triads).tolist()
}

with open("test.json", "w") as file:
    json.dump(output, file, indent=4, sort_keys=True)
