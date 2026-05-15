import logging
import numpy as np
import librosa
# import matplotlib.pyplot as plt
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

waveform, sample_rate = librosa.load("Time On My Hands.wav", sr=None, mono=True)
logging.info("Waveform: {waveform.shape:,} | Sample rate: {sample_rate:,}")

hop_length = 512
chroma = librosa.feature.chroma_cqt(
    y=waveform, sr=sample_rate,
    hop_length=hop_length,
    bins_per_octave=36
)
chroma = librosa.util.normalize(chroma, axis=0)
logging.info("Chroma: {chroma.shape}")

PITCHES = "C C# D D# E F F# G G# A A# B".split()
CHORDS = [f"{p}:{q}" for q in ("maj", "min") for p in PITCHES]
logging.info("Chords: {CHORDS}")


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
logging.info("Template: {T.shape}")

scores = T @ chroma  # (24, frames)
predictions = np.argmax(scores, axis=0)
times = np.arange(predictions.size) * (hop_length / sample_rate)


def prediction_to_triad(item):
    root = item // 2
    qual = "maj" if item % 2 == 0 else "min"
    return f"{PITCHES[root]}:{qual}"

output = pl.DataFrame({
    "S": times[:-1].round(3),
    "E": np.roll(times, -1)[:-1].round(3),
    "Triad": predictions[:-1]
}).with_columns(
    pl.col("Triad").map_elements(
        lambda item: prediction_to_triad(item),
        return_dtype=pl.String
    )
)

output.write_csv("test.csv")
