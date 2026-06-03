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
    - add 7th, maj7, min7, sus2/4, power (5) chords by extending templates and priors
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
import numpy as np
import librosa
from pathlib import Path
from utility import logging, chord_templates

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

T, L = chord_templates()
logging.info(f"Template: {T.shape} | Labels: {L}")

# Get harmony lane
hop_length = 28480
chroma = librosa.feature.chroma_cqt(
    y=waveform, sr=sample_rate,
    hop_length=hop_length,
    bins_per_octave=84
)
chroma = librosa.util.normalize(chroma, axis=0)
logging.info(f"Chroma: {chroma.shape}")

scores = T @ chroma  # raw estimates

# Smooth predictions?
predictions = np.argmax(scores, axis=0)
frames = (np.arange(predictions.size) * hop_length / sample_rate)

frame_time = hop_length / sample_rate  # Duration of one frame in seconds
current_note = None
start_time = 0

harmony_segments = []
prediction = None

# Merge segments
for p in range(len(predictions)):
    if predictions[p] != prediction:
        current_time = p * frame_time
        if prediction is not None:
            harmony_segments.append([
                start_time,
                current_time,
                L[predictions[p]]
            ])
        prediction = predictions[p]
        start_time = current_time

# Get melody lane
# Extract fundamental frequency (f0) using pYIN
hop_length = 512
f0, voiced_flag, voiced_prob = librosa.pyin(
    waveform, sr=sample_rate,
    hop_length=hop_length,
    fmin=librosa.note_to_hz("E2"),
    fmax=librosa.note_to_hz("E7"),
    fill_na=np.nan
)

# Clean the melody (replace unvoiced frames with NaN or 0)
melody_hz = np.where(voiced_flag, f0, np.nan)  # np.where(condition, x, y)

# Convert continuous frequencies to MIDI note numbers
# Clean up NaN values for the conversion step
melody_midi = np.zeros_like(melody_hz)
valid_mask = ~np.isnan(melody_hz)
melody_midi[valid_mask] = librosa.hz_to_midi(melody_hz[valid_mask])
melody_midi[~valid_mask] = np.nan  # Keep unvoiced segments as NaN

frame_time = hop_length / sample_rate  # Duration of one frame in seconds

melody_segments = []
note = None
start_time = 0

for n in range(len(melody_midi)):
    current_time = n * frame_time
    value = melody_midi[n]
    rounded_note = None if np.isnan(value) else int(np.round(melody_midi[n]))
    if rounded_note != note:
        if note is not None:
            if current_time - start_time > 0.005:  # Filter out noise shorter than 5ms
                melody_segments.append([
                    start_time,
                    current_time,
                    librosa.midi_to_note(note)
                ])

        # Start tracking the new state
        note = rounded_note
        start_time = current_time

# To Do: implement as a part of auto-adjustment pipeline
# flatness = librosa.feature.spectral_flatness(y=waveform, hop_length=hop_length)
# print(flatness, "flatness")
# print(flatness.shape)

output = {
    "file_path": audio_file_path.resolve().as_posix(),
    "sample_rate": sample_rate,
    "duration": waveform.shape[0] / sample_rate,
    "harmony_segments": harmony_segments,
    "melody_segments": melody_segments
}

with open("test.json", "w") as file:
    json.dump(output, file, indent=4, ensure_ascii=False, sort_keys=True)
