"""
metronome_player.py
-------------------
MetronomePlayer - drives a beat sound at a series of BPM values for
configurable durations.  Audio is handled with sounddevice + soundfile.

Beat timing uses time.perf_counter() for sub-millisecond accuracy:
    - The beat file is loaded once into RAM as a NumPy array.
    - Each beat is fired with sd.play() in NON-BLOCKING mode.
    - Any still-playing beat is stopped before the next one fires,
    preventing overlap at segment transitions and high BPMs.
    - Drift is corrected each beat by anchoring to the segment start time.

Noise:
    - bpm_amplitude: each segment's BPM is randomly offset by
                 uniform(-bpm_amplitude, +bpm_amplitude).
    - duration_amplitude: each segment's duration (minutes) is randomly offset
                      by uniform(-duration_amplitude, +duration_amplitude).
"""

import logging
import sounddevice as sd
import soundfile as sf
from numpy import ndarray, zeros
from random import uniform
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def _setup_audio_backend() -> bool:
    """Reset sounddevice to auto-select best available backend."""
    try:
        sd.default.reset()
        return True
    except Exception as exc:
        logging.error(f"Audio backend error: {exc}")
        return False


# Audio helpers

def load_beat(beat_file_path) -> Tuple[ndarray, int] | None:
    """Load the beat audio file.
    """
    try:
        data, sample_rate = sf.read(
            beat_file_path.as_posix(),
            dtype="float32",
            always_2d=True
        )
        logging.info(
            f"Beat audio file loaded: {beat_file_path.resolve()} "
            f"({data.shape[0]:,} samples; {sample_rate:,} Hz)"
        )
        return data, sample_rate
    except Exception as exc:
        raise Exception(
            f"Cannot load beat audio file: {beat_file_path}: {exc}")

# Noise helpers


def add_amplitude(
        value: int | float,
        amplitude: int | float,
        clamp: int | float = 1.0
) -> int | float:
    """Return value ± random offset, clamped by `clamp` variable."""
    if amplitude <= 0:
        return value
    new_value = value + uniform(-amplitude, amplitude)
    return max(clamp, new_value)


# Core loop

def play(bpm, duration, audio, sample_rate):
    """."""
    if not _setup_audio_backend():
        return

    samples = int(60 * sample_rate * duration)
    channels = audio.shape[1]
    buffer = zeros((samples, channels), dtype=audio.dtype)

    interval_length = int(sample_rate * 60 / bpm)  # maximum length of a beat according to BPM
    beat_length = min(audio.shape[0], interval_length)  # truncate if beat length > maximum allowed interval length

    for pos in range(0, samples, interval_length):
        if pos + beat_length > samples:
            break
        buffer[pos: pos + beat_length, :] = audio[:beat_length, :]

    sd.play(buffer, sample_rate)
    sd.wait()
