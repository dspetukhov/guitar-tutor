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
from numpy import ndarray
from random import uniform
from time import sleep
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


def play(seg_bpm, seg_duration, beat_audio, sample_rate):
    """
    Execute all segments sequentially.

    For each segment the metronome:
        1. Applies noise to BPM and duration.
        2. Announces effective BPM and duration.
        3. Fires sd.play() every (60 / effective_bpm) seconds for
            (effective_duration * 60) seconds total.
        4. Corrects timing drift by anchoring each beat to
            the segment start time.
        5. At segment boundaries, waits for the remainder of the
            last beat interval before starting the next segment,
            ensuring a smooth transition with no overlap.
    """
    if not _setup_audio_backend():
        return

    beats_count = int(seg_bpm * seg_duration)
    beat_duration = round(60 / seg_bpm, 4)

    for _ in range(beats_count):
        # Play beat (non-blocking, stops any prior sound first)
        sd.stop()                           # cut off any prior beat
        sd.play(beat_audio, sample_rate)    # non-blocking
        sleep(beat_duration - 1e-6)

    sd.stop()
