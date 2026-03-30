"""
metronome_player.py
~~~~~~~~~~~~~~~~~~~
MetronomePlayer — drives a beat sound at a series of BPM values for
configurable durations.  Audio is handled with sounddevice + soundfile.

Beat timing uses time.perf_counter() for sub-millisecond accuracy:
  • The beat file is loaded once into RAM as a NumPy array.
  • Each beat is fired with sd.play() in NON-BLOCKING mode.
  • Any still-playing beat is stopped before the next one fires,
    preventing overlap at segment transitions and high BPMs.
  • Drift is corrected each beat by anchoring to the segment start time.

Noise:
  • bpm_noise:      each segment's BPM is randomly offset by
                     uniform(-bpm_noise, +bpm_noise).
  • duration_noise:  each segment's duration (minutes) is randomly offset
                     by uniform(-duration_noise, +duration_noise).
"""

import time
import random
import logging
import sounddevice as sd
import soundfile   as sf
import numpy       as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class MetronomePlayer:
    """JSON-driven metronome player."""

    def __init__(self, config: dict) -> None:
        self.beat_file      = Path(config["beat_file"])
        self.segments       = config["segments"]        # list of {duration_minutes, bpm}
        self.bpm_noise      = config.get("bpm_noise", 0)
        self.duration_noise = config.get("duration_noise", 0)
        self._beat_data:    np.ndarray | None = None
        self._sample_rate:  int | None = None
        self._validate_config()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Raise ValueError for obviously bad config entries."""
        if not self.segments:
            raise ValueError("config['segments'] is empty — nothing to play.")
        for i, seg in enumerate(self.segments):
            if seg.get("bpm", 0) <= 0:
                raise ValueError(f"Segment {i}: bpm must be > 0 (got {seg.get('bpm')}).")
            if seg.get("duration_minutes", 0) <= 0:
                raise ValueError(
                    f"Segment {i}: duration_minutes must be > 0 "
                    f"(got {seg.get('duration_minutes')})."
                )

    # ── Noise helpers ─────────────────────────────────────────────────────────

    def _apply_bpm_noise(self, bpm: float) -> float:
        """Return bpm ± random offset, clamped to >= 1."""
        if self.bpm_noise <= 0:
            return bpm
        noisy = bpm + random.uniform(-self.bpm_noise, self.bpm_noise)
        return max(1.0, noisy)

    def _apply_duration_noise(self, duration_minutes: float) -> float:
        """Return duration ± random offset (minutes), clamped to >= 0.1."""
        if self.duration_noise <= 0:
            return duration_minutes
        noisy = duration_minutes + random.uniform(-self.duration_noise, self.duration_noise)
        return max(0.1, noisy)

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _load_beat(self) -> bool:
        """Load the beat WAV file into self._beat_data.  Returns success flag."""
        try:
            data, sr = sf.read(str(self.beat_file), dtype="float32", always_2d=True)
            self._beat_data   = data
            self._sample_rate = sr
            log.info("Beat file loaded: %s  (%d Hz, %d samples, %d ch)",
                     self.beat_file, sr, data.shape[0], data.shape[1])
            return True
        except Exception as exc:
            log.error("Cannot load beat file '%s': %s", self.beat_file, exc)
            return False

    @staticmethod
    def _setup_audio_backend() -> bool:
        """Reset sounddevice to auto-select best available backend."""
        try:
            sd.default.reset()
            return True
        except Exception as exc:
            log.error("Audio backend error: %s", exc)
            return False

    def _play_beat(self) -> None:
        """
        Fire one beat — NON-BLOCKING.

        Any previously-playing beat is stopped first so that beats
        never overlap, even when the beat sound is longer than the
        inter-beat interval.
        """
        sd.stop()                                       # cut off any prior beat
        sd.play(self._beat_data, self._sample_rate)     # non-blocking

    # ── Core loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
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
        if not self._load_beat():
            return
        if not self._setup_audio_backend():
            return

        total_segments = len(self.segments)
        log.info("Starting metronome — %d segment(s)", total_segments)

        try:
            for idx, seg in enumerate(self.segments, start=1):
                # ── Apply noise ───────────────────────────────────────────
                base_bpm      = seg["bpm"]
                base_duration = seg["duration_minutes"]
                eff_bpm       = self._apply_bpm_noise(base_bpm)
                eff_duration  = self._apply_duration_noise(base_duration)

                duration_seconds = eff_duration * 60
                beat_interval    = 60.0 / eff_bpm      # seconds between beats

                log.info(
                    "Segment %d/%d: %.1f BPM (base %d) for %.2f min (base %d)  "
                    "(beat every %.3f s)",
                    idx, total_segments, eff_bpm, base_bpm,
                    eff_duration, base_duration, beat_interval,
                )
                print(
                    f"\n▶  Segment {idx}/{total_segments}: "
                    f"{eff_bpm:.1f} BPM (base {base_bpm})  "
                    f"for {eff_duration:.2f} min (base {base_duration})",
                    flush=True,
                )

                segment_start = time.perf_counter()
                beat_index    = 0

                while True:
                    # Time when this beat SHOULD fire (drift-corrected)
                    scheduled_time = segment_start + beat_index * beat_interval
                    now = time.perf_counter()

                    # Segment finished?
                    if now - segment_start >= duration_seconds:
                        break

                    # Wait until it is time for the next beat
                    wait = scheduled_time - now
                    if wait > 0:
                        time.sleep(wait)

                    # Play beat (non-blocking, stops any prior sound first)
                    self._play_beat()
                    beat_index += 1

                # ── Smooth transition ─────────────────────────────────────
                # Wait out the remainder of the last beat interval so the
                # first beat of the next segment doesn't collide with the
                # last beat of this one.
                if beat_index > 0:
                    last_beat_time = segment_start + (beat_index - 1) * beat_interval
                    elapsed_since_last = time.perf_counter() - last_beat_time
                    remaining = beat_interval - elapsed_since_last
                    if remaining > 0:
                        time.sleep(remaining)

                log.info("Segment %d/%d complete.", idx, total_segments)

        except KeyboardInterrupt:
            sd.stop()
            log.info("Metronome stopped by user.")
        except Exception as exc:
            sd.stop()
            log.error("Unexpected error: %s", exc)
            raise

        sd.stop()
        log.info("All segments finished — metronome done.")
