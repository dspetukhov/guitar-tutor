"""
metronome_main.py
-----------------
Entry point for the configurable metronome.

Usage
-----
  python main.py                     # uses config.json
  python main.py my_config.json      # custom config path
"""

import sys
import json
import logging
from pathlib import Path
from player import _load_beat, _add_amplitude, play

DEFAULT_CONFIG = "config.json"


def load_config(path: str) -> dict:
    """
    Load and return JSON config; exit on failure.
    """
    config_path = Path(path)
    if not config_path.exists():
        logging.error("Config file not found: {config_path}")
        sys.exit(1)
    with config_path.open() as file:
        config = json.load(file)
        logging.info(f"Config loaded: {config_path}")
    return config


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    beat_audio = _load_beat(config.get("beat_file"))

    bpm_amplitude = config.get('bpm_amplitude', 0)
    duration_amplitude = config.get('duration_amplitude', 0)
    logging.info(f"BPM amplitude     : ±{bpm_amplitude}")
    logging.info(f"Duration amplitude: ±{duration_amplitude} min")

    if config.get("segments"):
        logging.info(f"Segments          : {len(config['segments'])}")
        for i, seg in enumerate(config.get("segments", []), start=1):
            if seg.get("bpm", 0) <= 0:
                logging.warning(
                    f"Segment {i}: bpm must be > 0 (got {seg.get('bpm')})"
                )
                continue
            if seg.get("duration_minutes", 0) <= 0:
                logging.warning(
                    f"Segment {i}: duration_minutes must be > 0 "
                    f"(got {seg.get('duration_minutes')})"
                )
                continue

            bpm = _add_amplitude(seg["bpm"], bpm_amplitude)
            duration_minutes = _add_amplitude(seg["duration_minutes"], duration_amplitude, 0.1)
            logging.info(f"  [{i}] {duration_minutes} min - {bpm} BPM")

            play(bpm, duration_minutes, beat_audio)
        sd.stop()
        logging.info("All segments finished - metronome done.")
    else:
        raise ValueError("segments aren't specified - nothing to play")


if __name__ == "__main__":
    main()
