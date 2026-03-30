"""
metronome_main.py
~~~~~~~~~~~~~~~~~
Entry point for the JSON-driven Python metronome.

Usage
-----
  python main.py                     # uses config.json
  python main.py my_config.json      # custom config path
"""

import sys
import json
import logging
from pathlib import Path
from player import MetronomePlayer

DEFAULT_CONFIG = "config.json"


def load_config(path: str) -> dict:
    """Load and return JSON config; exit with message on failure."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        logging.error("Config file not found: %s", cfg_path)
        sys.exit(1)
    with cfg_path.open() as fh:
        return json.load(fh)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path)

    print(f"Loaded config : {config_path}")
    print(f"Beat file     : {config.get('beat_file', '<not set>')}")
    print(f"BPM noise     : ±{config.get('bpm_noise', 0)}")
    print(f"Duration noise: ±{config.get('duration_noise', 0)} min")
    print(f"Segments      : {len(config.get('segments', []))}")
    for i, seg in enumerate(config.get("segments", []), start=1):
        print(f"  [{i}] {seg['duration_minutes']} min @ {seg['bpm']} BPM")

    player = MetronomePlayer(config)
    player.run()


if __name__ == "__main__":
    main()
