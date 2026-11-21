"""Independent research knobs. Not imported by the LLM service."""
import json
import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "research.json"


def load() -> dict:
    with open(os.getenv("RESEARCH_SETTINGS", str(DEFAULT)), encoding="utf-8") as file:
        settings = json.load(file)
    for name in ("learning_rate", "online_min_similarity", "online_min_margin", "semantic_min_similarity", "semantic_min_margin"):
        if float(settings[name]) < 0 or float(settings[name]) > 1:
            raise ValueError(f"Invalid research setting {name}")
    return settings
