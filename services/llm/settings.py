"""Independent model configuration, isolated from research parameters."""
import json
import os
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "llm.json"


def load() -> dict:
    with open(os.getenv("LLM_SETTINGS", str(DEFAULT)), encoding="utf-8") as file:
        cfg = json.load(file)
    for key in ("fast", "deep"):
        if key not in cfg.get("profiles", {}):
            raise ValueError(f"Missing LLM profile {key}")
    return cfg
