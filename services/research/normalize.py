"""Explicit, reproducible Russian/Ukrainian light-command extraction.

No language model is used here. The grammar is deliberately narrow: unknown
commands must abstain rather than trigger imagined actions.
"""
import re
import unicodedata
from collections import Counter

DEMO_TARGETS = [
    {"id": "light.lab_toilet", "name": "Світло Туалет"},
    {"id": "light.lab_corridor_toilet", "name": "Коридор в туалеті"},
    {"id": "light.lab_corridor", "name": "Світло Коридор"},
    {"id": "light.lab_bedroom", "name": "Світло Спальня"},
]

ON = {"включи", "включай", "включить", "включите", "вруби", "зажги", "зажечь", "увімкни", "вмикай", "увімкнути", "ввімкни", "ввімкнути", "вкл"}
OFF = {"выключи", "выключай", "выключить", "выключите", "выруби", "погаси", "вимкни", "вимикай", "вимкнути", "виключи", "выкл"}
NEG = {"не", "ні", "нет", "нельзя", "нехай", "ніколи", "no", "not"}
FILLER = {"пожалуйста", "будь", "ласка", "прошу", "мне", "мені", "сейчас", "зараз", "сделай", "зроби", "хочу"}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("ё", "е")
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def parse_command(text: str) -> dict:
    words = normalize(text).split()
    present_on = ON.intersection(words)
    present_off = OFF.intersection(words)
    if not present_on and not present_off:
        return {"is_command": False, "action": None, "target_text": "", "safe": True}
    if NEG.intersection(words):
        return {"is_command": True, "action": None, "target_text": "", "safe": False, "reason": "negation"}
    if present_on and present_off:
        return {"is_command": True, "action": None, "target_text": "", "safe": False, "reason": "conflicting_actions"}
    action = "on" if present_on else "off"
    words = [w for w in words if w not in ON | OFF | FILLER]
    return {"is_command": True, "action": action, "target_text": " ".join(words), "safe": True}


def features(text: str) -> dict[str, float]:
    """Bag of word and char n-gram features, L2 normalized; stable across hosts."""
    normal = normalize(text)
    word_seq = normal.split()
    raw = Counter()
    for w in word_seq:
        raw[f"w:{w}"] += 2
        padded = f"^{w}$"
        for size in (3, 4):
            for pos in range(max(0, len(padded) - size + 1)):
                raw[f"c{size}:{padded[pos:pos+size]}"] += 1
    total = sum(v*v for v in raw.values()) ** 0.5
    return {k: v/total for k, v in raw.items()} if total else {}
