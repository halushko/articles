"""Transparent, language-specific extraction of *candidates*, never auto-facts.

A deliberately small Russian grammar is a measurable baseline alongside the
general Qwen extractor, not a Home Assistant integration or a hidden truth source.
Every generated triple still requires teacher confirmation.
"""
import re

# A first-person ownership assertion, not a question or an instruction.
POSSESSION = re.compile(
    r"(?iu)(?:^|[.!;]\s+)(?:у\s+меня|в\s+меня)\s+"
    r"(?:есть\s+)?(?P<thing>[^!?\n.;]{2,150})"
)
PLACEMENT = re.compile(
    r"(?iu)^(.{2,80}?)\s+(на|использует|работает\s+на)\s+(.{2,80})$"
)


def grounded_candidates(text):
    """Return at most two exact, source-anchored relation candidates."""
    text = str(text or "").strip()
    if not text or "?" in text or len(text)>900:
        return []
    match = POSSESSION.search(text)
    if not match:
        return []
    phrase = match.group("thing").strip(" ,.:")
    # Avoid teaching from simple discourse markers.
    if not phrase or re.match(r"(?iu)^(вопрос|проблема|идея|предложение)\b", phrase):
        return []
    subject, link, target = phrase, None, None
    placement = PLACEMENT.match(phrase)
    if placement:
        subject=placement.group(1).strip()
        link=placement.group(2).strip()
        target=placement.group(3).strip()
    if not (2 <= len(subject) <= 80) or len(subject.split())>9:
        return []
    # Generic exact asserted relation. The word 'на' stays deliberately
    # ambiguous, rather than silently making it 'powered_by'/'located_at'.
    result=[{
        "kind":"fact", "name":f"Пользователь имеет {subject}"[:120],
        "description":f"Учитель сообщил о наличии: {subject}.",
        "domain":"dialogue", 
        "details":{"subject":"user","predicate":"имеет","object":subject}
    }]
    if link and target and len(target.split())<=9:
        result.append({
            "kind":"fact",
            "name":f"{subject}: {link} {target}"[:120],
            "description":f"Учитель сообщил: «{subject} {link} {target}».",
            "domain":"dialogue",
            "details":{"subject":subject,"predicate":link,"object":target}
        })
    return result
