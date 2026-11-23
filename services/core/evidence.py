"""Deterministic provenance, search query and answer checks.

No knowledge may be taken from Qwen weights. Gemini text is accepted only
as a cited synthesis of actual fetched public pages, not as a new source.
"""
import re
from urllib.parse import urlsplit

STOP = frozenset("""как что это мне для где когда кто какие какой какая почему
найди найти поищи ищи скажи расскажи дай покажи получить
в интернете пожалуйста а и или на по в у с из от к до не
the a an and or to for why how what where who search find
що як де коли мені для знайди пошукай будь ласка
про про свій свою сам тебе його її про які""".split())

PRIVATE = [
    r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"\b(?:token|password|secret|парол\w*|токен\w*|api.?key)\b",
    r"\b(?:мой|моя|моё|мои|меня|наша|наши|свой|своя|своё|мiй|мій|мої|наш|мене)\b",
    r"\b(?:http|https|file|ssh)://\S+",
]


def terms(value):
    return [s for s in re.findall(r"(?u)[^\W_]+", str(value).lower())
            if len(s) >= 4 and s not in STOP and not s.isdecimal()]


def query_eligible(text, query, task=None, private_names=()):
    """Reject Qwen additions not supported by user's words/task context.

    Security policy is conservative. When a paraphrase is too different,
    clarification is preferable to silently searching an invented topic.
    """
    query = " ".join(str(query or "").split())
    if not 3 <= len(query) <= 240:
        return False, "invalid_query"
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in PRIVATE):
        return False, "possible_private_data"
    for name in private_names:
        name = str(name or "").strip()
        if len(name) >= 4 and re.search(r"(?iu)(?<!\w)" + re.escape(name) + r"(?!\w)", query):
            return False, "personal_entity"
    subject = str((task or {}).get("subject") or "")
    evidence_words = terms(text + " " + subject)
    if not evidence_words:
        return False, "missing_subject"
    query_words = terms(query)
    if not query_words:
        return False, "missing_subject"
    # A generic word-root check allows inflection without whitelisting
    # recipes, devices or other human scenarios.
    for word in query_words:
        if not any(word[:5] == source[:5] for source in evidence_words):
            return False, "ungrounded_query_expansion"
    return True, ""


def reliable_pages(sources):
    return [s for s in (sources or [])
            if isinstance(s, dict) and s.get("status") == "read"
            and len(str(s.get("page_text") or "")) >= 100]


def cited_synthesis(answer, sources, question=""):
    """Strict lexical entailment screen; not a proof of factual correctness.

    Requiring numbered citations alone does not stop Gemini inventing an
    ingredient with [1]. Every substantive output word must also occur in
    fetched evidence or the original public question (inflections allowed).
    Paraphrases can be rejected; safe fallback is the unmodified excerpt.
    """
    if not isinstance(answer, str) or not answer.strip() or not sources:
        return False
    refs = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    if not refs or any(n < 1 or n > len(sources) for n in refs):
        return False
    allowed_urls = {s.get("url") for s in sources}
    found_urls = re.findall(r"https?://[^\s\]\)]+", answer)
    for url in found_urls:
        if url.rstrip(".,;") not in allowed_urls:
            return False
    # Source content is the only evidence; the model's own prior response
    # and world knowledge do not qualify as citations.
    source_text = question + " " + " ".join(
        str(s.get("page_text") or "") + " " + str(s.get("title") or "")
        for s in sources)
    def root(word):
        # Two common alphabets share the same normalization policy.
        word = word.casefold().replace("ё", "е")
        return word[:5] if len(word) >= 5 else word[:3]
    known = {root(word) for word in terms(source_text)}
    sans_urls = re.sub(r"https?://\S+", " ", answer)
    sans_refs = re.sub(r"\[\d+\]", " ", sans_urls)
    # If even one new substantive token appears, don't present the
    # generator's prose as supported fact. This may reject true paraphrases.
    extra = [word for word in terms(sans_refs) if root(word) not in known]
    return not extra


def summarize_memory(knowledge):
    """Verbatim confirmed records, no model-supplied claim generation."""
    facts = []
    for item in knowledge[:6]:
        if item.get("type") == "fact":
            facts.append(f"{item.get('subject')} — {item.get('predicate')} — {item.get('object')}")
        elif item.get("type") == "entity":
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            if name:
                facts.append(f"{name}: {description}" if description else name)
    return "Из подтверждённой памяти JARVIS:\n" + "\n".join(
        "- " + fact[:450] for fact in facts) if facts else ""


def excerpt_answer(sources):
    """Non-generative fallback when Gemini is absent or untrustworthy."""
    shown = []
    for source in (sources or [])[:3]:
        url = str(source.get("url") or "")
        if urlsplit(url).scheme != "https":
            continue
        snippet = str(source.get("page_text") or source.get("snippet") or "")
        # Avoid treating external content as instructions or trusted facts.
        title = str(source.get("title") or "Источник")[:120]
        shown.append(f"- {title}: {snippet[:450].strip()}\n  {url}")
    return ("Нашёл внешние материалы, но не могу надёжно обобщить их. "
            "Ниже выдержки для проверки, а не подтверждённые знания:\n" +
            "\n".join(shown)) if shown else "Проверяемых источников пока нет."
