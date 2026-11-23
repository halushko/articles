"""Deliberately conservative boundary before sending a question to Google.

This is NOT a complete DLP system. The only payload permitted across this
boundary is a short public question; session history, identity and world state
must never be included in the Gemini request.
"""
import re

PRIVATE = [
    r"(?iu)\b(?:мой|моя|моё|мои|моего|моем|меня|наша|наши|наш|"
    r"твой|твоя|твоё|тебя|мої|мій|моя|наш|наша|мене|мені)\b",
    r"(?iu)\b(?:я|мы|нас|нам|свой|своя|своё|свои|свого|нашому)\b",
    r"(?iu)\b(?:дим\w*|дмитр\w*|над\w*|надежд\w*|наді\w*|наскулил\w*|мелк\w*|игнат\w*)\b",
    r"(?iu)\b(?:home\s*assistant|ha\s*api|домашн\w*\s+"
    r"(?:сет\w*|устройств\w*|автоматизац\w*)|"
    r"парол\w*|токен\w*|секрет\w*|api.?key|api.?ключ|"
    r"паспорт\w*|диагноз\w*|болезн\w*|адрес\w*|"
    r"телефон\w*|геолокац\w*|координат\w*)\b",
    r"(?iu)\b(?:\d{1,3}\.){3}\d{1,3}\b",
    r"(?iu)[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    r"(?iu)\b(?:https?://|ssh://|smb://|file://)\S+",
]


def public_only(*items):
    return all(
        2 <= len(str(value).strip()) <= 3000
        and not any(re.search(pattern, str(value)) for pattern in PRIVATE)
        for value in items
    )


def recipe_request(text):
    """Direct external-knowledge route only for ordinary cooking questions.

    Other topics stay on the existing local memory/Qwen path unless the user
    explicitly asks for search or the local answer admits uncertainty.
    """
    return bool(re.match(
        r"(?iu)^\s*(?:(?:дай|подскажи|напомни|расскажи|"
        r"покажи)\s+(?:мне\s+)?)?"
        r"(?:рецепт\b|как\s+(?:приготовить|готовить|испечь)\b)",
        text.strip(),
    ))


def model_abstained(text):
    return bool(re.search(
        r"(?iu)(?:\bне\s+знаю\b|\bне\s+могу\s+(?:ответить|"
        r"подтвердить|проверить|подсказать)\b|"
        r"\bнет\s+(?:достоверной|проверенной|достаточной)\s+информации\b|"
        r"\bнужн\w*\s+(?:проверить|внешние\s+источники)\b)",
        text,
    ))


def public_information_request(text: str, intent: str = "") -> bool:
    """Conservative standalone public-knowledge request, not arbitrary dialogue.

    The intent classifier may see a spoken question without punctuation.
    A positive public-question shape is still required before cloud routing.
    """
    if not public_only(text):
        return False
    phrase = text.strip()
    if recipe_request(phrase):
        return True
    return bool(re.match(
        r"(?iu)^(?:"
        r"что\s+(?:такое|означает|будет|произойд[её]т|делать\s+если)\b|"
        r"кто\s+(?:такой|такая|такие|был|была|есть)\b|"
        r"как\s+(?:сделать|устроен|работает|готовить|приготовить|"
        r"устроена|появил|возник|научиться|выбрать|правильно|можно)\b|"
        r"почему\b|зачем\b|когда\b|где\b|сколько\b|"
        r"каков\b|какой\b|какая\b|какие\b|можно\s+ли\b|"
        r"объясни\b|расскажи\b|подскажи\s+(?:как|почему|что)\b|"
        r"what\b|why\b|how\b|when\b|where\b|who\b|"
        r"що\s+таке\b|чому\b|як\b|коли\b"
        r")", phrase))
