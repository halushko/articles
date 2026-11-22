"""Conservative speech/text intent routing.

Browser speech recognition returns a transcript, usually *without* a question
mark. It does not expose reliable prosody through the Web Speech API.
Question/command markers are optional; uncertain statements must not mutate
teacher-confirmed memory based solely on Qwen's guess.
"""
import re

# Correct only a small group of common recognition errors for *routing*, not
# in the original transcript, audit log or teacher-confirmed knowledge.
ASR_WORDS = re.compile(r"(?iu)\b(?:завут|зовуд|завутъ|завёт)\b")
PREFIX = re.compile(
    r"(?iu)^\s*(?:(?P<kind>вопрос|вапрос|команда|приказ|"
    r"утверждение|факт)\s*[:.,-]?\s+)"
)
QUESTION_OPEN = re.compile(
    r"(?iu)^\s*(?:а|ну|слушай|скажи|подскажи|пожалуйста|"
    r"уточни|объясни мне)\s+"
)
QUESTION_START = re.compile(
    r"(?iu)^(?:как|кто|что|где|когда|почему|зачем|сколько|"
    r"какой|какая|какое|какие|каким|чей|чья|чьи|чьё|"
    r"разве|неужели|правда\s+ли|можно\s+ли|"
    r"умеешь\s+ли|знаешь\s+ли|можешь\s+ли|" 
    r"ты\s+можешь\s+ли|ты\s+умеешь\s+ли)\b"
)
QUESTION_INSIDE = re.compile(
    r"(?iu)\b(?:как\s+(?:меня|тебя)\s+зовут|"
    r"как\s+зовут\s+(?:меня|тебя)|"
    r"какое\s+у\s+(?:меня|тебя)\s+имя|"
    r"(?:меня|тебя)\s+зовут\s+как|"
    r"як\s+(?:мене|тебе)\s+звати|"
    r"что\s+(?:ты|я)\s+уме(?:ешь|ю))\b"
)
DIRECTIVE = re.compile(
    r"(?iu)\b(?:переименуй|переименовать|поменяй|"
    r"измени|верни\s+(?:себе|мне)|назови\s+себя|"
    r"буду\s+(?:тебя\s+)?звать|тебя\s+буду\s+звать|" 
    r"включи|выключи|отключи|открой|закрой|разблокируй|переключи|"
    r"увімкни|вимкни|відкрий|закрий|запусти|останови|"
    r"подтверди|подтверждаю|отклоняю|отклони|"
    r"не\s+подтверждаю|отменяю|отмена|"
    r"найди|поищи|загугли|выполни|настрой|"
    r"создай|удали|пришли|покажи|расскажи|"
    r"объясни|проверь)\b"
)
EXPLICIT_TEACHING = re.compile(
    r"(?iu)^\s*(?:меня\s+зовут\s+\S+|"
    r"мое\s+имя\s+\S+|моё\s+имя\s+\S+|"
    r"у\s+меня\s+(?:есть\s+)?\S+|"
    r"мой\s+\S+|моя\s+\S+|мои\s+\S+|"
    r"запомни\s+(?:что|это)|"
    r"я\s+(?:хочу|знаю|имею|живу|использую|работаю)\b|"
    r"это\s+(?:новый|новая|новое|моя|мой|моё|наша|наш|"
    r"система|устройство|прибор|технология)\b)"
)
NEGATIVE_CONFIRM = re.compile(
    r"(?iu)^(?:нет|неа|не\s+надо|отмена|отмени|"
    r"отклоняю|отклони|не\s+подтверждаю|"
    r"отклонить|откажись)\s*[.!]*$"
)
POSITIVE_CONFIRM = re.compile(
    r"(?iu)^(?:да(?:\s+подтверждаю)?|ага|"
    r"подтверждаю|подтверди|подтвердить|"
    r"согласен|применяй|сохрани)\s*[.!]*$"
)

def classify(text):
    raw=str(text or "").strip()
    match=PREFIX.match(raw)
    explicit=match.group("kind").casefold() if match else ""
    body=raw[match.end():].strip() if match else raw
    routing=ASR_WORDS.sub("зовут",body)
    routing=re.sub(r"\s+"," ",routing).strip()
    if explicit in ("вопрос","вапрос"):
        kind,reason="question","explicit_spoken_prefix"
    elif explicit in ("команда","приказ"):
        kind,reason="command","explicit_spoken_prefix"
    elif explicit in ("утверждение","факт"):
        kind,reason="statement","explicit_spoken_prefix"
    elif not routing:
        kind,reason="unknown","empty"
    elif DIRECTIVE.search(routing) and not QUESTION_START.match(routing):
        kind,reason="command","imperative_verb"
    elif "?" in routing or QUESTION_START.match(
            QUESTION_OPEN.sub("",routing)) or QUESTION_INSIDE.search(routing):
        kind,reason="question","interrogative_grammar"
    elif EXPLICIT_TEACHING.match(routing):
        kind,reason="statement","explicit_declaration"
    else:
        kind,reason="unknown","no_confident_cue"
    return {"intent":kind,"normalized":routing,
            "explicit_prefix":bool(explicit),
            "confidence":"high" if kind!="unknown" else "low",
            "reason":reason,
            "original":raw,
            "allow_memory_proposal":kind=="statement"}


def confirmation(text):
    """Only a *whole utterance* can be a short voice review response."""
    parsed=classify(text)
    if parsed["intent"]=="question":
        return None
    body=parsed["normalized"]
    if POSITIVE_CONFIRM.fullmatch(body):
        return "confirm"
    if NEGATIVE_CONFIRM.fullmatch(body):
        return "reject"
    return None
