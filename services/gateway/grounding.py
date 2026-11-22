"""Deterministic dialogue introspection, not a second LLM or a truth store.

Questions about JARVIS' own tool execution, prior sources and the journal must
read server records. A language model may phrase verified evidence, but may
never invent whether a tool ran or whether a previous topic was remembered.
"""
import re
from collections import Counter


def _norm(value):
    return str(value or "").casefold().replace("ё","е").strip()


def response_from_records(text, memory):
    """Return (answer, route, evidence) or None if ordinary conversation."""
    t=_norm(text)
    operations=memory.get("operations") or {}
    current=operations.get("current") or {}
    last=operations.get("last_result") or {}
    dialogue=memory.get("dialogue") or {}

    wants_result=bool(re.search(
        r"(?:где|покажи|каков|какой|какие)\b.{0,50}"
        r"(?:результат|итог)\w*.{0,35}(?:поиск|интернет)|"
        r"(?:результат|итог)\w*.{0,20}(?:последнего|прошлого)\s+поиск",t))
    wants_status=bool(re.search(
        r"(?:какую|какие|что)\s+задач\w*.{0,35}(?:сейчас|выполня|делаешь)|"
        r"(?:сейчас|в данный момент).{0,35}(?:выполняешь|ищешь|делаешь)|"
        r"(?:незавершенн\w*|ожидающ\w*)\s+(?:операц|задач)|"
        r"(?:поиск|операция).{0,18}(?:завершен|выполняет|идет|в процессе)",t))
    if wants_status or wants_result:
        if not isinstance(operations,dict):
            operations={}
        pending=current.get("approval_status")=="pending"
        running=current.get("execution_status")=="running"
        if pending:
            answer=("Поиск ожидает разрешения. Запрос ещё НЕ выполнялся. "
                    "Подтверди конкретный поиск: «Разрешаю поиск».")
        elif running:
            answer="Поисковая операция запущена, но завершение ещё не записано. Это не результат поиска."
        elif current.get("approval_status")=="approved" and not current.get("execution_status"):
            answer=("Согласие на поиск записано, но запуск инструмента не зафиксирован. "
                    "Результата нет. Нужно повторить поисковый запрос.")
        elif wants_result and last:
            if last.get("execution_status")=="succeeded":
                answer="Последний поиск завершён. "+str(last.get("answer") or "")[:1700]
                urls=[str(x.get("url")) for x in (last.get("sources") or [])[:3]
                      if isinstance(x,dict) and x.get("url")]
                if urls:
                    answer+="\nИсточники: "+"; ".join(urls)
            else:
                answer=("Последний поиск завершился ошибкой: "+
                        str(last.get("error") or "источники недоступны")[:200]+
                        ". Найденного результата нет.")
        elif wants_status:
            if last:
                answer=("Сейчас активный поиск не выполняется. Последняя операция: "+
                        ("успешно завершена" if last.get("execution_status")=="succeeded"
                         else "завершилась ошибкой")+".")
            else:
                answer="Сейчас активных поисковых операций нет. Выполненного поиска в этом диалоге не зарегистрировано."
        else:
            answer="Результат поиска отсутствует: в этой сессии нет завершённой поисковой операции."
        return answer,"sqlite_operation_status",{"operation_status":operations}

    # A question about YOUR OWN previous turn's provenance is resolved by the
    # journal; Qwen must not supply invented sources, personal data, or web claims.
    is_source=bool(
        re.search(r"\bоткуда\b.{0,100}\b(узнал|знаешь|взял|сведения|информац|источник)\w*",t)
        or re.search(r"\b(?:источник|источники)\s+(?:этой|той|твоей|информац)",t)
        or ("откуда" in t and ("сведения" in t or "информац" in t))
        or re.search(r"\bна основании чего\b",t))
    if is_source:
        speaker=memory.get("speaker") or {}
        if re.search(r"\b(?:мое|моё|моего|моем|моём)\s+им[ея]|(?:меня|моё)\s+зовут",t):
            name=str(speaker.get("display_name") or
                     (memory.get("identity") or {}).get("user") or "")
            if name:
                return ("Я использую имя «"+name+"» из выбранного профиля SQLite. "
                        "Выбранный профиль не подтверждает личность по голосу.",
                        "sqlite_answer_provenance",{"source":"actor_profile",
                                                   "origin":str(speaker.get("origin") or "")[:60]})
        ev=dialogue.get("last_response_evidence") or {}
        kind=ev.get("source_kind")
        if kind=="external_web":
            sources=[x.get("url") for x in ev.get("sources") or []
                     if isinstance(x,dict) and x.get("url")]
            answer="В предыдущем ответе использованы результаты реального поиска."
            if sources:
                answer+=" Источники: "+"; ".join(sources[:3])
        elif kind=="confirmed_sqlite":
            answer="Предыдущий ответ получен из подтверждённых записей SQLite, а не из интернета."
        elif kind=="server_operation":
            answer="Предыдущий ответ сформирован по состоянию сервера, а не найден в интернете."
        elif kind=="unverified_model_answer":
            answer=("Предыдущий ответ сгенерировала модель "+str(ev.get("model") or "Qwen")+
                    " без проверенного источника. Это мог быть ошибочный ответ; "
                    "я не выполнял поиск для его проверки.")
        else:
            answer="В журнале нет проверяемого происхождения предыдущего ответа."
        return answer,"sqlite_answer_provenance",{"last_answer_evidence":ev}

    previous=dialogue.get("previous_topic") or {}
    if (re.search(r"\b(?:о\s+ч[её]м|что)\s+мы\s+(?:с\s+тобой\s+)?говорил\w*",t)
            and re.search(r"\b(?:перед|до|раньше|предыдущ)\b",t)):
        if previous.get("subject"):
            msg="Перед сменой темы мы обсуждали «"+str(previous["subject"])+"»."
            if previous.get("reference"):
                msg+=" В журнале есть вопрос: «"+str(previous["reference"])+"»."
            return msg,"sqlite_dialogue_recall",{"previous_topic":previous}
        return ("В журнале текущей сессии не найдено однозначного перехода "
                "между темами. Не стану придумывать прошлую беседу.",
                "sqlite_dialogue_recall",{"previous_topic":None})
    return None


def looks_like_loop(text):
    """Strong lexical repetition detection: refuse degenerated 3B output."""
    tokens=re.findall(r"(?u)\w+",_norm(text))
    if len(tokens)<35:
        return False
    five=[tuple(tokens[i:i+5]) for i in range(len(tokens)-4)]
    if not five:
        return False
    counts=Counter(five)
    return max(counts.values())>=5 and sum(n-1 for n in counts.values()) >= len(five)*0.20


def claims_unobserved_search(text):
    """Model text must not impersonate an ongoing tool operation."""
    t=_norm(text)
    return bool(re.search(
        r"\b(?:сейчас\s+)?(?:я\s+)?(?:ищу|ищем)\s+(?:информац|данны|в\s+интернете)|"
        r"\b(?:поиск|проверка)\s+(?:уже\s+)?(?:ведется|выполняется|идет|запущен)\b|"
        r"\bкак\s+только\s+найду\b",t))
