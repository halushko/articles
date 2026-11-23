"""Gemini API as a restricted external answer provider, never a home-action executor.

Only caller-supplied public questions go to Google; no SQLite, device state,
conversation history, credentials, or actor profiles are included.
"""
import os
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
SYSTEM = (
    "Ты внешний информационный помощник JARVIS. Отвечай на языке вопроса. "
    "Если используешь Google Search, опирайся на найденные материалы, "
    "не изобретай источники, ссылки, ингредиенты или действия. "
    "Если информации недостаточно, прямо скажи об этом. "
    "Ты НЕ подключён к Home Assistant, не знаешь частных данных пользователя "
    "и не можешь выполнять физические действия или сохранять память."
)


def configured():
    return os.getenv("GEMINI_ENABLED", "true").lower() in ("true", "1", "yes") and bool(
        os.getenv("GEMINI_API_KEY", "").strip()
    )


def request_payload(question: str, search: bool):
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1400},
    }
    if search:
        payload["tools"] = [{"google_search": {}}]
    return payload


def parse_response(data: dict, search: bool, model: str, elapsed: float):
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    answer = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)
                       and isinstance(p.get("text"), str)).strip()
    grounding = candidate.get("groundingMetadata") or {}
    sources = []
    seen = set()
    for chunk in grounding.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        url = str(web.get("uri") or "").strip()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or url in seen:
            continue
        seen.add(url)
        sources.append({"title": str(web.get("title") or parsed.hostname)[:180],
                        "url": url[:1800], "status": "google_grounding"})
        if len(sources) == 8:
            break
    queries = grounding.get("webSearchQueries") or []
    searched = bool(sources or queries) if search else False
    complete = candidate.get("finishReason") not in ("MAX_TOKENS", "SAFETY", "RECITATION")
    if search and not sources:
        answer = ("Gemini не предоставил проверяемых ссылок на источники. "
                  "Не буду выдавать его текст за подтверждённый результат поиска.")
    elif not answer:
        answer = "Gemini не вернул текстовый ответ."
    elif not complete:
        answer += "\n\n⚠️ Ответ Gemini неполный."
    usage = data.get("usageMetadata") or {}
    return {
        "answer": answer[:12000], "route": "gemini_google_grounded" if sources and search
                 else "gemini_search_unverified" if search else "gemini_general",
        "model": model, "provider": "google_gemini",
        "tokens": int(usage.get("candidatesTokenCount") or 0),
        "prompt_tokens": int(usage.get("promptTokenCount") or 0),
        "total_ms": round(elapsed * 1000, 2),
        "sources": sources, "search_executed": searched,
        "search_verified": bool(sources) if search else False,
        "search_status": "ok" if sources and search else
                         "unverified" if search else "not_requested",
        "response_complete": bool(complete),
        "executed": False, "proposals": [],
        "prompt_trace": {"origin": "gemini_public_question",
                         "answer_chars": len(answer),
                         "search_tool_requested": search,
                         "history_supplied": False,
                         "private_memory_supplied": False,
                         "source_count": len(sources)},
    }


async def generate(question: str, search: bool = False):
    if not configured():
        raise HTTPException(503, "Gemini не настроен: укажи GEMINI_API_KEY в .env")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,100}", model):
        raise HTTPException(500, "Некорректное имя модели Gemini")
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("GEMINI_TIMEOUT", "50"))) as client:
            response = await client.post(
                GEMINI_URL + model + ":generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"],
                         "Content-Type": "application/json"},
                json=request_payload(question, search),
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status == 429:
            raise HTTPException(503, "Gemini: достигнут лимит бесплатных запросов") from None
        if status in (401, 403):
            raise HTTPException(503, "Gemini: API-ключ не принят или нет доступа") from None
        if status == 404:
            raise HTTPException(503, "Gemini: модель недоступна этому API-ключу; проверь GEMINI_MODEL") from None
        raise HTTPException(503, "Gemini API: ошибка HTTP " + str(status)) from None
    except httpx.HTTPError as error:
        raise HTTPException(503, "Gemini API недоступен: " + type(error).__name__) from None
    try:
        return parse_response(response.json(), search, model,
                              time.perf_counter() - start)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(502, "Gemini вернул некорректный ответ") from None
