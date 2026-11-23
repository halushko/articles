"""Deterministic plan sequencing and evidence-aware response delivery.

Qwen proposes intent/steps. This module makes decisions about which
operations can run, checks prerequisites and never fabricates facts.
Search approval and execution state live in Research's SQLite.
"""
from .evidence import (cited_synthesis, excerpt_answer, query_eligible,
                       reliable_pages, summarize_memory)


def reply(answer, route, **more):
    return {"answer": answer, "route": route, "executed": False,
            "search_executed": False, "sources": [], "model": "Core",
            "tokens": 0, "proposals": [], **more}


def _safe_subject(req, context, query):
    """A proposed query is just text, not permission to expose user context."""
    private_names = context.get("private_names") or []
    return query_eligible(req["text"], query, context.get("task"), private_names)


async def research_answer(req, context, query, call, approval=None):
    """Use one exact approved query; synthesize only after real search evidence."""
    query = str(query).strip()[:240]
    original = {**req, "text": approval.get("question", req["text"])} if approval else req
    safe, why = _safe_subject(original, context, query)
    if not safe:
        return reply("Уточни публичный предмет поиска. Не стану отправлять "
                     "личную информацию или слова, которые не следуют из твоего запроса.",
                     "core_query_clarification", missing=why)

    sid = req.get("session_id") or ""
    operation_id = (approval or {}).get("operation_id")
    if operation_id:
        # Atomic claim prevents repeated clicks/requests from invoking search.
        started = await call("research", "/research/operations/web/start", {
            "session_id": sid, "actor_id": req.get("actor_id") or "",
            "operation_id": operation_id})
        if not started.get("started"):
            return reply("Поиск не запускаю повторно: операция уже обработана "
                         "или разрешение недействительно.",
                         "core_search_already_processed",
                         approval_id=operation_id,
                         operation_state=started)
    outcome = None
    try:
        result = await call("search", "/v1/search",
                            {"query": query, "session_id": sid})
        if not result.get("available") or not result.get("sources"):
            outcome = reply("Поиск не дал проверяемых источников: " +
                            str(result.get("reason") or "источники недоступны"),
                            "core_search_unavailable",
                            search_query=query, search_status=result.get("status"),
                            search_executed=bool(result.get("executed")))
        else:
            sources = result["sources"][:4]
            pages = reliable_pages(sources)
            outcome = reply(excerpt_answer(sources), "core_search_evidence_excerpt",
                            search_query=query, search_status="ok",
                            search_executed=True, sources=sources,
                            evidence_source="public_search_not_confirmed")
            # An unconfigured or unavailable external LLM is not a reason to
            # improvise a recipe from Qwen weights.
            if pages:
                try:
                    answer = await call("gemini", "/v1/synthesize", {
                        "question": query,
                        "sources": pages[:3]})
                    candidate = str(answer.get("answer") or "")
                    if (answer.get("response_usable")
                            and cited_synthesis(candidate, pages[:3])):
                        outcome.update(answer=candidate[:5500],
                                       route="gemini_searxng_evidence",
                                       model=answer.get("model", "Gemini"),
                                       tokens=int(answer.get("tokens") or 0))
                except Exception:
                    # No source-backed synthesis: explicit excerpts instead.
                    outcome["synthesis_status"] = "unavailable"
    except Exception as error:
        outcome = reply("Поисковый сервис недоступен. Не буду придумывать ответ.",
                        "core_search_failed", search_query=query,
                        search_status="failed",
                        search_error=type(error).__name__)
    finally:
        if operation_id:
            success = bool(outcome and outcome.get("search_status") == "ok")
            status = "succeeded" if success else "failed"
            # Use the actual executor outcome, never a Qwen assertion.
            finished = await call("research", "/research/operations/web/finish", {
                "session_id": sid, "actor_id": req.get("actor_id") or "",
                "operation_id": operation_id, "status": status,
                "answer": (outcome or {}).get("answer", "")[:5000],
                "sources": (outcome or {}).get("sources", [])[:5],
                "error": "" if success else (outcome or {}).get("search_error", "search failed")
            })
            if outcome is not None:
                outcome["approval_id"] = operation_id
                outcome["operation_state"] = {"status": status,
                                              "saved": finished.get("saved", False)}
                if not finished.get("saved"):
                    outcome["integrity_warning"] = "Operation outcome not persisted"
    return outcome


async def run_plan(req, context, plan, call):
    """Single bounded pass; no model-generated control flow or recursive replan."""
    if plan.get("status") == "needs_clarification":
        return reply(plan.get("clarifying_question") or "Уточни, пожалуйста, запрос.",
                     "core_needs_clarification", plan=plan)
    if plan.get("status") == "unavailable":
        return reply("Для этой задачи отсутствует разрешённый исполнитель: " +
                     str(plan.get("missing_capability") or "неизвестный инструмент"),
                     "core_capability_unavailable", plan=plan)
    if plan.get("status") != "proposed":
        return reply("Не удалось однозначно составить план. Переформулируй запрос.",
                     "core_plan_invalid", plan=plan)
    steps = plan.get("steps") or []
    supported = {"knowledge.lookup", "knowledge.propose", "web.search", "dialogue.respond"}
    if any(step["capability"] not in supported for step in steps):
        return reply("В плане встретилась операция без проверенного исполнителя.",
                     "core_capability_unavailable", plan=plan)
    web_steps = [s for s in steps if s["capability"] == "web.search"]
    if len(web_steps) > 1:
        return reply("В плане несколько независимых поисков. Уточни, какой нужен первым.",
                     "core_multi_search_clarification", plan=plan)
    if plan.get("intent") == "action":
        return reply("Я распознал запрос на действие, но в этом контуре "
                     "управление внешними системами не подключено.",
                     "core_action_unavailable", plan=plan)

    knowledge = context.get("knowledge") or []
    memory_answer = summarize_memory(knowledge)
    # A recalled confirmed fact is evidence; a user's unverified past
    # utterance and the text generated by Qwen are NOT evidence.
    if plan.get("intent") == "recall" and memory_answer and not web_steps:
        return reply(memory_answer, "core_confirmed_memory",
                     evidence_source="confirmed_sqlite", plan=plan)

    need_web = bool(web_steps) or (
        not knowledge and plan.get("intent") in ("question", "research"))
    if need_web:
        mode = req.get("web_mode")
        if mode == "off":
            return reply("В подтверждённой памяти недостаточно сведений. "
                         "Для проверки внешних источников включи поиск.",
                         "core_research_disabled", plan=plan)
        # A planner-supplied query must survive the provenance check. For a
        # public question with no search step use the original user wording.
        query = (web_steps[0].get("arguments") or {}).get("query") if web_steps else req["text"]
        query = str(query or "").strip()
        safe, why = _safe_subject(req, context, query)
        if not safe:
            return reply("Не уверен, что правильно определил предмет поиска. "
                         "Уточни его: я не буду добавлять выдуманные детали.",
                         "core_query_clarification", missing=why, plan=plan)
        if mode == "ask":
            if not req.get("session_id"):
                return reply("Для разрешения поиска нужен сохранённый диалог.",
                             "core_session_required", plan=plan)
            approval = await call("research", "/research/operations/web/propose", {
                "session_id": req["session_id"],
                "actor_id": req.get("actor_id") or "",
                "query": query, "question": req["text"]})
            return reply("Могу найти информацию по запросу «" + query +
                         "». Разрешаешь отправить этот запрос во внешний поиск?",
                         "core_search_approval_needed",
                         search_status="approval_needed", web_requested=True,
                         web_approval=approval, plan=plan)
        result = await research_answer(req, context, query, call)
        result["plan"] = plan
        if any(s["capability"] == "knowledge.propose" for s in steps):
            result["answer"] += ("\n\nНовое знание не записано. "
                                 "Для сохранения потребуется отдельное подтверждение.")
            result["knowledge_write_pending"] = True
        return result

    if memory_answer:
        return reply(memory_answer, "core_confirmed_memory",
                     evidence_source="confirmed_sqlite", plan=plan)

    if any(s["capability"] == "knowledge.propose" for s in steps):
        return reply("Я услышал новое утверждение, но в этом режиме "
                     "не записываю его как факт без отдельного подтверждения.",
                     "core_learning_review_required", plan=plan)

    return reply("У меня пока нет достаточных подтверждённых сведений. "
                 "Можешь уточнить вопрос или разрешить поиск в интернете?",
                 "core_missing_evidence", plan=plan)
