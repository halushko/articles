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
                            and cited_synthesis(candidate, pages[:3], query)):
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
    """Execute a validated DAG once, in its declared dependency order.

    Only known capabilities have handlers. A paused approval leaves dependent
    steps untouched. No recursion, speculative writes or concurrent branches.
    """
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
    if any(s["capability"] not in supported for s in steps):
        return reply("В плане есть операция без проверенного исполнителя.",
                     "core_capability_unavailable", plan=plan)
    if sum(s["capability"] == "web.search" for s in steps) > 1:
        return reply("План требует нескольких поисков. Уточни, какой нужен первым.",
                     "core_multi_search_clarification", plan=plan)
    if plan.get("intent") == "action":
        return reply("Я распознал запрос на действие, но управление внешними "
                     "системами пока не подключено.",
                     "core_action_unavailable", plan=plan)

    knowledge = [item for item in context.get("knowledge", [])
                 if item.get("verified")]
    memory_answer = summarize_memory(knowledge)
    trace, completed = [], set()
    web_result = None

    for item in steps:
        sid, capability = item["id"], item["capability"]
        dependencies = item.get("depends_on") or []
        if any(dep not in completed for dep in dependencies):
            return reply("План содержит зависимость от ещё не выполненной операции.",
                         "core_dependency_blocked", plan=plan,
                         execution_trace=trace)

        if capability == "knowledge.lookup":
            trace.append({"step": sid, "capability": capability,
                          "status": "succeeded" if knowledge else "empty"})
            completed.add(sid)
            continue

        if capability == "web.search":
            mode = req.get("web_mode")
            if mode == "off":
                trace.append({"step": sid, "capability": capability, "status": "disabled"})
                return reply("Поиск отключён. Недостающие факты не буду придумывать.",
                             "core_research_disabled", plan=plan,
                             execution_trace=trace)
            query = str((item.get("arguments") or {}).get("query") or "").strip()
            ok, why = _safe_subject(req, context, query)
            if not ok:
                trace.append({"step": sid, "capability": capability, "status": "blocked"})
                return reply("Qwen предложила неподтверждённую формулировку поиска. "
                             "Уточни предмет, чтобы не искать выдуманные детали.",
                             "core_query_clarification", missing=why, plan=plan,
                             execution_trace=trace)
            if mode == "ask":
                if not req.get("session_id"):
                    return reply("Для разрешения поиска нужен сохранённый диалог.",
                                 "core_session_required", plan=plan,
                                 execution_trace=trace)
                approval = await call("research", "/research/operations/web/propose", {
                    "session_id": req["session_id"],
                    "actor_id": req.get("actor_id") or "",
                    "query": query, "question": req["text"]})
                trace.append({"step": sid, "capability": capability,
                              "status": "waiting_for_approval"})
                return reply("Для проверки нужен публичный поиск «" + query +
                             "». Разрешаешь выполнить именно его?",
                             "core_search_approval_needed", plan=plan,
                             execution_trace=trace, search_status="approval_needed",
                             web_requested=True, web_approval=approval)
            web_result = await research_answer(req, context, query, call)
            succeeded = web_result.get("search_status") == "ok"
            trace.append({"step": sid, "capability": capability,
                          "status": "succeeded" if succeeded else "failed"})
            if not succeeded:
                web_result["plan"] = plan
                web_result["execution_trace"] = trace
                return web_result
            completed.add(sid)
            continue

        if capability == "knowledge.propose":
            trace.append({"step": sid, "capability": capability,
                          "status": "requires_separate_review"})
            # A proposal to remember something is not a database write and
            # cannot satisfy a dependent step's prerequisite.
            if web_result:
                web_result["answer"] += ("\n\nЭти сведения ещё не сохранены в памяти. "
                                         "Для этого потребуется отдельное подтверждение.")
                web_result["knowledge_write_pending"] = True
                web_result["plan"] = plan
                web_result["execution_trace"] = trace
                return web_result
            return reply("Я понял просьбу научиться новому, но пока не записываю "
                         "этот факт без отдельного подтверждения.",
                         "core_learning_review_required", plan=plan,
                         execution_trace=trace)

        if capability == "dialogue.respond":
            trace.append({"step": sid, "capability": capability, "status": "completed"})
            completed.add(sid)
            if web_result:
                web_result["plan"] = plan
                web_result["execution_trace"] = trace
                return web_result
            if memory_answer:
                return reply(memory_answer, "core_confirmed_memory",
                             evidence_source="confirmed_sqlite", plan=plan,
                             execution_trace=trace)
            # A response step without evidence is not permission for Qwen
            # to fill in content from its pretrained weights.
            continue

    if web_result:
        web_result["plan"] = plan
        web_result["execution_trace"] = trace
        return web_result
    if memory_answer:
        return reply(memory_answer, "core_confirmed_memory",
                     evidence_source="confirmed_sqlite", plan=plan,
                     execution_trace=trace)

    # Deterministic missing-evidence fallback. Still uses the exact original
    # question, never a fabricated query or a hidden Gemini call.
    if plan.get("intent") in ("question", "research"):
        mode = req.get("web_mode")
        if mode == "off":
            return reply("Подтверждённых сведений недостаточно. "
                         "Разреши исследование или уточни вопрос.",
                         "core_research_disabled", plan=plan,
                         execution_trace=trace)
        query = req["text"]
        ok, why = _safe_subject(req, context, query)
        if not ok:
            return reply("Уточни публичный предмет поиска. Не стану отправлять "
                         "в интернет личные или неоднозначные сведения.",
                         "core_query_clarification", plan=plan, missing=why,
                         execution_trace=trace)
        if mode == "ask":
            if not req.get("session_id"):
                return reply("Для поиска нужен сохранённый диалог.",
                             "core_session_required", plan=plan)
            approval = await call("research", "/research/operations/web/propose", {
                "session_id": req["session_id"], "actor_id": req.get("actor_id") or "",
                "query": query, "question": req["text"]})
            trace.append({"step": "evidence_fallback", "capability": "web.search",
                          "status": "waiting_for_approval"})
            return reply("Не нашёл ответ в памяти. Разрешаешь поиск «" + query + "»?",
                         "core_search_approval_needed", plan=plan,
                         execution_trace=trace, web_approval=approval,
                         search_status="approval_needed", web_requested=True)
        web_result = await research_answer(req, context, query, call)
        trace.append({"step": "evidence_fallback", "capability": "web.search",
                      "status": "succeeded" if web_result.get("search_status") == "ok"
                      else "failed"})
        web_result["plan"] = plan
        web_result["execution_trace"] = trace
        return web_result

    return reply("Пока недостаточно проверенных данных. "
                 "Можешь объяснить или уточнить запрос?",
                 "core_missing_evidence", plan=plan, execution_trace=trace)
