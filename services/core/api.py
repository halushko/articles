"""JARVIS core: propose plans, enforce policy and coordinate specialized executors.

Only local Qwen interprets natural language. Core never asks Qwen to make
up factual answers. Research owns the world and durable search approvals;
Search owns SearXNG calls; Gemini only synthesizes real public evidence.
"""
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .planner import catalog, validate
from .runtime import reply, research_answer, run_plan

URLS = {
    "research": os.getenv("RESEARCH_URL", "http://research:8000"),
    "llm": os.getenv("LLM_URL", "http://llm:8000"),
    "search": os.getenv("SEARCH_URL", "http://search:8000"),
    "gemini": os.getenv("GEMINI_URL", "http://gemini:8000"),
}
app = FastAPI(title="JARVIS Core", version="0.2.0", docs_url=None)


class PlanIn(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    actor_id: str = Field(default="", max_length=36)
    session_id: str = Field(default="", max_length=60)
    web_mode: str = Field(default="off", pattern="^(off|ask|auto)$")
    profile: str = Field(default="fast", pattern="^(fast|deep)$")


class ExecuteIn(PlanIn):
    approval_id: str = Field(default="", max_length=36)
    allow_web_search: bool = False


async def call(service, endpoint, payload):
    if service not in URLS or not endpoint.startswith("/"):
        raise ValueError("Unsupported internal endpoint")
    timeout = 250 if service == "llm" else 65 if service == "gemini" else 35
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(URLS[service] + endpoint, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as error:
        raise HTTPException(502, service + " returned HTTP " +
                            str(error.response.status_code)) from error
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(503, service + " unavailable: " +
                            type(error).__name__) from error


async def planning(req):
    """One context read, one bounded Qwen call, one deterministic validator."""
    context = await call("research", "/research/planner/context", {
        "text": req.text, "actor_id": req.actor_id, "session_id": req.session_id})
    capabilities = catalog(context.get("capabilities", []), req.web_mode)
    compact = {
        "task": context.get("task") or {},
        "history": (context.get("history") or [])[-6:],
        "knowledge": (context.get("knowledge") or [])[:6],
        "experiences": (context.get("experiences") or [])[:3],
    }
    try:
        parsed = await call("llm", "/plan", {
            "text": req.text, "profile": req.profile,
            "context": compact, "capabilities": list(capabilities.values())})
    except HTTPException:
        return context, {"status": "unavailable", "reason": "Qwen недоступна",
                         "steps": [], "executed": False,
                         "execution_allowed": False}
    outcome = validate(parsed.get("plan"), capabilities)
    outcome["model"] = parsed.get("model", "")
    outcome["tool_count"] = len(capabilities)
    outcome["task_context_used"] = bool((compact["task"] or {}).get("context_used"))
    return context, outcome


@app.get("/health")
async def health():
    return {"status": "ok", "module": "core",
            "execution_enabled": True,
            "physical_actions_enabled": False}


@app.post("/v1/plan")
async def propose_plan(req: PlanIn):
    """Preview without execution or side effects."""
    _, plan = await planning(req)
    return plan


@app.post("/v1/chat")
async def execute(req: ExecuteIn):
    """One turn, one validated plan, at most one public search.

    Before planning, the deterministic operation manager handles approvals
    (including a voice approval with no punctuation). Approval is bound to
    the original session, actor, query and TTL in Research SQLite.
    """
    original = req.model_dump()
    if req.session_id:
        approval = await call("research", "/research/operations/web/resolve", {
            "session_id": req.session_id, "actor_id": req.actor_id,
            "phrase": req.text if not req.approval_id else "",
            "approval_id": req.approval_id,
            "source": ("button" if req.approval_id and req.allow_web_search else "voice"),
        })
        if approval.get("handled"):
            if approval.get("status") == "approved":
                approved_query = str(approval.get("query") or "")
                # Rebuild *read-only* context around the original request;
                # don't plan again or let the model change approved arguments.
                context = await call("research", "/research/planner/context", {
                    "text": approval.get("question") or approved_query,
                    "actor_id": req.actor_id, "session_id": req.session_id})
                return await research_answer(original, context, approved_query,
                                             call, approval=approval)
            return reply(approval.get("answer") or
                         "Не удалось применить разрешение на поиск.",
                         "core_search_approval_" + str(approval.get("status") or "invalid"),
                         operation_state=approval)

    context, plan = await planning(req)
    if plan.get("reason") == "Qwen недоступна":
        return reply("Локальная модель распознавания намерения недоступна. "
                     "Ничего не выполняю.", "core_planner_unavailable")
    return await run_plan(original, context, plan, call)
