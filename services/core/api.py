"""Read-only planning API.

Only this service interprets multi-intent requests into proposed steps. It
neither executes tools nor mutates knowledge. Gateway remains the public edge;
Research owns SQLite, LLM owns access to local Ollama.
"""
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .planner import catalog, validate

RESEARCH_URL = os.getenv("RESEARCH_URL", "http://research:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:8000")

app = FastAPI(title="JARVIS Core - Planning", version="0.1.0", docs_url=None)


class PlanIn(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    actor_id: str = Field(default="", max_length=36)
    session_id: str = Field(default="", max_length=60)
    web_mode: str = Field(default="off", pattern="^(off|ask|auto)$")
    profile: str = Field(default="fast", pattern="^(fast|deep)$")


@app.get("/health")
async def health():
    return {"status": "ok", "module": "core", "execution_enabled": False}


@app.post("/v1/plan")
async def propose_plan(req: PlanIn):
    """Untrusted Qwen returns candidates; the server validates every step."""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.post(
                RESEARCH_URL + "/research/planner/context",
                json={"text": req.text, "actor_id": req.actor_id,
                      "session_id": req.session_id})
            response.raise_for_status()
            context = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, "Planning context unavailable: " +
                                type(exc).__name__) from exc

    tools = catalog(context.get("capabilities", []), req.web_mode)
    # Never send capabilities or person/device permissions chosen by a client.
    # Only Research's actual registered routes can appear in this catalog.
    compact = {
        "task": context.get("task") or {},
        "history": (context.get("history") or [])[-6:],
        "knowledge": (context.get("knowledge") or [])[:6],
        "experiences": (context.get("experiences") or [])[:3],
    }
    async with httpx.AsyncClient(timeout=160) as client:
        try:
            response = await client.post(LLM_URL + "/plan", json={
                "text": req.text, "profile": req.profile,
                "context": compact, "capabilities": list(tools.values())})
            response.raise_for_status()
            parsed = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"status": "unavailable", "reason": "Локальный планировщик недоступен",
                    "error_type": type(exc).__name__, "steps": [],
                    "executed": False, "execution_allowed": False}
    outcome = validate(parsed.get("plan"), tools)
    outcome["model"] = parsed.get("model", "")
    outcome["tool_count"] = len(tools)
    outcome["task_context_used"] = bool((compact["task"] or {}).get("context_used"))
    return outcome
