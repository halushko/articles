"""SearXNG executor with its own operational search journal, no world memory."""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.research.web_research import FreeWebResearch


class SearchStore:
    """Only rate-limit/audit data; never JARVIS's authoritative knowledge."""
    def __init__(self, path):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=20)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()


app = FastAPI(title="JARVIS Search Executor", version="0.1", docs_url=None)
search = FreeWebResearch(SearchStore(os.getenv("SEARCH_DB", "/data/search.sqlite3")))


class SearchIn(BaseModel):
    query: str = Field(min_length=3, max_length=240)
    session_id: str = Field(default="", max_length=60)


@app.get("/health")
def health():
    return {"status": "ok", "module": "search", "executor": "public_web.search"}


@app.post("/v1/search")
async def web_search(req: SearchIn):
    # Search is internal only. Public gateway + Core enforce user policy and
    # approvals BEFORE requesting this endpoint. No caller-supplied URLs.
    return await search.search(req.query, req.session_id)
