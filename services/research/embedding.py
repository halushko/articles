"""Optional frozen BGE embeddings: feature extractor, not user's online novelty."""
import os
import httpx


async def embed(text: str, model: str) -> list[float] | None:
    if os.getenv("SEMANTIC_ENABLED", "false").lower() not in {"true","1","yes"}:
        return None
    async with httpx.AsyncClient(timeout=35.0) as client:
        response = await client.post(os.getenv("OLLAMA_URL", "http://ollama:11434") + "/api/embed",
                                     json={"model": model, "input": text, "truncate": True})
        response.raise_for_status()
    vectors = response.json().get("embeddings", [])
    return vectors[0] if vectors and isinstance(vectors[0], list) else None
