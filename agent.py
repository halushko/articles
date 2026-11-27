import json
import os
from common import post, serve

MODEL = os.getenv("MODEL", "qwen3:4b")
BASE = os.getenv("OLLAMA_URL", "http://ollama:11434")
SYSTEM = ("You are a scientist studying an unknown simulated system. "
          "Do not assume hidden laws. Use only supplied observations and hypotheses. "
          "Return one JSON object matching the requested schema. Be concise.")

def dispatch(method, path, data):
    if method != "POST" or path != "/ask":
        raise ValueError("Unknown endpoint")
    request = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": data["prompt"]}],
        "format": data["schema"], "stream": False, "think": False,
        "options": {"temperature": 0, "seed": data["seed"], "num_ctx": 8192, "num_predict": 1800}
    }
    try:
        response = post(BASE, "/api/chat", request)
    except Exception as exc:
        return {"request": request, "error": str(exc)}
    result = {"request": request, "raw_response": response,
              "model": response.get("model", MODEL),
              "tokens": response.get("eval_count"), "prompt_tokens": response.get("prompt_eval_count")}
    try:
        answer = json.loads(response["message"]["content"])
        if not isinstance(answer, dict):
            raise ValueError("Expected a JSON object")
        result["answer"] = answer
    except (ValueError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result

if __name__ == "__main__":
    serve(dispatch)
