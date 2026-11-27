import os
from common import post, serve

MODEL = os.getenv("MODEL", "qwen3:4b")
BASE = os.getenv("OLLAMA_URL", "http://ollama:11434")

def dispatch(method, path, data):
    if method != "POST" or path != "/ask":
        raise ValueError("Unknown endpoint")
    response = post(BASE, "/api/chat", {
        "model": MODEL,
        "messages": [{"role": "system", "content":
            "You are a scientist studying an unknown simulated system. "
            "Do not assume hidden laws. Use only supplied observations and hypotheses. "
            "Return one JSON object matching the requested fields. Be concise."},
            {"role": "user", "content": data["prompt"]}],
        "format": "json", "stream": False, "think": False,
        "options": {"temperature": 0, "seed": data["seed"], "num_ctx": 4096, "num_predict": 900}
    })
    import json
    answer = json.loads(response["message"]["content"])
    if not isinstance(answer, dict):
        raise ValueError("Expected a JSON object")
    return {"answer": answer, "model": response.get("model", MODEL),
            "tokens": response.get("eval_count"), "prompt_tokens": response.get("prompt_eval_count")}

if __name__ == "__main__":
    serve(dispatch)
