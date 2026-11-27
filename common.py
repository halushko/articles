import json
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

def post(base, path, data, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"{base}{path}: HTTP {exc.code}: {detail}") from exc

def serve(dispatch, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            try:
                url = urlsplit(self.path)
                data = {}
                if self.command == "POST":
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 1_000_000:
                        raise ValueError("Invalid body length")
                    data = json.loads(self.rfile.read(size))
                result = dispatch(self.command, url.path, data)
                if isinstance(result, tuple):
                    content_type, body = result
                    body = body.encode()
                else:
                    content_type, body = "application/json", json.dumps(result, ensure_ascii=False).encode()
                status = 200
            except (ValueError, KeyError, TypeError) as exc:
                status, content_type = 400, "application/json"
                body = json.dumps({"error": str(exc)}).encode()
            except Exception as exc:
                status, content_type = 500, "application/json"
                body = json.dumps({"error": str(exc)}).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        do_GET = do_POST = handle_request
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
