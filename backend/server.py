"""Serves the BLACKOUT UI and the search API from the Python standard library.

    python server.py            # http://127.0.0.1:8000, opens a browser
    python server.py --port 9000 --no-browser

Everything is served from disk: the page, the search results, and the manual PDFs. There
is no dependency on the network, and the server binds to localhost only, so nothing off
this machine can reach it.

Routes
    GET  /                    frontend/index.html (and any other file beside it)
    GET  /api/stats           manuals, counts and threshold for the status bar
    POST /api/search          {"question": str, "k": int} -> search()
    GET  /api/protocol?id=    get_protocol(), for a result the user opened
    GET  /manuals/<file>.pdf  the manual itself, so a citation can be checked
"""
import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import search as backend

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT.parent / "frontend"
DOCS = ROOT.parent / "Documents"
MAX_BODY = 64 * 1024

# Short label and source PDF for each manual, by the name the chunks carry. A manual added
# without an entry here still works; it is cited by its full name and has no PDF link.
MANUALS = {
    "Emergency Response Guidebook 2024": ("Hazmat", "2024EmergencyResponseGuide.pdf"),
    "NIOSH Pocket Guide to Chemical Hazards": ("Chemical", "CDCPocketGuide.pdf"),
    "West Virginia Statewide EMS Protocols": ("Medical", "WestVirginiaEmergencyGuide.pdf"),
}


def manual_table() -> list[dict]:
    """Every manual in the index: full name, short label, PDF route, and how much of the
    corpus it holds. The UI shows the totals in the status bar and the labels on citations."""
    counts: dict[str, dict] = {}
    for c in backend._engine().chunks:
        m = counts.setdefault(c["manual"], {"chunks": 0, "protocols": set()})
        m["chunks"] += 1
        m["protocols"].add(c["protocol_id"])
    out = []
    for name, m in sorted(counts.items()):
        short, pdf = MANUALS.get(name, (name, None))
        out.append({
            "manual": name,
            "short": short,
            "pdf": f"/manuals/{pdf}" if pdf and (DOCS / pdf).exists() else None,
            "chunks": m["chunks"],
            "protocols": len(m["protocols"]),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "BLACKOUT"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- replies

    def send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send(status, body, "application/json; charset=utf-8")

    def send_file(self, path: Path) -> None:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self.send(200, path.read_bytes(), ctype)

    # ---------------------------------------------------------------- routing

    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path
        try:
            if path == "/api/stats":
                chunks = backend._engine().chunks
                return self.send_json({
                    "manuals": manual_table(),
                    "chunks": len(chunks),
                    "protocols": len(backend._engine().protocols),
                    "threshold": backend.THRESHOLD,
                    "model": backend._engine().model_name,
                })
            if path == "/api/protocol":
                pid = (parse_qs(route.query).get("id") or [""])[0]
                protocol = backend.get_protocol(pid)
                if protocol is None:
                    return self.send_json({"error": f"no protocol {pid!r}"}, 404)
                return self.send_json(protocol)
            if path.startswith("/manuals/"):
                return self.serve_static(DOCS, path[len("/manuals/"):])
            return self.serve_static(FRONTEND, path.lstrip("/") or "index.html")
        except Exception as exc:  # a broken request must not take the server down mid-demo
            self.log_error("%s", exc)
            self.send_json({"error": str(exc)}, 500)

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/search":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self.send_json({"error": "request too large"}, 413)
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", ""))
            k = int(payload.get("k", backend.TOP_K))
            self.send_json(backend.search(question, max(1, min(k, 10))))
        except (ValueError, TypeError) as exc:
            self.send_json({"error": f"bad request: {exc}"}, 400)
        except Exception as exc:
            self.log_error("%s", exc)
            self.send_json({"error": str(exc)}, 500)

    def serve_static(self, base: Path, name: str) -> None:
        """A file from one directory, and only from it: the path is resolved and checked to
        be inside `base`, so "../" cannot walk out of it."""
        target = (base / name).resolve()
        if base.resolve() not in target.parents or not target.is_file():
            return self.send_json({"error": "not found"}, 404)
        self.send_file(target)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (FRONTEND / "index.html").exists():
        print(f"missing {FRONTEND / 'index.html'}", file=sys.stderr)
        return 1

    # load the index and the model before the first question, not during it
    print("loading index and embedding model ...")
    try:
        engine = backend._engine()
        engine.embed(["warm up"])
    except (RuntimeError, OSError) as exc:
        print(f"\n{exc}\n\nRun `python ingest.py` then `python index.py` first.", file=sys.stderr)
        return 1
    missing = [m["manual"] for m in manual_table() if m["pdf"] is None]
    if missing:
        print(f"note: no PDF link for {', '.join(missing)}")
    print(f"{len(engine.chunks)} chunks in {len(engine.protocols)} protocols, "
          f"model {engine.model_name}")

    url = f"http://{args.host}:{args.port}/"
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"BLACKOUT is at {url}   (ctrl-c to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
