# Serves the demo locally with the same headers as Vercel (see vercel.json).
# "Cross-origin isolation" lets the browser run the AI on several CPU cores instead of one.
#   python serve.py            ->  http://localhost:8000
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    demo = Path(__file__).resolve().parent / "demo"
    print(f"Lantern on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(demo))).serve_forever()
