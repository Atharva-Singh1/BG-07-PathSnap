"""FAKE API for testing the UI without laptop 1. NOT used in the demo.

It serves synthetic trips (ids start with "MOCK-") with made-up coordinates and
made-up metrics, following docs/api_contract.md. It uses no dataset and no model.

Run:  python ui/mock_api.py --port 8000
"""

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_TRIPS = [f"MOCK-{i:02d}" for i in range(1, 4)]


def fake_prediction(trip_id: str) -> dict:
    k = MOCK_TRIPS.index(trip_id) + 1
    lon0, lat0 = 8.65, 49.87
    pts = [[lon0 + 0.0004 * i, lat0 + 0.0015 * math.sin(i / (8 * k))] for i in range(120)]
    route = [[[lon0 + 0.0004 * i, lat0 + 0.0015 * math.sin(i / (8 * k))],
              [lon0 + 0.0004 * (i + 10), lat0 + 0.0015 * math.sin((i + 10) / (8 * k))]]
             for i in range(0, 110, 10)]
    return {"trip_id": trip_id, "model": "MOCK (fake data)", "n_points": len(pts),
            "gps_points": pts, "predicted_route": route,
            "metrics": {"point_accuracy": 0.5, "route_continuity": 0.5, "macro_f1": 0.5,
                        "unmatched_rate": 0.0},
            "inference_time_s": 0.01}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": "MOCK (fake data)"})
        elif self.path == "/trips":
            self._send(200, {"trips": MOCK_TRIPS})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, {"detail": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            trip_id = body["trip_id"]
        except (ValueError, KeyError):
            return self._send(422, {"detail": "body must be {\"trip_id\": ...}"})
        if trip_id not in MOCK_TRIPS:
            return self._send(404, {"detail": "unknown trip_id"})
        self._send(200, fake_prediction(trip_id))

    def log_message(self, fmt, *args):
        print(f"[MOCK API {time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")


def make_server(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fake PathSnap API for UI testing only.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"*** MOCK API (fake data, not the real model) on http://{a.host}:{a.port} ***")
    make_server(a.port, a.host).serve_forever()
