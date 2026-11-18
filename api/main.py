"""FastAPI backend for the PathSnap demo UI."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_data
from src.evaluate import load_matcher, score_trip

logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _request_log_handler = logging.StreamHandler()
    _request_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_request_log_handler)
DEFAULT_MATCHER = "src.hmm_matcher:match"


class PredictRequest(BaseModel):
    trip_id: str


class RequestLoggingMiddleware:
    """Log one structured line for each HTTP request, including trip_id."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        request_body = bytearray()
        status_code = 500

        async def logged_receive():
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        async def logged_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, logged_receive, logged_send)
        finally:
            trip_id: Any = "-"
            if request_body:
                try:
                    trip_id = json.loads(request_body.decode("utf-8")).get("trip_id", "-")
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    trip_id = "-"
            path = scope.get("path", "")
            if path != "/predict":
                trip_id = "-"
            duration = time.perf_counter() - started
            logger.info(
                "time=%s method=%s path=%s trip_id=%s status_code=%d duration_s=%.6f",
                datetime.now(timezone.utc).isoformat(), scope.get("method", ""),
                path, trip_id, status_code, duration,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = os.environ.get("PATHSNAP_DATA_DIR")
    if not data_dir:
        raise RuntimeError("PATHSNAP_DATA_DIR must point to one trip partition folder")
    model_spec = os.environ.get("PATHSNAP_MATCHER", DEFAULT_MATCHER)
    app.state.trips = load_data(data_dir)
    app.state.model_spec = model_spec
    app.state.matcher = load_matcher(model_spec)
    logger.info("Loaded %d trips; matcher=%s", len(app.state.trips), model_spec)
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)


def _plain_metric(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.model_spec}


@app.get("/trips")
def trips():
    return {"trips": sorted(app.state.trips)}


@app.post("/predict")
def predict(request: PredictRequest):
    trip_id = request.trip_id
    trip_data = app.state.trips.get(trip_id)
    if trip_data is None:
        raise HTTPException(status_code=404, detail="unknown trip_id")

    trace = np.asarray(trip_data["trace"], dtype=float)
    network = trip_data["road_network"]
    try:
        started = time.perf_counter()
        pred = np.asarray(app.state.matcher(trace, network), dtype=np.int64).reshape(-1)
        inference_time_s = time.perf_counter() - started

        trip_for_scoring = {
            "trip_id": trip_id,
            "track": trace,
            "nodes": network["nodes"],
            "arcs": network["arcs"],
            "route": trip_data["route"],
        }
        scored = score_trip(trip_for_scoring, pred, inference_time_s=inference_time_s)
        if scored.get("status") != "ok":
            raise ValueError(scored.get("error") or "matcher returned an invalid prediction")

        arcs = np.asarray(network["arcs"], dtype=np.int64)
        nodes = np.asarray(network["nodes"], dtype=float)
        kept = [int(arc_id) for arc_id in pred if int(arc_id) != -1]
        collapsed = [arc_id for i, arc_id in enumerate(kept)
                     if i == 0 or arc_id != kept[i - 1]]
        predicted_route = [
            [nodes[int(arcs[arc_id, 0])].astype(float).tolist(),
             nodes[int(arcs[arc_id, 1])].astype(float).tolist()]
            for arc_id in collapsed
        ]

        metric_names = ("point_accuracy", "route_continuity", "macro_f1", "unmatched_rate")
        metrics = {name: _plain_metric(scored[name]) for name in metric_names}
        return {
            "trip_id": trip_id,
            "model": app.state.model_spec,
            "n_points": int(len(trace)),
            "gps_points": trace[:, :2].astype(float).tolist(),
            "predicted_route": predicted_route,
            "metrics": metrics,
            "inference_time_s": float(inference_time_s),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
