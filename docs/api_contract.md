# PathSnap API contract (laptop 1 ↔ laptop 2)

The UI (laptop 2, `ui/app.py`) talks to the FastAPI backend (laptop 1) only through these three
endpoints. Every body is JSON. The default base URL is `http://<laptop-1-ip>:8000`.

> **Ground truth never leaves the server.** No response may contain the reference route,
> reference labels or anything derived from them other than the metrics below. The model never
> sees them either; the server uses them only to compute the metrics after prediction
> (`src.evaluate.score_trip`).

## `GET /health`

```json
{"status": "ok", "model": "random_forest + HMM"}
```

## `GET /trips`

The trip ids the evaluator can choose from (the 10 evaluation trips in the final demo).

```json
{"trips": ["00000000", "00000007", "..."]}
```

## `POST /predict`

Request:

```json
{"trip_id": "00000000"}
```

Response `200`:

```json
{
  "trip_id": "00000000",
  "model": "random_forest + HMM",
  "n_points": 2503,
  "gps_points": [[8.5952, 49.8440], [8.5953, 49.8444]],
  "predicted_route": [
    [[8.5950, 49.8438], [8.5957, 49.8449]],
    [[8.5957, 49.8449], [8.5961, 49.8460]]
  ],
  "metrics": {
    "point_accuracy": 0.93,
    "route_continuity": 0.97,
    "macro_f1": 0.88,
    "unmatched_rate": 0.0
  },
  "inference_time_s": 1.42
}
```

| Field | Meaning |
|---|---|
| `gps_points` | the trip's GPS points as `[lon, lat]`, in order |
| `predicted_route` | one `[[lon, lat], [lon, lat]]` line per **predicted** arc (start node → end node), in driving order, consecutive repeats removed |
| `metrics` | computed **on the server** by `src.evaluate.score_trip` from the prediction and the server-side reference |
| `inference_time_s` | server time for the prediction pipeline only (not network time) |

Errors: `404 {"detail": "unknown trip_id"}`, `422` for a malformed request, `500 {"detail": "..."}`
if the pipeline fails. The server should log every request (method, path, trip id, status,
time); the logs are shown to the evaluator.

The values in the examples above are placeholders that illustrate the format. They are not
results.
