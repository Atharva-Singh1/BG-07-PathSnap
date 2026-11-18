"""Evaluation for PathSnap map matchers.

Scores any matcher that follows the shared interface

    match(trace, road_network) -> list[int]   # one arc id per GPS point

against the reference routes shipped with the dataset.

Reference labels
----------------
The dataset gives the ground truth as a *route*: the ordered list of arc ids the
vehicle traversed (`<trip>.route`). It does NOT give a road segment per GPS
point. Observation-level metrics therefore need a per-point reference label,
which this module derives by a monotone alignment: each GPS point is assigned
to one arc of the reference route so that (a) the order of the route is
respected and (b) the summed point-to-arc distance is minimal (dynamic
programming). These derived labels are an approximation of the truth, not the
truth itself; `ref_dist_*` columns in the output report how far points lie from
their assigned reference arc so this can be checked.

Inputs passed to match()
------------------------
    trace        : numpy array (N, 3) -> longitude, latitude, timestamp [s]
    road_network : dict with
                     "nodes": numpy array (K, 2) -> longitude, latitude
                     "arcs" : numpy int array (E, 2) -> from_node, to_node
                   (arc id == row index in "arcs"; node id == row index in "nodes")
Every trip ships its own local road network, so arc ids are trip-local.

Usage
-----
    python -m src.evaluate --data-dir <partition dir> --matcher src.baseline:match --name baseline
    python -m src.evaluate --data-dir <partition dir> --predictions preds.json --name api_run
    python -m src.evaluate --data-dir <partition dir> --matcher oracle --name oracle   # self-check

<partition dir> is a folder of trip sub-folders (e.g. .../student_test).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
UNMATCHED = -1   # a matcher may return -1 for a point it could not match; scored as wrong

PER_TRIP_FIELDS = [
    "trip_id", "n_points", "n_route_arcs", "status",
    "point_accuracy", "point_accuracy_undirected", "unmatched_rate",
    "macro_precision", "macro_recall", "macro_f1",
    "route_continuity", "route_breaks",
    "route_arc_precision", "route_arc_recall", "route_arc_f1",
    "sequence_similarity",
    "inference_time_s",
    "ref_dist_median_m", "ref_dist_p95_m", "ref_route_continuity",
    "error",
]


# --------------------------------------------------------------------------- #
# Raw data reading (ground truth side)
# --------------------------------------------------------------------------- #

def read_trip(trip_dir: Path) -> dict:
    """Read one trip folder (<id>.track/.nodes/.arcs/.route)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)
    route = np.loadtxt(trip_dir / f"{tid}.route", dtype=np.int64, ndmin=1)
    return {"trip_id": tid, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def list_trips(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    trips = sorted(p for p in data_dir.iterdir()
                   if p.is_dir() and (p / f"{p.name}.track").exists())
    if not trips:
        raise FileNotFoundError(f"no trip folders (<id>/<id>.track) in {data_dir}")
    return trips


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def project_local(lonlat: np.ndarray, lat0: float) -> np.ndarray:
    """Equirectangular projection to metres around reference latitude lat0.

    Trips span up to ~45 km (student_test); the east-west scale error from
    using one cos(lat0) is below 0.5% there, negligible for the short
    point-to-arc distances computed here.
    """
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return np.column_stack((EARTH_RADIUS_M * lon * math.cos(math.radians(lat0)),
                            EARTH_RADIUS_M * lat))


def point_segment_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances (N, M) from points p (N,2) to segments a[j]-b[j] (M,2)."""
    ab = b - a                                     # (M,2)
    ab_len2 = np.einsum("ij,ij->i", ab, ab)        # (M,)
    ap = p[:, None, :] - a[None, :, :]             # (N,M,2)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.einsum("nmk,mk->nm", ap, ab) / ab_len2
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    return np.linalg.norm(p[:, None, :] - closest, axis=2)


# --------------------------------------------------------------------------- #
# Reference labels
# --------------------------------------------------------------------------- #

def derive_reference_labels(trip: dict) -> tuple[np.ndarray, np.ndarray]:
    """Monotone alignment of GPS points to the reference route.

    Returns (labels, dist_m): labels[i] is the route arc id assigned to point i,
    dist_m[i] the distance in metres from point i to that arc.
    """
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    lat0 = float(np.mean(track[:, 1]))
    pts = project_local(track[:, :2], lat0)
    nxy = project_local(nodes, lat0)
    a = nxy[arcs[route, 0]]
    b = nxy[arcs[route, 1]]
    d = point_segment_distances(pts, a, b)         # (N, R)

    n, r = d.shape
    back = np.empty((n, r), dtype=np.int32)
    cost = d[0].copy()
    back[0] = np.arange(r)
    idx = np.arange(r)
    for i in range(1, n):
        # best predecessor position k <= j: prefix minimum of cost and its argmin
        pm = np.minimum.accumulate(cost)
        is_new = np.concatenate(([True], cost[1:] < pm[:-1]))
        back[i] = np.maximum.accumulate(np.where(is_new, idx, 0))
        cost = pm + d[i]

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], d[np.arange(n), pos]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def point_accuracy(pred: Sequence[int], ref: Sequence[int]) -> float:
    pred, ref = np.asarray(pred), np.asarray(ref)
    return float(np.mean(pred == ref)) if len(ref) else float("nan")


def point_accuracy_undirected(pred, ref, arcs: np.ndarray) -> float:
    """Accuracy where an arc and its reverse twin (same two nodes) count as equal.

    Unmatched points (-1) count as wrong.
    """
    pred, ref = np.asarray(pred), np.asarray(ref)
    if not len(ref):
        return float("nan")
    matched = pred != UNMATCHED
    pk = np.sort(arcs[np.where(matched, pred, 0)], axis=1)
    rk = np.sort(arcs[ref], axis=1)
    return float(np.mean(matched & np.all(pk == rk, axis=1)))


def unmatched_rate(pred) -> float:
    pred = np.asarray(pred)
    return float(np.mean(pred == UNMATCHED)) if len(pred) else float("nan")


def macro_prf(pred, ref) -> tuple[float, float, float]:
    """Macro precision/recall/F1 over arc ids appearing in pred or ref.

    -1 (unmatched) is not a class: an unmatched point is a missed reference
    point, so it lowers recall of its reference arc.
    """
    from sklearn.metrics import precision_recall_fscore_support
    labels = np.union1d(np.asarray(pred), np.asarray(ref))
    labels = labels[labels != UNMATCHED]
    p, r, f, _ = precision_recall_fscore_support(
        ref, pred, labels=labels, average="macro", zero_division=0)
    return float(p), float(r), float(f)


def route_continuity(pred, arcs: np.ndarray) -> tuple[float, int]:
    """Share of consecutive point pairs whose predicted arcs are the same arc or
    directly connected (to_node of first == from_node of second).

    Returns (continuity, number_of_breaks). Note that at 1 Hz a very short arc
    can be skipped legitimately, so even perfect labels may score < 1; compare
    with `ref_route_continuity`. A pair involving an unmatched point (-1) is a break.
    """
    pred = np.asarray(pred)
    if len(pred) < 2:
        return float("nan"), 0
    prev, nxt = pred[:-1], pred[1:]
    both = (prev != UNMATCHED) & (nxt != UNMATCHED)
    p0, n0 = np.where(both, prev, 0), np.where(both, nxt, 0)
    ok = both & ((prev == nxt) | (arcs[p0, 1] == arcs[n0, 0]))
    return float(np.mean(ok)), int(np.sum(~ok))


def route_arc_prf(pred, route) -> tuple[float, float, float]:
    """Set-level precision/recall/F1 of the predicted arc set vs reference route arcs."""
    ps, rs = set(map(int, pred)) - {UNMATCHED}, set(map(int, route))
    tp = len(ps & rs)
    p = tp / len(ps) if ps else 0.0
    r = tp / len(rs) if rs else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def collapse(seq) -> list[int]:
    """Remove consecutive duplicates: [5,5,7,7,7,5] -> [5,7,5]."""
    out: list[int] = []
    for x in seq:
        x = int(x)
        if not out or out[-1] != x:
            out.append(x)
    return out


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sequence_similarity(pred, route) -> float:
    """1 - normalised edit distance between collapsed prediction and the route.

    Unmatched points (-1) are dropped before collapsing.
    """
    cp = collapse(x for x in pred if int(x) != UNMATCHED)
    cr = collapse(route)
    denom = max(len(cp), len(cr))
    return 1.0 - levenshtein(cp, cr) / denom if denom else float("nan")


def candidate_miss_rate(candidates: Sequence[Sequence[int]], ref) -> float:
    """Share of points whose reference arc is absent from their candidate set.

    Not used by the CLI (match() returns no candidates); call it directly from
    candidate-generation code.
    """
    miss = [int(r) not in set(map(int, c)) for c, r in zip(candidates, ref)]
    return float(np.mean(miss)) if miss else float("nan")


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #

def score_trip(trip: dict, pred, ref_labels=None, ref_dist=None,
               inference_time_s: float = float("nan")) -> dict:
    if ref_labels is None:
        ref_labels, ref_dist = derive_reference_labels(trip)
    arcs, route = trip["arcs"], trip["route"]
    row = {"trip_id": trip["trip_id"], "n_points": len(trip["track"]),
           "n_route_arcs": len(route), "status": "ok", "error": "",
           "inference_time_s": inference_time_s,
           "ref_dist_median_m": float(np.median(ref_dist)),
           "ref_dist_p95_m": float(np.percentile(ref_dist, 95)),
           "ref_route_continuity": route_continuity(ref_labels, arcs)[0]}

    pred = np.asarray(pred, dtype=np.int64).ravel()
    if len(pred) != len(ref_labels):
        row.update(status="invalid",
                   error=f"prediction length {len(pred)} != number of GPS points {len(ref_labels)}")
        return row
    if len(pred) and (pred.min() < UNMATCHED or pred.max() >= len(arcs)):
        row.update(status="invalid", error="prediction contains arc ids outside this trip's network")
        return row

    row["point_accuracy"] = point_accuracy(pred, ref_labels)
    row["point_accuracy_undirected"] = point_accuracy_undirected(pred, ref_labels, arcs)
    row["unmatched_rate"] = unmatched_rate(pred)
    row["macro_precision"], row["macro_recall"], row["macro_f1"] = macro_prf(pred, ref_labels)
    row["route_continuity"], row["route_breaks"] = route_continuity(pred, arcs)
    (row["route_arc_precision"], row["route_arc_recall"],
     row["route_arc_f1"]) = route_arc_prf(pred, route)
    row["sequence_similarity"] = sequence_similarity(pred, route)
    return row


def load_matcher(spec: str) -> Callable:
    """'package.module:function' or 'path/to/file.py:function'."""
    mod_part, _, fn = spec.rpartition(":")
    if not mod_part or not fn or "\\" in fn or "/" in fn:
        raise ValueError("matcher must look like 'src.baseline:match' or 'file.py:match'")
    if mod_part.endswith(".py"):
        s = importlib.util.spec_from_file_location(Path(mod_part).stem, mod_part)
        module = importlib.util.module_from_spec(s)
        s.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    return getattr(module, fn)


def evaluate(data_dir, matcher: Callable | None = None, predictions: dict | None = None,
             max_trips: int | None = None, trip_ids: Sequence[str] | None = None,
             oracle: bool = False, log=print) -> tuple[list[dict], dict]:
    """Run/score a matcher (or stored predictions) on every trip in data_dir."""
    trips = list_trips(data_dir)
    if trip_ids:
        wanted = set(trip_ids)
        trips = [t for t in trips if t.name in wanted]
    if max_trips:
        trips = trips[:max_trips]

    rows = []
    for tdir in trips:
        trip = read_trip(tdir)
        ref, dist = derive_reference_labels(trip)
        t_inf = float("nan")
        try:
            if oracle:
                pred = ref
            elif predictions is not None:
                if trip["trip_id"] not in predictions:
                    raise KeyError("no prediction for this trip")
                pred = predictions[trip["trip_id"]]
            else:
                # Ground truth (route / ref labels) is never passed to the matcher.
                trace = trip["track"].copy()
                network = {"nodes": trip["nodes"].copy(), "arcs": trip["arcs"].copy()}
                t0 = time.perf_counter()
                pred = matcher(trace, network)
                t_inf = time.perf_counter() - t0
            row = score_trip(trip, pred, ref, dist, t_inf)
        except Exception as exc:  # keep going; record the failure
            row = score_trip(trip, [], ref, dist, t_inf)
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        log(f"{row['trip_id']}: status={row['status']} "
            f"acc={_fmt(row.get('point_accuracy'))} "
            f"f1={_fmt(row.get('macro_f1'))} "
            f"cont={_fmt(row.get('route_continuity'))} "
            f"t={_fmt(row.get('inference_time_s'))}s"
            + (f"  [{row['error']}]" if row["error"] else ""))
    return rows, summarise(rows)


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"n_trips": len(rows), "n_trips_ok": len(ok),
               "n_trips_failed": len(rows) - len(ok)}
    if not ok:
        return summary
    n_pts = np.array([r["n_points"] for r in ok])
    acc = np.array([r["point_accuracy"] for r in ok])
    summary["point_accuracy_pooled"] = float(np.sum(acc * n_pts) / np.sum(n_pts))
    for k in ["point_accuracy", "point_accuracy_undirected", "unmatched_rate",
              "macro_precision", "macro_recall",
              "macro_f1", "route_continuity", "route_arc_precision", "route_arc_recall",
              "route_arc_f1", "sequence_similarity", "inference_time_s", "ref_dist_median_m",
              "ref_route_continuity"]:
        vals = np.array([r[k] for r in ok], dtype=float)
        if np.all(np.isnan(vals)):
            summary[f"{k}_mean"] = None
            continue
        summary[f"{k}_mean"] = float(np.nanmean(vals))
        summary[f"{k}_std"] = float(np.nanstd(vals))
    summary["route_breaks_total"] = int(sum(r["route_breaks"] for r in ok))
    return summary


def write_results(rows, summary, out_dir: Path, name: str, meta: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_per_trip.csv"
    json_path = out_dir / f"{name}_summary.json"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_TRIP_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PER_TRIP_FIELDS})
    with open(json_path, "w") as f:
        json.dump({**meta, "summary": summary, "per_trip": rows}, f, indent=2, default=_json_default)
    return csv_path, json_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    raise TypeError(type(o))


def _fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a PathSnap matcher against reference routes.")
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="partition folder containing trip sub-folders (e.g. .../student_test)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--matcher", help="'src.baseline:match', 'path/file.py:match', or 'oracle' "
                                       "(scores the derived reference labels; pipeline self-check)")
    src.add_argument("--predictions", type=Path,
                     help="JSON file {trip_id: [arc_id per GPS point, ...]}")
    ap.add_argument("--name", default=None, help="run name used for output file names")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--max-trips", type=int, default=None, help="evaluate only the first N trips")
    ap.add_argument("--trips", nargs="*", default=None, help="explicit trip ids to evaluate")
    args = ap.parse_args(argv)

    oracle = args.matcher == "oracle"
    matcher = None
    preds = None
    if args.predictions:
        preds = json.loads(args.predictions.read_text())
        source = f"predictions:{args.predictions.name}"
    elif oracle:
        source = "oracle"
    else:
        matcher = load_matcher(args.matcher)
        source = args.matcher
    name = args.name or (source.split(":")[0].split(".")[-1] if not oracle else "oracle")

    rows, summary = evaluate(args.data_dir, matcher=matcher, predictions=preds,
                             max_trips=args.max_trips, trip_ids=args.trips, oracle=oracle)
    meta = {"run_name": name, "source": source, "partition": args.data_dir.name,
            "trip_ids": [r["trip_id"] for r in rows],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reference_labels": "monotone DP alignment of GPS points to the reference route"}
    csv_path, json_path = write_results(rows, summary, args.out_dir, name, meta)
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"wrote {csv_path} and {json_path}")
    return 0 if summary["n_trips_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
