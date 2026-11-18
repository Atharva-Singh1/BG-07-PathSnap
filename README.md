# BG-07-PathSnap
The challenge is PathSnap, a GPS-to-road map-matching system. The rules explicitly require genuine experiments and forbid invented metrics

## Evaluation

Code: [`src/evaluate.py`](src/evaluate.py) · tests: [`tests/test_evaluate.py`](tests/test_evaluate.py)

### What is compared

The dataset's ground truth is a **route**: `<trip>.route` lists, in order, the arc ids the vehicle
drove along (arc id = row index in `<trip>.arcs`). There is **no reference road segment per GPS
point**, and the arc ids belong to each trip's own road network.

Observation-level metrics need one reference label per point, so `evaluate.py` creates them with a
**monotone alignment**. Each GPS point is assigned to one arc of the reference route, keeping the
route's order and minimising the total point-to-arc distance (dynamic programming over a local
metric projection). These labels are *derived*, so they are an approximation of the truth.
`ref_dist_median_m` / `ref_dist_p95_m` report how far points lie from their assigned reference arc.

### Metrics

The problem statement ("Models and Training Guidelines", §5 and §9) asks for these where applicable.
The primary metric is observation-level Top-1 accuracy.

| Column | Meaning |
|---|---|
| `point_accuracy` | **Observation-level Top-1 accuracy.** Share of GPS points whose predicted arc id equals the derived reference arc. Primary metric. |
| `point_accuracy_undirected` | Same, but an arc and its reverse twin (same two nodes) count as a match. A diagnostic for direction errors. |
| `unmatched_rate` | Share of points the matcher returned `-1` for (no match). These points count as wrong in every metric. |
| `macro_precision` / `macro_recall` / `macro_f1` | Precision, recall and F1 per arc id, macro-averaged over the arcs that appear in the prediction or the reference (sklearn, `zero_division=0`). |
| `route_continuity`, `route_breaks` | Share of consecutive point pairs whose predicted arcs are the same arc or directly connected (`to_node` of the first = `from_node` of the second), plus the count of breaks. |
| `route_arc_precision` / `_recall` / `_f1` | Sequence level: the set of predicted arcs compared with the set of reference route arcs. |
| `sequence_similarity` | Sequence level: 1 − normalised edit distance between the prediction (consecutive duplicates removed) and the reference route. |
| `inference_time_s` | Wall-clock time of the `match()` call for the trip. |
| `ref_route_continuity` | `route_continuity` of the derived reference labels: a **reference point, not an upper bound**. At 1 Hz a short arc can fall between two GPS points, so even the reference labels score below 1, while a matcher can score higher, for example by predicting the same arc for many points. |

Summary JSON: the mean and standard deviation over trips of each metric, plus `point_accuracy_pooled`
(accuracy over all points of all trips).

Not computed by the CLI, because `match()` returns only one arc per point:
- **Log loss**: needs class probabilities. Not implemented yet (N/A).
- **Candidate miss rate**: `candidate_miss_rate(candidates, ref)` exists for candidate-generation
  code to call directly.
- **Confusion matrix**: not implemented. Arc ids are trip-local and number in the thousands, so a
  per-arc matrix is not meaningful. If we need one, it should be over candidate rank or classifier classes.

### Matcher interface the evaluator calls

```python
match(trace, road_network) -> list[int]        # one arc id per GPS point, len == len(trace)
# trace:        numpy (N, 3)  longitude, latitude, timestamp [s]   (<trip>.track)
# road_network: {"nodes": numpy (K, 2) lon, lat,                    (<trip>.nodes)
#                "arcs":  numpy int (E, 2) from_node, to_node}      (<trip>.arcs)
```

The matcher never sees `.route` or the reference labels. A prediction of the wrong length, or with
arc ids outside the trip's network, gives the trip the status `invalid`. A matcher may return `-1` for a point it
cannot match; that point is scored as wrong, not as invalid. An exception gives the
status `error`. Both are recorded, and evaluation continues with the next trip.

### How to run

Requirements: Python 3.10+, `numpy`, `scikit-learn` (and `pytest` for the tests).
The data is **not** in this repository. Pass the partition folder (the one that holds the trip
sub-folders) as `--data-dir`.

```bash
# self-check: score the derived reference labels against themselves (accuracy must be 1.0)
python -m src.evaluate --data-dir "<path>/student_test" --matcher oracle --max-trips 3

# evaluate a matcher module
python -m src.evaluate --data-dir "<path>/student_test" --matcher src.baseline:match    --name baseline
python -m src.evaluate --data-dir "<path>/student_test" --matcher src.hmm_matcher:match --name hmm

# score stored predictions, e.g. from the API: JSON {"<trip_id>": [arc_id, ...], ...}
python -m src.evaluate --data-dir "<path>/student_test" --predictions preds.json --name api

# unit tests (synthetic network, no data needed)
python -m pytest -q tests
```

Options: `--max-trips N` (first N trips), `--trips ID [ID ...]`, `--out-dir` (default `results/`).

Each run writes `results/<name>_per_trip.csv` and `results/<name>_summary.json`.

### Results

All values are means over the 10 `student_test` trips, taken from the committed files
`results/baseline_summary.json` and `results/hmm_summary.json` (run with the default settings,
including `configs/hmm_default.json` for the HMM). Per-trip values are in the matching
`*_per_trip.csv` files.

| Matcher | Top-1 accuracy | Undirected accuracy | Macro F1 | Route continuity | Route breaks (total) | Route arc F1 | Sequence similarity | Inference time (s/trip) |
|---|---|---|---|---|---|---|---|---|
| Nearest-road baseline (`src/baseline.py`) | 0.564 | 0.879 | 0.382 | 0.801 | 3,444 | 0.558 | 0.374 | 2.81 |
| HMM/Viterbi, distance-only emission (`src/hmm_matcher.py`) | 0.921 | 0.950 | 0.866 | 0.976 | 371 | 0.923 | 0.837 | 2.14 |

The pooled Top-1 accuracy over all 19,160 points is 0.544 for the baseline and 0.917 for the HMM.
Neither matcher left any point unmatched (`unmatched_rate` = 0).

The HMM does not use a trained classifier yet. Its emission is a Gaussian on the distance from the
point to the arc. Inference times depend on the machine they were run on.

### Known limitations

- The per-point reference labels are derived from the route, not given in the data. Points far from
  every route arc (large `ref_dist_*`) have uncertain labels.
- The reference routes themselves come from the dataset and are not independently verified.
- `levenshtein` is pure Python (O(n·m)). It is fine for trips of this size (≤ a few thousand arcs).
