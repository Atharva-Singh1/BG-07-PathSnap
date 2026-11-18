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

## Model selection

Code: [`src/features.py`](src/features.py) (candidates + features),
[`src/train_models.py`](src/train_models.py) (training, cross-validation, selection),
[`src/classifier_matcher.py`](src/classifier_matcher.py) (the selected model as a matcher).
Models: one JSON file per model in [`configs/models/`](configs/models/).

### Approach

1. **Candidates.** For each GPS point, the up to 8 nearest arcs within 50 m. If none are within
   50 m, the single nearest arc is used. The candidate miss rate is the share of points whose
   reference arc is not among their candidates.
2. **18 features** for each (point, candidate arc) pair, all in `src/features.py`:

   | Group | Features |
   |---|---|
   | Distance | `dist_m` (point to arc), `dist_rank` (rank among the candidates), `dist_gap_m` (gap to the nearest candidate) |
   | Direction | `heading_cos` (**direction of travel vs the arc's direction**, which separates the two directed arcs of a two-way road), `heading_valid`, `turn_angle_deg` (how sharply the GPS track turns) |
   | Motion | `speed_mps`, `accel_mps2`, `jump_prev_m` (distance from the previous point), `dt_prev_s` / `dt_next_s` (time gaps) |
   | Arc and junction | `arc_len_m`, `frac_along` (where the point projects along the arc), `node_dist_m` (distance to the nearest arc end), `end_out_degree` (number of arcs leaving the end node) |
   | Context | `n_cands` (number of candidates), `has_reverse` (the opposite direction is also a candidate), `links_prev` (shares a node with the previous point's nearest arc) |

   Only the GPS trace and the road network are used.
3. **Target:** 1 if the candidate is the point's reference arc (derived as in the Evaluation
   section), otherwise 0. A model's prediction for a point is its highest-probability candidate
   (observation-level Top-1).
4. **Five models from five families**, each owned by one team member:

   | Model | Family | Owner |
   |---|---|---|
   | Logistic regression | Linear | Aryan |
   | Gaussian Naive Bayes | Probabilistic | Aryan |
   | MLP | Neural network | Atharva |
   | Random forest | Tree ensemble (bagging) | Arjav |
   | Histogram gradient boosting | Boosting | Blason |

5. **Cross-validation:** 5-fold `GroupKFold` grouped by trip on the train partition, so no trip
   appears in both training and validation. Each fold samples at most 40,000 training points and
   20,000 validation points, the same sample for every model, with seed 42.
6. **Selection:** the highest mean CV Top-1 accuracy, with ties broken by lower CV log loss.
   **`student_test` is not used for selection.** Every model is then refit on up to 80,000 train
   points and scored on `student_test` as a classifier-only matcher with `src/evaluate.py`.

**Data-quality filter (train only).** Train trips whose GPS points lie a median of more than
50 m from their reference route are excluded, because their derived labels are unreliable (the
route doesn't match the trace). This removes trips `00000054` and `00000033`. The excluded trips
are recorded in each run's `selection.json`.

### Adding your model

Create `configs/models/<name>.json`:

```json
{
  "name": "random_forest",
  "owner": "Arjav",
  "family": "Tree ensemble (bagging)",
  "estimator": "sklearn.ensemble.RandomForestClassifier",
  "params": {"n_estimators": 100, "min_samples_leaf": 5, "n_jobs": -1},
  "scale": false
}
```

- `estimator` must be a scikit-learn class.
- `scale: true` puts a `StandardScaler` in front of the estimator.
- `random_state` is set from `--seed` automatically.

Then run only your model:

```bash
python -m src.train_models --train-dir "<path>/train" --test-dir "<path>/student_test" --models random_forest
```

To try other settings without editing the file:

```bash
python -m src.train_models ... --models random_forest --params '{"random_forest": {"n_estimators": 300}}'
```

Runs with `--models`, `--params` or `--quick` write to `results/experiments/<tag>/`. Commit that
folder: it records the settings, the seed and the results.

### Official comparison

When all five configs exist, run without `--models` or `--params`:

```bash
python -m src.train_models --train-dir "<path>/train" --test-dir "<path>/student_test"
```

This is the only kind of run that writes `results/model_selection/` (`cv_folds.csv`,
`model_comparison.csv`, `test_<model>_*`, `<model>_loss_curve.csv` for models with an epoch
history, `selection.json`) and `models/classifier_selected.joblib`. With fewer than five configs,
it writes to `results/experiments/` instead.

```bash
# evaluate the saved selected model as a matcher
python -m src.evaluate --data-dir "<path>/student_test" --matcher src.classifier_matcher:match --name classifier
```

### Results

_Filled in from `results/model_selection/` after the official five-model run._

## UI (laptop 2)

Code: [`ui/app.py`](ui/app.py) (Streamlit page), [`ui/client.py`](ui/client.py) (API calls and
Folium map). API format: [`docs/api_contract.md`](docs/api_contract.md).

```bash
pip install streamlit folium streamlit-folium requests
streamlit run ui/app.py
```

1. In the sidebar, set **API URL** to laptop 1's address, e.g. `http://192.168.1.20:8000`, and
   click **Check connection**. The URL can also come from the `PATHSNAP_API_URL` environment variable.
2. Pick an evaluation trip and click **Match route**.
3. The page shows:
   - the Top-1 accuracy, route continuity, macro F1 and unmatched rate, all **computed on the API**
   - the server time and the round-trip time
   - a Folium map of the GPS points and the predicted route, with a legend and layer control

The UI never receives ground truth. `ui/client.py` rejects any response that contains
reference-route fields.

**Testing without laptop 1:** `python ui/mock_api.py --port 8000` starts a **fake** API with
synthetic trips (`MOCK-01` …) and made-up numbers, for checking the UI only. It is never used in
the demo.
