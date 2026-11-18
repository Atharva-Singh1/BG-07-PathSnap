# Evaluation Methodology

This section describes how every matcher in this report is scored. All metrics are computed by
`src/evaluate.py` and written to `results/<run>_per_trip.csv` and `results/<run>_summary.json`.
Every number in this report can be reproduced from those files with the commands in the
Reproduction subsection at the end.

## Data partitions used

| Partition | Trips | Used for |
|---|---|---|
| Training | 80 | Model fitting, group-aware cross-validation (grouped by trip), hyperparameter optimisation |
| Student testing | 10 | Development-stage comparison of models and of the HMM decoder |
| Evaluator testing | 10 | Not available to the team; final evaluation by the organisers only |

The student-testing partition contains 19,160 GPS points in total (262 to 3,958 per trip).

## Ground truth and per-point reference labels

The dataset does not give a road segment for each GPS point. For each trip, the reference is a
**route**: `<trip>.route` lists, in driving order, the arc ids the vehicle traversed. An arc id is
the row index in that trip's `<trip>.arcs` file. Each trip ships its own road network, so arc ids
are only meaningful within one trip.

Observation-level metrics need one reference arc per GPS point. We derive these labels with a
**monotone alignment** of the GPS points to the reference route:

- Let `p_1 ... p_N` be the GPS points of a trip and `r_1 ... r_R` the arcs of its route.
- Let `d(i, j)` be the distance in metres from point `p_i` to the line segment of arc `r_j`.
- The alignment assigns each point a route position `j(i)`, with `j(1) <= j(2) <= ... <= j(N)`,
  that minimises the total distance `d(1, j(1)) + ... + d(N, j(N))`.
- It is solved exactly by dynamic programming:
  `C(i, j) = d(i, j) + min over k <= j of C(i-1, k)`,
  followed by backtracking from the smallest `C(N, j)`.
- The reference label of point `i` is the arc `r_j(i)`.

The order constraint matters when a route passes the same place twice, for example an out-and-back
trip on the same street. Assigning each point to its nearest route arc, without the constraint,
would mix up the two passes.

**Distance calculation.** Coordinates are longitude/latitude, so distances in degrees are not
physical distances. Each trip is projected to metres with an equirectangular projection centred on
the trip's mean latitude. The student-testing trips span up to about 45 km and lie between
latitudes 42° and 62°. From each trip's latitude span, the resulting east–west scale error is
below 0.5%. Since the evaluator only measures short distances (from a point to nearby arcs), this
error is far smaller than GPS noise.

## Metrics

| Metric | Definition | Level |
|---|---|---|
| **Observation-level Top-1 accuracy** (primary) | Share of GPS points whose predicted arc id equals the reference label. | Point |
| Undirected accuracy | As above, but an arc and its reverse twin (same two nodes, opposite direction) count as a match. Used only to diagnose direction errors. | Point |
| Macro precision / recall / F1 | Precision, recall and F1 computed for each arc id, then averaged with equal weight over the arcs that appear in the prediction or the reference of a trip. | Point |
| Route continuity | Share of consecutive point pairs whose predicted arcs are the same arc, or connected (the first arc's end node is the second arc's start node). The number of breaks is also reported. | Sequence |
| Route arc precision / recall / F1 | The set of distinct predicted arcs compared with the set of reference route arcs. | Sequence |
| Sequence similarity | 1 − (edit distance ÷ length of the longer sequence), between the predicted arc sequence (consecutive repeats removed) and the reference route. | Sequence |
| Inference time | Wall-clock time of one call to the matcher for a whole trip. | Efficiency |

Per-trip values are averaged over trips (mean and standard deviation). We also report a pooled
accuracy over all points of all trips, which gives long trips more weight.

**Why Top-1 accuracy is the primary metric.** The guidelines define observation-level Top-1
accuracy as the comparison of each point's prediction with its reference road segment, and the
required output is exactly one road segment per GPS point. Every model and decoder can be scored
on it in the same way. Accuracy alone can hide a route that jumps between disconnected roads, so we
always read it together with route continuity, which shows whether the HMM decoder produces a
route that could actually be driven.

### Metrics reported as N/A

- **Log loss.** It needs a probability for each candidate arc. The shared matcher interface
  returns only one arc per point, so log loss is N/A for matchers. [Fill in if a classifier
  reports it separately during model experiments.]
- **Candidate miss rate.** It applies only to methods that generate candidate arcs. The function
  `candidate_miss_rate` computes it from the candidate sets. [Fill in from the candidate-generation
  code, or state N/A.]
- **Confusion matrix.** It is not reported at arc level, because each trip has its own network
  with thousands of arc ids, so a per-arc matrix is not interpretable.

## Validation of the evaluation pipeline

**Unit tests.** Nine unit tests (`tests/test_evaluate.py`) check every metric on a small
hand-built network where the correct values can be worked out by hand. They also check the
alignment on an out-and-back route, and that the matcher receives only the GPS trace and the road
network, never the route.

**Reference-label check.** We scored the derived reference labels against themselves on all 10
student-testing trips. By construction this gives an accuracy of 1.0, so it is not a model result.
It does measure how well the labels fit the data, and it gives the best score any matcher could
reach on the sequence metrics:

| Trip | GPS points | Route arcs | Median point-to-arc distance (m) | 95th percentile (m) | Continuity ceiling | Route arc recall ceiling |
|---|---|---|---|---|---|---|
| 00000000 | 2,503 | 179 | 6.1 | 16.6 | 0.998 | 0.972 |
| 00000007 | 1,354 | 123 | 4.5 | 26.1 | 1.000 | 0.984 |
| 00000009 | 1,049 | 429 | 4.7 | 18.1 | 0.954 | 0.841 |
| 00000040 | 2,995 | 309 | 4.9 | 14.9 | 0.997 | 0.968 |
| 00000052 | 262 | 95 | 4.2 | 12.6 | 0.966 | 0.874 |
| 00000056 | 3,958 | 1,323 | 4.8 | 14.5 | 0.966 | 0.890 |
| 00000065 | 768 | 145 | 4.4 | 11.8 | 0.980 | 0.890 |
| 00000076 | 1,486 | 374 | 6.8 | 18.7 | 0.978 | 0.890 |
| 00000077 | 3,911 | 268 | 6.1 | 18.4 | 0.997 | 0.959 |
| 00000091 | 874 | 232 | 4.4 | 12.5 | 0.973 | 0.784 |
| **Mean** | | | **5.1** | | **0.981** | **0.905** |

What the check shows:

- Half of the GPS points of a trip lie within 4.2 to 6.8 m of their reference arc (the median
  distance), which is consistent with normal GPS noise. On every trip, 95% of points are within
  26.1 m.
- **Even perfect point labels cannot reach 1.0 on route arc recall** (mean 0.905). Some route arcs
  are so short that no GPS point falls on them at 1 Hz sampling, so a method that labels points
  cannot cover them. Trip 00000091 has the lowest ceiling (0.784).
- For the same reason, **route continuity cannot reach 1.0 either** (mean 0.981): consecutive
  points sometimes lie on arcs separated by a short skipped arc. Model scores on these two metrics
  must be read against these ceilings, not against 1.0.

## Limitations

- The per-point labels are derived from the route, not given in the data. They are an
  approximation of the truth. Labels are least certain for points far from every route arc and
  near junctions, where a point is about equally close to two consecutive arcs.
- The reference routes come with the dataset and have not been independently verified. As the
  dataset documentation notes, they may contain errors.
- In this dataset every road appears as two opposite arcs with the same geometry. Choosing the
  wrong direction counts as an error under Top-1 accuracy. Undirected accuracy is reported so
  that direction errors can be told apart from choosing the wrong road.
- Route arc precision/recall ignore order and repetition. Sequence similarity accounts for both,
  but it is sensitive to the short arcs that 1 Hz sampling skips.

## Results

[To be filled only from committed files in `results/`. Do not fill in any value by hand.]

| Model | Top-1 accuracy | Undirected accuracy | Macro F1 | Route continuity | Route arc F1 | Sequence similarity | Inference time (s/trip) | Source file |
|---|---|---|---|---|---|---|---|---|
| Nearest-road baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | `results/baseline_summary.json` |
| Classifier only | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Classifier + HMM/Viterbi | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | `results/hmm_summary.json` |

All values are means over the 10 student-testing trips.

## Reproduction

```bash
# score a matcher on the student-testing partition (the data path is passed in, never hardcoded)
python -m src.evaluate --data-dir "<path>/student_test" --matcher src.baseline:match    --name baseline
python -m src.evaluate --data-dir "<path>/student_test" --matcher src.hmm_matcher:match --name hmm

# reference-label check (accuracy must be 1.0)
python -m src.evaluate --data-dir "<path>/student_test" --matcher oracle --name labelcheck_student_test

# unit tests
python -m pytest -q tests
```
