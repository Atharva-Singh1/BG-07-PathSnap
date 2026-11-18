"""Candidate generation and per-candidate features for the PathSnap classifiers.

For every GPS point we list the K nearest road arcs within `radius_m` (the
candidates). Each (point, candidate arc) pair becomes one row with the
features below. A classifier scores every row; the candidate with the highest
score is the point's predicted arc (observation-level Top-1).

Features (FEATURE_NAMES):
  dist_m          distance from the GPS point to the arc segment
  dist_rank       0 = nearest candidate of this point, 1 = second nearest, ...
  dist_gap_m      dist_m minus the distance of this point's nearest candidate
  heading_cos     cosine of the angle between the direction of travel and the
                  arc's direction (+1 same direction, -1 opposite); 0 if the
                  direction of travel is unknown (vehicle almost stationary)
  heading_valid   1 if the direction of travel could be estimated, else 0
  speed_mps       speed estimated from neighbouring points
  arc_len_m       length of the candidate arc
  frac_along      where the point projects onto the arc (0 = start, 1 = end)
  n_cands         number of candidates this point has
  dt_prev_s       seconds since the previous GPS point (0 for the first point)
  dt_next_s       seconds until the next GPS point (0 for the last point)
  jump_prev_m     distance the GPS point moved from the previous point
                  (large jumps indicate noise or a gap)
  accel_mps2      change in speed between the neighbouring windows
  turn_angle_deg  how sharply the GPS track turns at this point (0 = straight,
                  180 = U-turn); 0 if the vehicle is almost stationary
  node_dist_m     distance from the projected point to the nearest end of the
                  arc (small = close to a junction, where errors are common)
  has_reverse     1 if the same road in the opposite direction is also a
                  candidate for this point (a two-way road), else 0
  links_prev      1 if the arc shares a node with the previous point's nearest
                  candidate arc (a simple sequence clue), else 0
  end_out_degree  number of arcs leaving the arc's end node (junction size)

Only the GPS trace and the road network are used. The route / reference
labels are never inputs; they are used only to build training labels
(`label_candidates`).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import cKDTree

from src.evaluate import project_local

FEATURE_NAMES = ["dist_m", "dist_rank", "dist_gap_m", "heading_cos", "heading_valid",
                 "speed_mps", "arc_len_m", "frac_along", "n_cands",
                 "dt_prev_s", "dt_next_s", "jump_prev_m", "accel_mps2", "turn_angle_deg",
                 "node_dist_m", "has_reverse", "links_prev", "end_out_degree"]


@dataclass
class CandidateConfig:
    k: int = 8                 # max candidates per point
    radius_m: float = 50.0     # candidate search radius
    sample_step_m: float = 10.0  # spacing of index samples along each arc
    k_samples: int = 96        # index samples examined per point
    heading_window: int = 2    # direction of travel from point i-w to i+w
    min_move_m: float = 1.0    # below this displacement the heading is unknown

    def to_dict(self):
        return asdict(self)


@dataclass
class Candidates:
    point: np.ndarray   # (C,) index of the GPS point
    arc: np.ndarray     # (C,) candidate arc id
    X: np.ndarray       # (C, len(FEATURE_NAMES)) features
    n_points: int


def _arc_index(nodes_xy, arcs, step):
    """KD-tree over points sampled every `step` metres along every arc."""
    a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    length = np.hypot(*(b - a).T)
    nseg = np.maximum(1, np.ceil(length / step)).astype(np.int64)
    counts = nseg + 1
    arc_of_sample = np.repeat(np.arange(len(arcs)), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    frac = (np.arange(counts.sum()) - starts) / np.repeat(nseg, counts)
    samples = a[arc_of_sample] + frac[:, None] * (b - a)[arc_of_sample]
    return cKDTree(samples), arc_of_sample, a, b, length


def _project(p, a, b):
    ab = b - a
    l2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / l2, 0.0, 1.0)
    q = a + t[:, None] * ab
    return np.hypot(*(p - q).T), t


def _travel_direction(xy, ts, w, min_move):
    n = len(xy)
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    disp = xy[hi] - xy[lo]
    move = np.hypot(*disp.T)
    dt = ts[hi] - ts[lo]
    speed = np.where(dt > 0, move / np.where(dt > 0, dt, 1), 0.0)
    valid = move >= min_move
    unit = np.where(valid[:, None], disp / np.maximum(move, 1e-9)[:, None], 0.0)
    return unit, valid, speed


def build_candidates(trace, road_network, cfg: CandidateConfig | None = None) -> Candidates:
    cfg = cfg or CandidateConfig()
    trace = np.asarray(trace, dtype=float)
    nodes = np.asarray(road_network["nodes"], dtype=float)
    arcs = np.asarray(road_network["arcs"], dtype=np.int64)
    lat0 = float(np.mean(trace[:, 1]))
    xy = project_local(trace[:, :2], lat0)
    nxy = project_local(nodes, lat0)
    tree, arc_of_sample, a, b, length = _arc_index(nxy, arcs, cfg.sample_step_m)
    n, n_arcs = len(xy), len(arcs)

    k = min(cfg.k_samples, len(arc_of_sample))
    dd, ii = tree.query(xy, k=k, distance_upper_bound=cfg.radius_m + cfg.sample_step_m)
    dd, ii = dd.reshape(n, -1), ii.reshape(n, -1)
    ok = np.isfinite(dd)
    rows = np.repeat(np.arange(n), dd.shape[1])[ok.ravel()]
    cand = arc_of_sample[ii[ok]]
    _, first = np.unique(rows.astype(np.int64) * n_arcs + cand, return_index=True)
    rows, cand = rows[first], cand[first]
    dist, frac = _project(xy[rows], a[cand], b[cand])
    keep = dist <= cfg.radius_m
    rows, cand, dist, frac = rows[keep], cand[keep], dist[keep], frac[keep]

    # every point gets at least one candidate: its nearest arc, however far
    missing = np.setdiff1d(np.arange(n), rows)
    if len(missing):
        _, i1 = tree.query(xy[missing], k=1)
        c1 = arc_of_sample[np.atleast_1d(i1)]
        d1, f1 = _project(xy[missing], a[c1], b[c1])
        rows = np.concatenate([rows, missing]); cand = np.concatenate([cand, c1])
        dist = np.concatenate([dist, d1]); frac = np.concatenate([frac, f1])

    order = np.lexsort((dist, rows))
    rows, cand, dist, frac = rows[order], cand[order], dist[order], frac[order]
    first_of_row = np.searchsorted(rows, rows, side="left")
    rank = np.arange(len(rows)) - first_of_row
    keep = rank < cfg.k
    rows, cand, dist, frac, rank = rows[keep], cand[keep], dist[keep], frac[keep], rank[keep]

    n_cands = np.bincount(rows, minlength=n)[rows]
    min_d = dist[np.searchsorted(rows, rows, side="left")]
    unit, valid, speed = _travel_direction(xy, trace[:, 2], cfg.heading_window, cfg.min_move_m)
    ab = b[cand] - a[cand]
    ab_unit = ab / np.maximum(length[cand], 1e-9)[:, None]
    heading_cos = np.einsum("ij,ij->i", unit[rows], ab_unit)

    # ---- point-level motion features -------------------------------------
    ts = trace[:, 2]
    dt_prev = np.r_[0.0, np.diff(ts)]
    dt_next = np.r_[np.diff(ts), 0.0]
    jump_prev = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    w = cfg.heading_window
    lo = np.clip(np.arange(n) - w, 0, n - 1)
    hi = np.clip(np.arange(n) + w, 0, n - 1)
    dt_win = ts[hi] - ts[lo]
    accel = np.where(dt_win > 0, (speed[hi] - speed[lo]) / np.where(dt_win > 0, dt_win, 1), 0.0)
    v_in, v_out = xy - xy[lo], xy[hi] - xy
    n_in, n_out = np.hypot(*v_in.T), np.hypot(*v_out.T)
    turn_ok = (n_in >= cfg.min_move_m) & (n_out >= cfg.min_move_m)
    cos_turn = np.einsum("ij,ij->i", v_in, v_out) / np.maximum(n_in * n_out, 1e-9)
    turn = np.where(turn_ok, np.degrees(np.arccos(np.clip(cos_turn, -1.0, 1.0))), 0.0)

    # ---- candidate-level network features --------------------------------
    u, v = arcs[cand, 0], arcs[cand, 1]
    node_dist = np.minimum(frac, 1.0 - frac) * length[cand]
    n_nodes = len(nodes)
    key = (rows.astype(np.int64) * n_nodes + u) * n_nodes + v
    rev_key = (rows.astype(np.int64) * n_nodes + v) * n_nodes + u
    has_reverse = np.isin(rev_key, key).astype(float)
    nearest_arc = cand[rank == 0]                     # one per point (rows are sorted)
    nearest_of_point = np.empty(n, dtype=np.int64)
    nearest_of_point[rows[rank == 0]] = nearest_arc
    has_prev = rows > 0
    prev_arc = nearest_of_point[np.maximum(rows - 1, 0)]
    pu, pv = arcs[prev_arc, 0], arcs[prev_arc, 1]
    links_prev = (has_prev & ((u == pu) | (u == pv) | (v == pu) | (v == pv))).astype(float)
    out_degree = np.bincount(arcs[:, 0], minlength=n_nodes)
    end_out_degree = out_degree[v]

    X = np.column_stack([dist, rank, dist - min_d, heading_cos, valid[rows].astype(float),
                         speed[rows], length[cand], frac, n_cands,
                         dt_prev[rows], dt_next[rows], jump_prev[rows], accel[rows], turn[rows],
                         node_dist, has_reverse, links_prev, end_out_degree]).astype(np.float64)
    assert X.shape[1] == len(FEATURE_NAMES)
    return Candidates(point=rows, arc=cand, X=X, n_points=n)


def label_candidates(c: Candidates, ref_labels: np.ndarray) -> np.ndarray:
    """1 where the candidate is the point's reference arc, else 0."""
    return (c.arc == np.asarray(ref_labels)[c.point]).astype(np.int8)


def top1(point: np.ndarray, arc: np.ndarray, score: np.ndarray, n_points: int) -> np.ndarray:
    """Highest-scoring candidate arc for each point (every point has >= 1 candidate)."""
    order = np.lexsort((-score, point))
    p_sorted = point[order]
    first = np.r_[0, np.flatnonzero(np.diff(p_sorted)) + 1]
    out = np.full(n_points, -1, dtype=np.int64)
    out[p_sorted[first]] = arc[order][first]
    return out
