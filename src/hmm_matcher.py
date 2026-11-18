"""HMM map matcher (Viterbi) for PathSnap.

Hidden state      : a candidate road arc (arc_id) for a GPS observation.
Observation       : one GPS point (lon, lat), projected to local metres.
Emission          : Gaussian on the distance GPS point -> arc segment.
Transition        : exponential on | great-circle(p_t, p_t+1) - route distance |
                    (Newson & Krumm style), route distance via Dijkstra.
Decoding          : Viterbi in log space.

SIMPLIFICATIONS (search for "SIMPLIFIED" in this file):
  S1. Fixed candidate radius, no adaptive widening; nearest `max_candidates` kept.
  S2. Emission is a pure Gaussian of distance; no heading / speed / classifier
      score yet (the classifier probability hook is `emission_extra`, unused).
  S3. Route distance is measured between the *projected positions* on the two
      candidate arcs, using node-to-node Dijkstra with a distance cutoff.
      Arc length is straight node-to-node (no shape points exist in .nodes).
  S4. Transition uses a single exponential scale `beta` (no time/speed scaling),
      so long sampling gaps are penalised more than they should be.
  S5. Dead end (all transitions impossible) -> "break": decoding restarts a new
      segment from emissions alone. No interpolation across the gap.
  S6. Same-arc / backwards movement on an arc is treated as zero route distance
      plus a small `backward_penalty_m`, not a U-turn model.
  S7. Consecutive GPS points closer than `min_step_m` are skipped (stationary),
      and they inherit the previous match.
  S8. Arcs are directed as given in .arcs; one-way vs two-way is not inferred.
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

EARTH_R = 6371008.8


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class HMMConfig:
    candidate_radius_m: float = 50.0     # S1: small start radius
    max_candidates: int = 8              # keep K nearest arcs within radius
    sigma_m: float = 15.0                # emission std-dev (GPS noise)
    beta_m: float = 30.0                 # transition scale
    max_route_factor: float = 4.0        # dijkstra cutoff = factor*gc + slack
    route_slack_m: float = 200.0
    backward_penalty_m: float = 5.0      # S6
    min_step_m: float = 2.0              # S7
    max_gap_s: float = 60.0              # time gap above which we force a break
    log_floor: float = -1e9              # stand-in for log(0)

    @staticmethod
    def from_json(path) -> "HMMConfig":
        return HMMConfig(**json.loads(Path(path).read_text()))

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------
# Data loading + geometry
# --------------------------------------------------------------------------
def load_trip(trip_dir, with_route: bool = True) -> dict:
    """Load one trip folder. `route` is reference only; never used by matcher."""
    trip_dir = Path(trip_dir)
    t = trip_dir.name
    track = np.loadtxt(trip_dir / f"{t}.track")
    nodes = np.loadtxt(trip_dir / f"{t}.nodes")
    arcs = np.loadtxt(trip_dir / f"{t}.arcs", dtype=int)
    out = dict(trip_id=t, track=np.atleast_2d(track), nodes=nodes, arcs=np.atleast_2d(arcs))
    if with_route:
        rp = trip_dir / f"{t}.route"
        out["route"] = np.loadtxt(rp, dtype=int) if rp.exists() else None
    return out


class LocalProjection:
    """Equirectangular projection around a trip centre -> metres.

    Fine at city scale (error << GPS noise); one projection per trip since
    trips are in different countries."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = math.radians(1.0) * EARTH_R * math.cos(math.radians(lat0))
        self.ky = math.radians(1.0) * EARTH_R

    def __call__(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])


class RoadGraph:
    """Directed arc graph with a uniform-grid spatial index on arc segments."""

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection,
                 cell_m: float = 100.0):
        self.proj = proj
        self.xy = proj(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs
        self.u, self.v = arcs[:, 0], arcs[:, 1]
        d = self.xy[self.v] - self.xy[self.u]
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.out_arcs: list[list[int]] = [[] for _ in range(len(self.xy))]
        for a, (u, _) in enumerate(arcs):
            self.out_arcs[u].append(a)
        # spatial grid over arcs (arc registered in every cell its bbox touches)
        self.cell = cell_m
        self.grid: dict[tuple[int, int], list[int]] = {}
        p0, p1 = self.xy[self.u], self.xy[self.v]
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        ilo, ihi = np.floor(lo / cell_m).astype(int), np.floor(hi / cell_m).astype(int)
        for a in range(len(arcs)):
            for i in range(ilo[a, 0], ihi[a, 0] + 1):
                for j in range(ilo[a, 1], ihi[a, 1] + 1):
                    self.grid.setdefault((i, j), []).append(a)

    def arcs_near(self, p: np.ndarray, radius: float) -> np.ndarray:
        r = int(math.ceil(radius / self.cell))
        ci, cj = int(p[0] // self.cell), int(p[1] // self.cell)
        found: set[int] = set()
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                found.update(self.grid.get((i, j), ()))
        return np.fromiter(found, dtype=int) if found else np.empty(0, dtype=int)

    def project(self, p: np.ndarray, arc_ids: np.ndarray):
        """Distance from p to each arc segment and fraction along the arc [0,1]."""
        a, b = self.xy[self.u[arc_ids]], self.xy[self.v[arc_ids]]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        f = np.clip(((p - a) * ab).sum(1) / L2, 0.0, 1.0)
        q = a + ab * f[:, None]
        return np.hypot(*(p - q).T), f


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------
@dataclass
class MatchResult:
    arc_ids: np.ndarray          # one arc per input GPS point (-1 = unmatched)
    breaks: list = field(default_factory=list)   # indices where chain restarted
    n_candidates: np.ndarray = None


class HMMMatcher:
    def __init__(self, graph: RoadGraph, cfg: HMMConfig | None = None):
        self.g = graph
        self.cfg = cfg or HMMConfig()

    # ---- probabilities ---------------------------------------------------
    def log_emission(self, dist: np.ndarray, emission_extra: np.ndarray | None = None):
        s = self.cfg.sigma_m
        le = -0.5 * (dist / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))
        if emission_extra is not None:           # S2: hook for classifier log-probs
            le = le + emission_extra
        return le

    def log_transition(self, gc: float, route: np.ndarray):
        """route: array of route distances (inf = unreachable)."""
        b = self.cfg.beta_m
        lt = -np.abs(gc - route) / b - math.log(b)
        return np.where(np.isfinite(route), lt, self.cfg.log_floor)

    # ---- routing ---------------------------------------------------------
    def _dijkstra(self, src_node: int, cutoff: float) -> dict[int, float]:
        g = self.g
        dist = {src_node: 0.0}
        pq = [(0.0, src_node)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, math.inf) or d > cutoff:
                continue
            for a in g.out_arcs[n]:
                nd = d + g.length[a]
                m = g.v[a]
                if nd < dist.get(m, math.inf) and nd <= cutoff:
                    dist[m] = nd
                    heapq.heappush(pq, (nd, m))
        return dist

    def _route_dist(self, a1, f1, a2, f2, sp_cache) -> float:
        """S3: distance travelled from point at fraction f1 on a1 to f2 on a2."""
        g, cfg = self.g, self.cfg
        if a1 == a2:
            delta = (f2 - f1) * g.length[a1]
            return delta if delta >= 0 else abs(delta) + cfg.backward_penalty_m  # S6
        rest = (1.0 - f1) * g.length[a1]
        sp = sp_cache.get(a1)
        if sp is None:
            return math.inf
        d_nodes = sp.get(g.u[a2])
        if d_nodes is None:
            return math.inf
        return rest + d_nodes + f2 * g.length[a2]

    # ---- main entry ------------------------------------------------------
    def match(self, track: np.ndarray, emission_extra_fn=None) -> MatchResult:
        """track: (N,3) lon, lat, t.  Returns one arc id per GPS point."""
        g, cfg = self.g, self.cfg
        pts = g.proj(track[:, 0], track[:, 1])
        ts = track[:, 2]
        N = len(pts)
        out = np.full(N, -1, dtype=int)
        ncand = np.full(N, -1, dtype=int)   # -1 = skipped (S7), >=0 = real candidate count

        # S7: drop near-stationary points from the HMM, fill them in afterwards
        keep = [0]
        for i in range(1, N):
            if np.hypot(*(pts[i] - pts[keep[-1]])) >= cfg.min_step_m:
                keep.append(i)
        keep = np.array(keep)

        # candidates per kept point
        cands = []  # list of (arc_ids, dist, frac)
        for i in keep:
            ids = g.arcs_near(pts[i], cfg.candidate_radius_m)
            if len(ids):
                d, f = g.project(pts[i], ids)
                m = d <= cfg.candidate_radius_m
                ids, d, f = ids[m], d[m], f[m]
                if len(ids) > cfg.max_candidates:       # S1
                    o = np.argsort(d)[: cfg.max_candidates]
                    ids, d, f = ids[o], d[o], f[o]
            else:
                ids, d, f = np.empty(0, int), np.empty(0), np.empty(0)
            cands.append((ids, d, f))
            ncand[i] = len(ids)

        # Viterbi, chain restarts at breaks
        breaks: list[int] = []
        chain_scores = None
        chain_prev = None          # list of backpointer arrays
        chain_idx = None           # kept-index positions in the current chain
        decoded = {}               # kept position k -> arc id

        def flush():
            if chain_scores is None:
                return
            best = int(np.argmax(chain_scores))
            for pos in range(len(chain_idx) - 1, -1, -1):
                k = chain_idx[pos]
                decoded[k] = int(cands[k][0][best])
                if pos > 0:
                    best = int(chain_prev[pos][best])

        def start_chain(k):
            nonlocal chain_scores, chain_prev, chain_idx
            ids, d, f = cands[k]
            extra = emission_extra_fn(keep[k], ids) if emission_extra_fn else None
            chain_scores = self.log_emission(d, extra)
            chain_prev = [None]
            chain_idx = [k]

        last = None  # last kept position with candidates in the active chain
        for k in range(len(keep)):
            ids, d, f = cands[k]
            if len(ids) == 0:            # missing candidates -> break
                flush()
                chain_scores = chain_prev = chain_idx = None
                last = None
                breaks.append(int(keep[k]))
                continue
            if chain_scores is None:
                start_chain(k)
                last = k
                continue
            pi, pd_, pf = cands[last]
            gc = float(np.hypot(*(pts[keep[k]] - pts[keep[last]])))
            gap = ts[keep[k]] - ts[keep[last]]
            # routes from every previous candidate (cached by arc id)
            cutoff = cfg.max_route_factor * gc + cfg.route_slack_m
            sp_cache = {int(a): self._dijkstra(int(g.v[a]), cutoff) for a in pi}
            trans = np.full((len(pi), len(ids)), cfg.log_floor)
            for i, a1 in enumerate(pi):
                for j, a2 in enumerate(ids):
                    rd = self._route_dist(int(a1), pf[i], int(a2), f[j], sp_cache)
                    trans[i, j] = self.log_transition(gc, np.array([rd]))[0]
            extra = emission_extra_fn(keep[k], ids) if emission_extra_fn else None
            emis = self.log_emission(d, extra)
            tot = chain_scores[:, None] + trans
            bp = tot.argmax(0)
            new = tot[bp, np.arange(len(ids))] + emis
            if gap > cfg.max_gap_s or new.max() < cfg.log_floor / 2:   # S5
                flush()
                breaks.append(int(keep[k]))
                start_chain(k)
                last = k
                continue
            chain_scores = new
            chain_prev.append(bp)
            chain_idx.append(k)
            last = k
        flush()

        for k, arc in decoded.items():
            out[keep[k]] = arc
        # S7: fill skipped stationary points with previous match
        prev = -1
        keep_set = set(keep.tolist())
        for i in range(N):
            if out[i] == -1 and i not in keep_set:
                out[i] = prev
            elif out[i] != -1:
                prev = out[i]
        return MatchResult(arc_ids=out, breaks=breaks, n_candidates=ncand)


def build_matcher(trip: dict, cfg: HMMConfig | None = None) -> HMMMatcher:
    lon0, lat0 = trip["track"][:, 0].mean(), trip["track"][:, 1].mean()
    proj = LocalProjection(lon0, lat0)
    return HMMMatcher(RoadGraph(trip["nodes"], trip["arcs"], proj), cfg)


# --------------------------------------------------------------------------
# Quick evaluation helpers (reference route used ONLY here, never in matching)
# --------------------------------------------------------------------------
def quick_eval(pred: np.ndarray, route: np.ndarray, graph: RoadGraph) -> dict:
    """Route-level metrics, since .route is not per-GPS-point labelled.

    arc_recall   : share of reference arcs that appear in the predicted arc set
    arc_precision: share of predicted distinct arcs that are in the reference
    continuity   : share of consecutive distinct predicted arcs that are
                   connected head-to-tail
    """
    seq = [int(a) for a in pred if a >= 0]
    dedup = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    ref, ps = set(route.tolist()), set(dedup)
    ok = sum(graph.v[dedup[i]] == graph.u[dedup[i + 1]] for i in range(len(dedup) - 1))
    return dict(arc_recall=len(ref & ps) / max(len(ref), 1),
                arc_precision=len(ref & ps) / max(len(ps), 1),
                continuity=ok / max(len(dedup) - 1, 1),
                n_pred_arcs=len(dedup), n_ref_arcs=len(ref))

# --------------------------------------------------------------------------
# Shared interface used by src/evaluate.py (and the API)
# --------------------------------------------------------------------------
def match(trace, road_network, cfg: HMMConfig | None = None) -> list[int]:
    """match(trace, road_network) -> one arc id per GPS point (-1 = unmatched)."""
    trace = np.atleast_2d(np.asarray(trace, dtype=float))
    trip = {"track": trace, "nodes": np.asarray(road_network["nodes"], dtype=float),
            "arcs": np.atleast_2d(np.asarray(road_network["arcs"], dtype=int))}
    return build_matcher(trip, cfg).match(trace).arc_ids.tolist()
