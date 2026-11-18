"""Nearest-road-segment baseline for PathSnap.

Public evaluator interface: match(trace, road_network) -> list[int].
Arc IDs are zero-based row indices in the trip's .arcs file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def _project(lonlat: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular projection to metres (appropriate for city-sized trips)."""
    r = 6371008.8
    out = np.asarray(lonlat, dtype=float).copy()
    out[..., 0] *= (np.pi / 180) * r * np.cos(np.deg2rad(lat0))
    out[..., 1] *= (np.pi / 180) * r
    return out


def _dist2(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance to finite segments, not infinite supporting lines."""
    ab = b - a
    denom = np.einsum('ij,ij->i', ab, ab)
    t = np.divide(np.einsum('ij,ij->i', point - a, ab), denom,
                  out=np.zeros_like(denom), where=denom > 0)
    t = np.clip(t, 0.0, 1.0)
    delta = point - (a + t[:, None] * ab)
    return np.einsum('ij,ij->i', delta, delta)


def match(trace, road_network) -> list[int]:
    """Snap each GPS point independently to its closest road arc.

    Uses a Shapely STRtree when available; falls back to exact chunked NumPy
    distance calculations otherwise. Reverse directed arcs may tie; the lowest
    arc ID wins. No ground-truth data is used in matching.
    """
    trace = np.asarray(trace, dtype=float)
    nodes = np.asarray(road_network['nodes'], dtype=float)
    arcs = np.asarray(road_network['arcs'], dtype=np.int64)
    if trace.ndim != 2 or trace.shape[1] < 2:
        raise ValueError('trace must have shape (N, >=2)')
    if nodes.ndim != 2 or nodes.shape[1] != 2 or arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError('nodes and arcs must have shape (K,2) and (E,2)')
    if len(trace) == 0:
        return []
    if len(arcs) == 0 or len(nodes) == 0 or np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError('road network has no valid arcs')
    if not (np.isfinite(trace[:, :2]).all() and np.isfinite(nodes).all()):
        raise ValueError('coordinates must be finite')

    lat0 = float(np.mean(trace[:, 1]))
    xy = _project(trace[:, :2], lat0)
    projected_nodes = _project(nodes, lat0)
    a, b = projected_nodes[arcs[:, 0]], projected_nodes[arcs[:, 1]]

    try:
        from shapely.geometry import LineString, Point
        from shapely.strtree import STRtree
        lines = [LineString((u, v)) for u, v in zip(a, b)]
        tree = STRtree(lines)
        # query_nearest(all_matches=True) captures ties, including reverse arcs.
        indices, distances = tree.query_nearest(
            [Point(p) for p in xy], all_matches=True, return_distance=True
        )
        answer = np.full(len(xy), -1, dtype=np.int64)
        # Choose lowest arc ID among geometrically equidistant candidates.
        for point_id, arc_id in zip(indices[0], indices[1]):
            if answer[point_id] < 0 or arc_id < answer[point_id]:
                answer[point_id] = arc_id
        return answer.tolist()
    except ImportError:
        # Exact fallback: O(N*E) but bounded memory.
        result = []
        for p in xy:
            best_id, best_d2 = -1, float('inf')
            for start in range(0, len(arcs), 8192):
                stop = min(start + 8192, len(arcs))
                d2 = _dist2(p, a[start:stop], b[start:stop])
                local = int(np.argmin(d2))
                if d2[local] < best_d2:
                    best_d2 = float(d2[local])
                    best_id = start + local
            result.append(best_id)
        return result


def _load_trip(folder: Path):
    trip_id = folder.name
    trace = np.loadtxt(folder / f'{trip_id}.track', ndmin=2)
    nodes = np.loadtxt(folder / f'{trip_id}.nodes', ndmin=2)
    arcs = np.loadtxt(folder / f'{trip_id}.arcs', dtype=np.int64, ndmin=2)
    return trace, {'nodes': nodes, 'arcs': arcs}


def main():
    parser = argparse.ArgumentParser(description='Run nearest-segment PathSnap baseline')
    parser.add_argument('--data-dir', type=Path, required=True, help='train or student_test directory')
    parser.add_argument('--max-trips', type=int, default=3)
    args = parser.parse_args()
    if args.max_trips < 1:
        parser.error('--max-trips must be >= 1')
    folders = sorted(p for p in args.data_dir.iterdir() if p.is_dir())[:args.max_trips]
    if not folders:
        parser.error('No trip folders found')
    total = 0
    for folder in folders:
        trace, network = _load_trip(folder)
        predictions = match(trace, network)
        total += len(predictions)
        print(f'{folder.name}: matched {len(predictions)}/{len(trace)} GPS points')
        route_path = folder / f'{folder.name}.route'
        if route_path.exists():
            route = set(np.loadtxt(route_path, dtype=np.int64, ndmin=1).tolist())
            in_route = sum(arc_id in route for arc_id in predictions)
            print(f'  Reference-route arc membership: {in_route}/{len(predictions)} '
                  f'({in_route / len(predictions):.2%}); NOT point-level accuracy')
    print(f'TOTAL: {total} points matched across {len(folders)} trips')
    print('For derived per-point accuracy use: python -m src.evaluate '
          '--data-dir <partition> --matcher src.baseline:match --name baseline')


if __name__ == '__main__':
    main()
