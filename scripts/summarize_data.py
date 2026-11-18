"""Print a compact inventory of PathSnap trips in an external data directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_data  # noqa: E402


def largest_time_gap(trace: np.ndarray) -> float | None:
    """Return the largest finite consecutive timestamp difference in seconds."""
    if len(trace) < 2:
        return None
    gaps = np.diff(trace[:, 2])
    gaps = gaps[np.isfinite(gaps)]
    return float(gaps.max()) if len(gaps) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print GPS, road-network, route, and time-gap counts per trip."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="External directory containing trip folders (or partition folders).",
    )
    args = parser.parse_args()

    trips = load_data(args.data_dir)
    print(f"Trips loaded: {len(trips)}")
    print(
        f"{'Trip ID':<16} {'GPS points':>11} {'Nodes':>10} "
        f"{'Arcs':>10} {'Route arcs':>12} {'Largest gap (s)':>16}"
    )
    print("-" * 81)
    for trip_id, trip in trips.items():
        gap = largest_time_gap(trip["trace"])
        gap_text = "N/A" if gap is None else f"{gap:.3f}"
        print(
            f"{trip_id:<16} {len(trip['trace']):>11} "
            f"{len(trip['road_network']['nodes']):>10} "
            f"{len(trip['road_network']['arcs']):>10} "
            f"{len(trip['route']):>12} {gap_text:>16}"
        )


if __name__ == "__main__":
    main()
