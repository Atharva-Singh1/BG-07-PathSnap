#!/usr/bin/env python3
"""Dataset inventory and EDA for one PathSnap partition (real data only)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_data
from src.evaluate import project_local


INVENTORY_COLUMNS = [
    "partition_name", "trip_id", "n_gps_points", "n_nodes", "n_arcs",
    "n_route_arcs", "duration_s", "trip_extent_km",
    "median_time_gap_s", "max_time_gap_s",
]
MISSING_COLUMNS = [
    "partition_name", "trip_id", "file", "nan_inf_values",
    "duplicate_timestamps", "non_increasing_timestamps",
    "gaps_gt_1s", "gaps_gt_10s", "gaps_gt_60s",
    "arcs_node_ids_out_of_range", "route_arc_ids_out_of_range",
    "nodes_not_used_by_any_arc",
]
MISSING_COUNT_COLUMNS = MISSING_COLUMNS[3:]


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


def _trace_stats(trace: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    ts = np.asarray(trace[:, 2], dtype=float)
    finite_ts = _finite_values(ts)
    duration = float(np.max(finite_ts) - np.min(finite_ts)) if finite_ts.size else float("nan")
    gaps = np.diff(ts)
    valid_gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    median_gap = float(np.median(valid_gaps)) if valid_gaps.size else float("nan")
    max_gap = float(np.max(valid_gaps)) if valid_gaps.size else float("nan")
    return duration, median_gap, max_gap, gaps


def _trip_extent_km(trace: np.ndarray) -> float:
    coords = np.asarray(trace[:, :2], dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if not len(coords):
        return float("nan")
    xy = project_local(coords, float(np.mean(coords[:, 1])))
    return float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) / 1000.0)


def _trip_inventory(partition: str, trip_id: str, trip: dict) -> dict:
    trace = np.asarray(trip["trace"], dtype=float)
    nodes = np.asarray(trip["road_network"]["nodes"], dtype=float)
    arcs = np.asarray(trip["road_network"]["arcs"], dtype=np.int64)
    route = np.asarray(trip["route"], dtype=np.int64).reshape(-1)
    duration, median_gap, max_gap, _ = _trace_stats(trace)
    return {
        "partition_name": partition,
        "trip_id": trip_id,
        "n_gps_points": int(len(trace)),
        "n_nodes": int(len(nodes)),
        "n_arcs": int(len(arcs)),
        "n_route_arcs": int(len(route)),
        "duration_s": duration,
        "trip_extent_km": _trip_extent_km(trace),
        "median_time_gap_s": median_gap,
        "max_time_gap_s": max_gap,
    }


def _trip_missing_rows(partition: str, trip_id: str, trip: dict) -> list[dict]:
    trace = np.asarray(trip["trace"], dtype=float)
    nodes = np.asarray(trip["road_network"]["nodes"], dtype=float)
    arcs = np.asarray(trip["road_network"]["arcs"], dtype=np.int64)
    route = np.asarray(trip["route"], dtype=np.int64).reshape(-1)

    rows = []
    arrays = {
        ".track": trace,
        ".nodes": nodes,
        ".arcs": arcs,
        ".route": route,
    }
    for suffix, values in arrays.items():
        count = int(np.count_nonzero(~np.isfinite(values))) if values.size else 0
        rows.append({
            "partition_name": partition,
            "trip_id": trip_id,
            "file": suffix,
            "nan_inf_values": count,
            "duplicate_timestamps": 0,
            "non_increasing_timestamps": 0,
            "gaps_gt_1s": 0,
            "gaps_gt_10s": 0,
            "gaps_gt_60s": 0,
            "arcs_node_ids_out_of_range": 0,
            "route_arc_ids_out_of_range": 0,
            "nodes_not_used_by_any_arc": 0,
        })

    ts = trace[:, 2] if trace.ndim == 2 and trace.shape[1] >= 3 else np.empty(0)
    finite_ts = _finite_values(ts)
    gaps = np.diff(ts)
    valid_pairs = np.isfinite(ts[:-1]) & np.isfinite(ts[1:]) if len(ts) > 1 else np.array([], dtype=bool)
    valid_diffs = gaps[valid_pairs]
    track_row = rows[0]
    track_row["duplicate_timestamps"] = int(finite_ts.size - np.unique(finite_ts).size)
    track_row["non_increasing_timestamps"] = int(np.count_nonzero(valid_diffs <= 0))
    positive_gaps = valid_diffs[valid_diffs > 0]
    track_row["gaps_gt_1s"] = int(np.count_nonzero(positive_gaps > 1))
    track_row["gaps_gt_10s"] = int(np.count_nonzero(positive_gaps > 10))
    track_row["gaps_gt_60s"] = int(np.count_nonzero(positive_gaps > 60))

    if arcs.size:
        valid_arc_nodes = (arcs >= 0) & (arcs < len(nodes))
        invalid_arc_rows = ~np.all(valid_arc_nodes, axis=1)
        rows[2]["arcs_node_ids_out_of_range"] = int(np.count_nonzero(invalid_arc_rows))
        used = np.zeros(len(nodes), dtype=bool)
        valid_ids = arcs[valid_arc_nodes]
        used[valid_ids] = True
        rows[1]["nodes_not_used_by_any_arc"] = int(np.count_nonzero(~used))
    else:
        rows[1]["nodes_not_used_by_any_arc"] = int(len(nodes))

    if route.size:
        rows[3]["route_arc_ids_out_of_range"] = int(
            np.count_nonzero((route < 0) | (route >= len(arcs)))
        )
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_sampling(partition: str, data: dict, path: Path) -> None:
    all_gaps, point_counts = [], []
    for trip in data.values():
        trace = np.asarray(trip["trace"], dtype=float)
        point_counts.append(len(trace))
        if len(trace) > 1:
            ts = trace[:, 2]
            gaps = np.diff(ts)
            all_gaps.extend(gaps[np.isfinite(gaps) & (gaps > 0)].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if all_gaps:
        axes[0].hist(all_gaps, bins=50, log=True, color="steelblue", edgecolor="white")
    else:
        axes[0].text(0.5, 0.5, "No positive finite time gaps", ha="center", va="center",
                     transform=axes[0].transAxes)
    axes[0].set(title=f"GPS time gaps — partition: {partition}", xlabel="Gap (s)",
                ylabel="Count (log scale)")
    axes[0].set_yscale("log")
    axes[1].hist(point_counts, bins=min(20, max(1, len(point_counts))),
                 color="darkorange", edgecolor="white")
    axes[1].set(title=f"GPS points per trip — partition: {partition}",
                xlabel="GPS points", ylabel="Trips")
    fig.suptitle(f"Sampling overview — partition: {partition}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_trajectories(partition: str, data: dict, path: Path) -> None:
    selected = list(sorted(data))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = np.asarray(axes).reshape(-1)
    for ax, trip_id in zip(axes, selected):
        trace = np.asarray(data[trip_id]["trace"], dtype=float)
        coords = trace[:, :2]
        valid = np.isfinite(coords).all(axis=1)
        coords = coords[valid]
        if len(coords):
            xy = project_local(coords, float(np.mean(coords[:, 1])))
            ax.plot(xy[:, 0], xy[:, 1], color="steelblue", linewidth=1.1, label="GPS trajectory")
            ax.scatter(*xy[0], color="green", marker="o", label="Start", zorder=3)
            ax.scatter(*xy[-1], color="red", marker="x", label="End", zorder=3)
        ax.set_title(f"Trip {trip_id} — partition: {partition}")
        ax.set_xlabel("East (m, local projection)")
        ax.set_ylabel("North (m, local projection)")
        ax.grid(alpha=0.2)
        if len(coords):
            ax.legend()
    for ax in axes[len(selected):]:
        ax.set_visible(False)
    fig.suptitle(f"GPS trajectories — partition: {partition}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_network(partition: str, trip_id: str, trip: dict, path: Path) -> None:
    trace = np.asarray(trip["trace"], dtype=float)
    nodes = np.asarray(trip["road_network"]["nodes"], dtype=float)
    arcs = np.asarray(trip["road_network"]["arcs"], dtype=np.int64)
    route = np.asarray(trip["route"], dtype=np.int64).reshape(-1)
    coords = trace[:, :2]
    valid_gps = np.isfinite(coords).all(axis=1)
    gps = coords[valid_gps]
    if len(gps) == 0 or len(nodes) == 0:
        raise ValueError(f"Trip {trip_id} has no finite GPS points or no nodes to plot")
    lat0 = float(np.mean(gps[:, 1]))
    gps_xy = project_local(gps, lat0)
    node_xy = project_local(nodes, lat0)
    margin = 200.0
    xmin, xmax = float(gps_xy[:, 0].min() - margin), float(gps_xy[:, 0].max() + margin)
    ymin, ymax = float(gps_xy[:, 1].min() - margin), float(gps_xy[:, 1].max() + margin)

    valid_arc_mask = np.all((arcs >= 0) & (arcs < len(nodes)), axis=1)
    valid_arc_ids = np.flatnonzero(valid_arc_mask)
    valid_arcs = arcs[valid_arc_mask]
    start = node_xy[valid_arcs[:, 0]]
    end = node_xy[valid_arcs[:, 1]]
    arc_min = np.minimum(start, end)
    arc_max = np.maximum(start, end)
    visible = ((arc_max[:, 0] >= xmin) & (arc_min[:, 0] <= xmax) &
               (arc_max[:, 1] >= ymin) & (arc_min[:, 1] <= ymax))
    fig, ax = plt.subplots(figsize=(10, 8))
    if np.any(visible):
        network_segments = np.stack((start[visible], end[visible]), axis=1)
        ax.add_collection(LineCollection(
            network_segments, colors="lightgrey", linewidths=0.7, alpha=0.8,
            label="Road network (GPS bounds +200 m)",
        ))
    valid_route = route[(route >= 0) & (route < len(arcs))]
    valid_route = valid_route[valid_arc_mask[valid_route]] if len(valid_route) else valid_route
    if len(valid_route):
        route_segments = np.stack((node_xy[arcs[valid_route, 0]],
                                   node_xy[arcs[valid_route, 1]]), axis=1)
        ax.add_collection(LineCollection(
            route_segments, colors="green", linewidths=2, alpha=0.9,
            label="Reference route",
        ))
    ax.scatter(gps_xy[:, 0], gps_xy[:, 1], s=8, color="blue", alpha=0.7, label="GPS points")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m, local projection)")
    ax.set_ylabel("North (m, local projection)")
    ax.set_title(f"Road network and reference route — trip {trip_id}, partition: {partition}")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _print_summary(partition: str, inventory: list[dict], missing_rows: list[dict]) -> None:
    total_points = sum(row["n_gps_points"] for row in inventory)
    largest_trip, largest_gap = "-", float("nan")
    for row in inventory:
        gap = row["max_time_gap_s"]
        if np.isfinite(gap) and (not np.isfinite(largest_gap) or gap > largest_gap):
            largest_trip, largest_gap = row["trip_id"], gap
    totals = {column: sum(int(row[column]) for row in missing_rows)
              for column in MISSING_COUNT_COLUMNS}
    print(f"Partition: {partition}")
    print(f"Trips: {len(inventory)}; total GPS points: {total_points}")
    print(f"Largest positive time gap: {largest_gap:.6g} s (trip {largest_trip})")
    print("Totals by missing/invalid column: " +
          ", ".join(f"{column}={totals[column]}" for column in MISSING_COUNT_COLUMNS))
    structural_columns = (
        "nan_inf_values", "duplicate_timestamps", "non_increasing_timestamps",
        "arcs_node_ids_out_of_range", "route_arc_ids_out_of_range",
        "nodes_not_used_by_any_arc",
    )
    if any(totals[column] for column in structural_columns):
        print("Missing or structurally invalid values were found; inspect the partition-specific missing-values CSV.")
    else:
        print("Missing/non-finite values and structural ID issues: none detected in loaded arrays.")
    if inventory:
        extents = [row["trip_extent_km"] for row in inventory
                   if np.isfinite(row["trip_extent_km"])]
        print(f"Trip extent range: {min(extents):.3f} to {max(extents):.3f} km" if extents
              else "Trip extent range: unavailable")
    print(f"CSV inventory: {partition}_inventory.csv")
    print(f"CSV missing values: {partition}_missing_values.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory and plot one PathSnap data partition.")
    parser.add_argument("--data-dir", required=True, help="one partition directory (train or student_test)")
    parser.add_argument("--partition-name", required=True, choices=("train", "student_test"))
    parser.add_argument("--out-dir", default="results/eda", help="directory for partition CSV files")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    partition = args.partition_name
    inventory = [_trip_inventory(partition, trip_id, trip) for trip_id, trip in sorted(data.items())]
    missing_rows = [
        row for trip_id, trip in sorted(data.items())
        for row in _trip_missing_rows(partition, trip_id, trip)
    ]

    out_dir = Path(args.out_dir)
    figure_dir = out_dir.parent / "figures"
    _write_csv(out_dir / f"{partition}_inventory.csv", INVENTORY_COLUMNS, inventory)
    _write_csv(out_dir / f"{partition}_missing_values.csv", MISSING_COLUMNS, missing_rows)
    _save_sampling(partition, data, figure_dir / f"eda_{partition}_sampling.png")
    _save_trajectories(partition, data, figure_dir / f"eda_{partition}_trajectories.png")
    first_trip_id = sorted(data)[0]
    _save_network(partition, first_trip_id, data[first_trip_id],
                  figure_dir / f"eda_{partition}_network.png")
    _print_summary(partition, inventory, missing_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
