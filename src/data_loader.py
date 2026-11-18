"""Load PathSnap trip files from an external data directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TRIP_SUFFIXES = ("track", "nodes", "arcs", "route")


def _has_trip_files(folder: Path) -> bool:
    """Recognize a trip folder by a file named after its directory."""
    return any((folder / f"{folder.name}.{suffix}").is_file() for suffix in TRIP_SUFFIXES)


def _find_trip_folders(data_dir: Path) -> list[Path]:
    """Find direct trip folders or one-level partition folders of trips."""
    direct: list[Path] = []
    nested: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if _has_trip_files(child):
            direct.append(child)
            continue
        nested.extend(
            grandchild
            for grandchild in sorted(child.iterdir())
            if grandchild.is_dir() and _has_trip_files(grandchild)
        )
    return direct + nested


def _load_matrix(path: Path, columns: int, dtype: Any) -> np.ndarray:
    """Load one headerless whitespace-separated table with a fixed width."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required trip file: {path.name}")
    try:
        values = np.loadtxt(path, dtype=dtype, ndmin=2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse {path.name}: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(
            f"{path.name} must have {columns} columns; found shape {values.shape}"
        )
    return values


def load_data(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load all trip folders under data_dir.

    The directory may contain trip folders directly or one partition level
    (for example train/<trip_id>/). Each result contains a float GPS trace,
    the trip-specific nodes and directed arcs, and integer reference arc IDs.
    If a trip has no .route file, route is an empty int64 array.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    trip_folders = _find_trip_folders(root)
    if not trip_folders:
        raise FileNotFoundError(
            f"No trip folders found in {root}; expected <trip_id>/<trip_id>.track"
        )

    data: dict[str, dict[str, Any]] = {}
    for folder in trip_folders:
        trip_id = folder.name
        if trip_id in data:
            raise ValueError(
                f"Duplicate trip ID {trip_id!r} found under {root}; "
                "pass one partition directory at a time"
            )

        trace = _load_matrix(folder / f"{trip_id}.track", 3, np.float64)
        nodes = _load_matrix(folder / f"{trip_id}.nodes", 2, np.float64)
        arcs = _load_matrix(folder / f"{trip_id}.arcs", 2, np.int64)

        route_path = folder / f"{trip_id}.route"
        if route_path.is_file():
            try:
                route = np.loadtxt(route_path, dtype=np.int64, ndmin=1).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Could not parse {route_path.name}: {exc}") from exc
        else:
            route = np.empty(0, dtype=np.int64)

        data[trip_id] = {
            "trace": trace,
            "road_network": {"nodes": nodes, "arcs": arcs},
            "route": route,
        }

    return data
