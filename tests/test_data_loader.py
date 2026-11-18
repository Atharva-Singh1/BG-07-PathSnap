"""Tests for src/data_loader.py on tiny synthetic trip folders (no real data)."""
import numpy as np
import pytest

from src.data_loader import load_data


def make_trip(folder, trip_id):
    d = folder / trip_id
    d.mkdir(parents=True)
    np.savetxt(d / f"{trip_id}.track", [[8.0, 50.0, 1.0], [8.001, 50.0, 2.0]])
    np.savetxt(d / f"{trip_id}.nodes", [[8.0, 50.0], [8.001, 50.0]])
    np.savetxt(d / f"{trip_id}.arcs", [[0, 1], [1, 0]], fmt="%d")
    np.savetxt(d / f"{trip_id}.route", [0], fmt="%d")


def test_loads_one_partition_directly(tmp_path):
    make_trip(tmp_path / "train", "t1")
    make_trip(tmp_path / "train", "t2")
    data = load_data(tmp_path / "train")
    assert sorted(data) == ["t1", "t2"]
    assert data["t1"]["trace"].shape == (2, 3)
    assert data["t1"]["road_network"]["arcs"].dtype == np.int64


def test_loads_single_nested_partition(tmp_path):
    make_trip(tmp_path / "train", "t1")
    assert sorted(load_data(tmp_path)) == ["t1"]


def test_rejects_several_partitions(tmp_path):
    make_trip(tmp_path / "train", "t1")
    make_trip(tmp_path / "student_test", "t2")
    with pytest.raises(ValueError, match="more than one partition"):
        load_data(tmp_path)


def test_rejects_direct_trips_mixed_with_partition(tmp_path):
    make_trip(tmp_path, "t1")
    make_trip(tmp_path / "student_test", "t2")
    with pytest.raises(ValueError, match="more than one partition"):
        load_data(tmp_path)
