"""Tests for tng_smbhb.bh_details (blackhole_details.hdf5 pre-merger mass lookup).

All tests build a synthetic miniature ``blackhole_details.hdf5`` in a tmp dir
so nothing here touches the real 5.2 GB TNG100-1 file.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tng_smbhb.bh_details import (
    DetailsIndex,
    _pick_premerger_row,
    batch_lookup_premerger_masses,
    load_details_index,
    lookup_premerger_mass,
)

# TNG internal mass unit scale with h=1.0, used throughout these tests
# to keep the arithmetic transparent (mass_msun = mass_tng * 1e10).
_H = 1.0
_SCALE = 1e10  # M_sun per TNG unit when h = 1


def _make_synthetic_details(
    path: Path,
    *,
    bh_ids: list[int],
    records_per_bh: list[list[tuple[float, float]]],
) -> None:
    """Write a miniature blackhole_details.hdf5.

    ``records_per_bh[i]`` is a list of ``(time, mass_tng_internal)`` tuples for
    the BH with particle id ``bh_ids[i]``.  Must be sorted by time ascending.
    IDs get sorted ascending on write (as in the real file).
    """
    order = np.argsort(bh_ids)
    ids_sorted = [bh_ids[i] for i in order]
    runs_sorted = [records_per_bh[i] for i in order]

    first_index: list[int] = []
    num_entries: list[int] = []
    times: list[float] = []
    masses: list[float] = []

    cursor = 0
    for run in runs_sorted:
        first_index.append(cursor)
        num_entries.append(len(run))
        for t, m in run:
            times.append(t)
            masses.append(m)
        cursor += len(run)

    with h5py.File(path, "w") as f:
        f.create_dataset("id", data=np.zeros(len(times), dtype=np.uint64))
        f.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        f.create_dataset("mass", data=np.asarray(masses, dtype=np.float64))
        grp = f.create_group("unique")
        grp.create_dataset("id", data=np.asarray(ids_sorted, dtype=np.uint64))
        grp.create_dataset(
            "first_index", data=np.asarray(first_index, dtype=np.int64)
        )
        grp.create_dataset(
            "num_entries", data=np.asarray(num_entries, dtype=np.int64)
        )


@pytest.fixture()
def tiny_details_file(tmp_path: Path) -> Path:
    """Three BHs with a few records each."""
    path = tmp_path / "details.hdf5"
    _make_synthetic_details(
        path,
        bh_ids=[100, 42, 7],
        records_per_bh=[
            # BH 100: grows 1e-5 -> 5e-5 over a = 0.2 .. 0.9
            [(0.2, 1e-5), (0.5, 3e-5), (0.9, 5e-5)],
            # BH 42: 2 records
            [(0.3, 2e-6), (0.7, 8e-6)],
            # BH 7: single record (edge case)
            [(0.5, 1e-7)],
        ],
    )
    return path


# --------------------------------------------------------------------------
# DetailsIndex / load_details_index
# --------------------------------------------------------------------------


def test_load_details_index_basic(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    assert isinstance(idx, DetailsIndex)
    assert idx.num_blackholes == 3
    # IDs are sorted ascending on write
    assert list(idx.id) == [7, 42, 100]
    # Row offsets align with record counts (7, 42, 100 => 1, 2, 3 records)
    assert list(idx.num_entries) == [1, 2, 3]
    assert list(idx.first_index) == [0, 1, 3]


def test_load_details_index_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_details_index(tmp_path / "nope.hdf5")


def test_load_details_index_no_unique_group(tmp_path: Path) -> None:
    path = tmp_path / "broken.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=np.zeros(3, dtype=np.float64))
    with pytest.raises(ValueError, match="no '/unique' subgroup"):
        load_details_index(path)


def test_load_details_index_rejects_unsorted_ids(tmp_path: Path) -> None:
    path = tmp_path / "unsorted.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=np.zeros(2, dtype=np.float64))
        f.create_dataset("mass", data=np.zeros(2, dtype=np.float64))
        grp = f.create_group("unique")
        grp.create_dataset("id", data=np.array([5, 3], dtype=np.uint64))
        grp.create_dataset("first_index", data=np.array([0, 1], dtype=np.int64))
        grp.create_dataset("num_entries", data=np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="not sorted ascending"):
        load_details_index(path)


# --------------------------------------------------------------------------
# _pick_premerger_row
# --------------------------------------------------------------------------


def test_pick_premerger_row_picks_last_before() -> None:
    times = np.array([0.2, 0.5, 0.9])
    # merger at a = 0.7 -> last record with t<0.7 is row 1 (t=0.5)
    assert _pick_premerger_row(times, 0.7) == 1


def test_pick_premerger_row_all_after_falls_back_to_zero() -> None:
    times = np.array([0.8, 0.9])
    # merger at a = 0.5, every record is later -> fall back to row 0
    assert _pick_premerger_row(times, 0.5) == 0


def test_pick_premerger_row_exact_match_excluded() -> None:
    times = np.array([0.2, 0.5, 0.9])
    # The merger record (t==0.5) itself reflects post-merger state, so
    # we want t < 0.5 strictly -> row 0.
    assert _pick_premerger_row(times, 0.5) == 0


# --------------------------------------------------------------------------
# lookup_premerger_mass (single-query convenience wrapper)
# --------------------------------------------------------------------------


def test_lookup_premerger_mass_happy_path(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    # BH 100 at merger time 0.7 -> last record before is (0.5, 3e-5) ->
    # M_sun = 3e-5 * 1e10 / h = 3e5
    m = lookup_premerger_mass(idx, bh_id=100, merger_time=0.7, hubble_h=_H)
    assert m is not None
    assert m == pytest.approx(3e5, rel=1e-10)


def test_lookup_premerger_mass_missing_id_returns_none(
    tiny_details_file: Path,
) -> None:
    idx = load_details_index(tiny_details_file)
    assert lookup_premerger_mass(idx, bh_id=999, merger_time=0.5) is None


def test_lookup_premerger_mass_id_above_max_returns_none(
    tiny_details_file: Path,
) -> None:
    idx = load_details_index(tiny_details_file)
    # 10_000 > all ids (max 100); searchsorted returns num_blackholes
    assert lookup_premerger_mass(idx, bh_id=10_000, merger_time=0.5) is None


# --------------------------------------------------------------------------
# batch_lookup_premerger_masses
# --------------------------------------------------------------------------


def test_batch_lookup_matches_single(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    # Mix of found + one missing
    ids = np.array([100, 42, 999, 7], dtype=np.uint64)
    times = np.array([0.95, 0.5, 0.5, 0.6], dtype=np.float64)
    masses, found = batch_lookup_premerger_masses(
        idx, ids, times, hubble_h=_H, progress_every=0
    )
    assert list(found) == [True, True, False, True]

    # BH 100 at 0.95: last record before is (0.9, 5e-5) -> 5e5 M_sun
    assert masses[0] == pytest.approx(5e5, rel=1e-10)
    # BH 42 at 0.5: (0.3, 2e-6) -> 2e4 M_sun
    assert masses[1] == pytest.approx(2e4, rel=1e-10)
    # missing -> NaN
    assert np.isnan(masses[2])
    # BH 7 at 0.6: single record (0.5, 1e-7) -> 1e3 M_sun
    assert masses[3] == pytest.approx(1e3, rel=1e-10)


def test_batch_lookup_shape_mismatch_raises(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    with pytest.raises(ValueError, match="same length"):
        batch_lookup_premerger_masses(
            idx,
            bh_ids=np.array([1, 2], dtype=np.uint64),
            merger_times=np.array([0.5], dtype=np.float64),
        )


def test_batch_lookup_all_missing(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    ids = np.array([1, 2, 3], dtype=np.uint64)  # none of these exist
    times = np.full(3, 0.5)
    masses, found = batch_lookup_premerger_masses(idx, ids, times, progress_every=0)
    assert not found.any()
    assert np.all(np.isnan(masses))


def test_batch_lookup_hubble_h_scales_masses(tiny_details_file: Path) -> None:
    idx = load_details_index(tiny_details_file)
    ids = np.array([100], dtype=np.uint64)
    times = np.array([0.95])
    m_h1, _ = batch_lookup_premerger_masses(idx, ids, times, hubble_h=1.0, progress_every=0)
    m_h2, _ = batch_lookup_premerger_masses(idx, ids, times, hubble_h=0.5, progress_every=0)
    # Halving h doubles the M_sun conversion.
    assert m_h2[0] == pytest.approx(2.0 * m_h1[0], rel=1e-12)
