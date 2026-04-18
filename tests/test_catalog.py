"""Tests for tng_smbhb.catalog — TNGMergerCatalog construction and HDF5 loading.

All tests are self-contained: no real TNG file is required.  The HDF5
round-trip test writes a tiny synthetic file using h5py and a pytest
tmp_path fixture.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tng_smbhb.catalog import TNGMergerCatalog, catalog_from_arrays, load_tng_hdf5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H: float = 0.6774  # default TNG Hubble h
_MASS_UNIT: float = 1.0e10 / _H  # conversion factor: TNG unit → M_sun


# ---------------------------------------------------------------------------
# catalog_from_arrays: ordering
# ---------------------------------------------------------------------------


class TestCatalogFromArraysSorting:
    """catalog_from_arrays must ensure m1 >= m2 element-wise."""

    def test_swap_when_m2_greater(self) -> None:
        """Reversed inputs are correctly reordered so m1 >= m2."""
        m1_in = np.array([1.0e7, 5.0e8], dtype=np.float64)
        m2_in = np.array([9.0e8, 2.0e7], dtype=np.float64)  # both > m1_in
        a = np.array([0.5, 0.8], dtype=np.float64)

        cat = catalog_from_arrays(m1_in, m2_in, a, simulation="test", hubble_h=_H)

        assert np.all(cat.m1_msun >= cat.m2_msun), "m1 must be >= m2 for every event."

    def test_already_ordered_unchanged(self) -> None:
        """When m1 > m2 on input, values are not swapped."""
        m1_in = np.array([1.0e9, 2.0e9], dtype=np.float64)
        m2_in = np.array([1.0e8, 5.0e8], dtype=np.float64)
        a = np.array([0.6, 0.9], dtype=np.float64)

        cat = catalog_from_arrays(m1_in, m2_in, a, simulation="test", hubble_h=_H)

        np.testing.assert_array_equal(cat.m1_msun, m1_in)
        np.testing.assert_array_equal(cat.m2_msun, m2_in)


# ---------------------------------------------------------------------------
# catalog_from_arrays: redshift conversion
# ---------------------------------------------------------------------------


class TestRedshiftConversion:
    """z = 1/a - 1 must be computed exactly."""

    def test_scale_factor_half_gives_z_one(self) -> None:
        """scale_factor=0.5 must yield redshift=1.0 exactly."""
        cat = catalog_from_arrays(
            [1.0e8],
            [1.0e7],
            [0.5],
            simulation="test",
            hubble_h=_H,
        )
        assert cat.redshift[0] == pytest.approx(1.0, rel=1e-12)

    def test_scale_factor_one_gives_z_zero(self) -> None:
        """scale_factor=1.0 (present day) must yield redshift=0.0."""
        cat = catalog_from_arrays(
            [1.0e8],
            [1.0e7],
            [1.0],
            simulation="test",
            hubble_h=_H,
        )
        assert cat.redshift[0] == pytest.approx(0.0, abs=1e-15)

    def test_multiple_scale_factors(self) -> None:
        """Vectorised redshift conversion is self-consistent."""
        a_arr = np.array([0.25, 0.5, 1.0], dtype=np.float64)
        cat = catalog_from_arrays(
            [1e8, 1e8, 1e8],
            [1e7, 1e7, 1e7],
            a_arr,
            simulation="test",
            hubble_h=_H,
        )
        expected = 1.0 / a_arr - 1.0
        np.testing.assert_allclose(cat.redshift, expected, rtol=1e-15)


# ---------------------------------------------------------------------------
# catalog_from_arrays: validation errors
# ---------------------------------------------------------------------------


class TestCatalogFromArraysValidation:
    """catalog_from_arrays must raise ValueError for invalid inputs."""

    def test_negative_m1_raises(self) -> None:
        with pytest.raises(ValueError, match="m1_msun"):
            catalog_from_arrays([-1.0e8], [1.0e7], [0.5], simulation="t", hubble_h=_H)

    def test_negative_m2_raises(self) -> None:
        with pytest.raises(ValueError, match="m2_msun"):
            catalog_from_arrays([1.0e8], [-1.0e7], [0.5], simulation="t", hubble_h=_H)

    def test_zero_m1_raises(self) -> None:
        with pytest.raises(ValueError, match="m1_msun"):
            catalog_from_arrays([0.0], [1.0e7], [0.5], simulation="t", hubble_h=_H)

    def test_nan_m2_raises(self) -> None:
        with pytest.raises(ValueError, match="m2_msun"):
            catalog_from_arrays([1.0e8], [float("nan")], [0.5], simulation="t", hubble_h=_H)

    def test_mismatched_m1_m2_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            catalog_from_arrays(
                [1.0e8, 2.0e8],
                [1.0e7],
                [0.5, 0.6],
                simulation="t",
                hubble_h=_H,
            )

    def test_mismatched_scale_factor_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            catalog_from_arrays(
                [1.0e8, 2.0e8],
                [1.0e7, 2.0e7],
                [0.5],
                simulation="t",
                hubble_h=_H,
            )

    def test_negative_scale_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_factor"):
            catalog_from_arrays([1.0e8], [1.0e7], [-0.5], simulation="t", hubble_h=_H)


# ---------------------------------------------------------------------------
# Frozen dataclass: mutation raises
# ---------------------------------------------------------------------------


class TestFrozenDataclass:
    """TNGMergerCatalog is a frozen dataclass; field assignment must raise."""

    def test_mutation_raises_frozen_instance_error(self) -> None:
        cat = catalog_from_arrays(
            [1.0e9],
            [1.0e8],
            [0.7],
            simulation="test",
            hubble_h=_H,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cat.m1_msun = np.array([9.9e9])  # type: ignore[misc]

    def test_n_mergers_mutation_raises(self) -> None:
        cat = catalog_from_arrays(
            [1.0e9],
            [1.0e8],
            [0.7],
            simulation="test",
            hubble_h=_H,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cat.n_mergers = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HDF5 round-trip test
# ---------------------------------------------------------------------------


class TestHDF5RoundTrip:
    """Write a tiny synthetic HDF5 file and verify load_tng_hdf5 output.

    These tests cover the Phase 1b equal-mass proxy loader.  The real TNG
    ``blackhole_mergers.hdf5`` file has a documented data bug where
    ``mass_out`` carries the dynamical cell mass instead of the primary BH
    mass (see ``catalog.py`` module docstring), so the loader reads only
    ``time`` and ``mass_in`` and sets ``m1 = m2 = mass_in``.
    """

    def test_real_tng_schema_lowercase(self, tmp_path) -> None:
        """Real TNG schema: lowercase root-level ``time`` and ``mass_in``."""
        import h5py

        h5_file = tmp_path / "synthetic_mergers.hdf5"

        # Three synthetic mergers (mass_in in TNG internal units: 10^10 M_sun / h)
        time_arr = np.array([0.5, 0.7, 0.9], dtype=np.float64)
        mass_in_arr = np.array([0.5, 0.8, 1.0], dtype=np.float64)

        with h5py.File(h5_file, "w") as f:
            f.create_dataset("time", data=time_arr)
            f.create_dataset("mass_in", data=mass_in_arr)
            # Include the (unreliable) mass_out to mirror the real schema;
            # the loader must ignore it.
            f.create_dataset(
                "mass_out",
                data=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            )

        cat = load_tng_hdf5(h5_file, simulation="TestSim", hubble_h=_H)

        # Metadata
        assert cat.simulation == "TestSim"
        assert cat.hubble_h == pytest.approx(_H)
        assert cat.n_mergers == 3

        # Equal-mass proxy: m1 == m2 == mass_in (in M_sun)
        expected_mass = mass_in_arr * _MASS_UNIT
        np.testing.assert_allclose(cat.m1_msun, expected_mass, rtol=1e-12)
        np.testing.assert_allclose(cat.m2_msun, expected_mass, rtol=1e-12)
        np.testing.assert_array_equal(cat.m1_msun, cat.m2_msun)

        # Redshifts
        expected_z = 1.0 / time_arr - 1.0
        np.testing.assert_allclose(cat.redshift, expected_z, rtol=1e-12)

    def test_legacy_capitalized_schema(self, tmp_path) -> None:
        """Legacy capitalized schema (``Time`` / ``BHMass_In``) still works."""
        import h5py

        h5_file = tmp_path / "synthetic_legacy.hdf5"

        time_arr = np.array([0.5, 0.7], dtype=np.float64)
        bhmass_in_arr = np.array([0.5, 0.8], dtype=np.float64)

        with h5py.File(h5_file, "w") as f:
            f.create_dataset("Time", data=time_arr)
            f.create_dataset("BHMass_In", data=bhmass_in_arr)

        cat = load_tng_hdf5(h5_file, simulation="Legacy", hubble_h=_H)

        assert cat.n_mergers == 2
        expected_mass = bhmass_in_arr * _MASS_UNIT
        np.testing.assert_allclose(cat.m1_msun, expected_mass, rtol=1e-12)
        np.testing.assert_allclose(cat.m2_msun, expected_mass, rtol=1e-12)

    def test_mergers_group_layout(self, tmp_path) -> None:
        """Datasets under a 'Mergers' subgroup are also found and loaded."""
        import h5py

        h5_file = tmp_path / "synthetic_group.hdf5"

        time_arr = np.array([0.3, 0.6], dtype=np.float64)
        mass_in_arr = np.array([0.1, 0.4], dtype=np.float64)

        with h5py.File(h5_file, "w") as f:
            grp = f.create_group("Mergers")
            grp.create_dataset("time", data=time_arr)
            grp.create_dataset("mass_in", data=mass_in_arr)

        cat = load_tng_hdf5(h5_file, simulation="TestGroup", hubble_h=_H)

        assert cat.n_mergers == 2
        # Equal-mass proxy
        np.testing.assert_array_equal(cat.m1_msun, cat.m2_msun)

    def test_corrupt_rows_skipped(self, tmp_path) -> None:
        """Rows with non-positive mass_in or time are dropped with a warning."""
        import h5py

        h5_file = tmp_path / "synthetic_corrupt.hdf5"

        # Row 1: mass_in = 0.0 → corrupt, should be dropped.
        time_arr = np.array([0.5, 0.6, 0.8], dtype=np.float64)
        mass_in_arr = np.array([0.2, 0.0, 0.3], dtype=np.float64)

        with h5py.File(h5_file, "w") as f:
            f.create_dataset("time", data=time_arr)
            f.create_dataset("mass_in", data=mass_in_arr)

        with pytest.warns(UserWarning, match="corrupt"):
            cat = load_tng_hdf5(h5_file)

        assert cat.n_mergers == 2  # one row dropped

    def test_missing_datasets_raises(self, tmp_path) -> None:
        """A file missing the expected datasets raises ValueError."""
        import h5py

        h5_file = tmp_path / "bad.hdf5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("SomeOtherField", data=np.array([1.0]))

        with pytest.raises(ValueError, match="required datasets"):
            load_tng_hdf5(h5_file)

    def test_file_not_found_raises(self, tmp_path) -> None:
        """A non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tng_hdf5(tmp_path / "does_not_exist.hdf5")
