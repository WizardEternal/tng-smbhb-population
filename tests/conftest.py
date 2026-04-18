"""pytest configuration and shared fixtures for tng-smbhb-population tests.

All tests use synthetic data; no real HDF5 file is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from tng_smbhb.catalog import TNGMergerCatalog, catalog_from_arrays


@pytest.fixture(scope="session")
def synthetic_tng_catalog() -> TNGMergerCatalog:
    """Return a TNGMergerCatalog built from 500 deterministic synthetic mergers.

    Parameters
    ----------
    None — seeded with numpy default_rng(42) for reproducibility.

    Merger properties
    -----------------
    m1 : log-uniform in [1e5, 1e10] M_sun
    m2 : m1 * uniform(0.1, 1.0)  (so m2 <= m1 always)
    scale_factor : uniform in [0.3, 1.0]
    redshift : derived as z = 1/a - 1  (computed by catalog_from_arrays)
    simulation : "TNG100-1-synthetic"
    hubble_h : 0.6774
    """
    rng = np.random.default_rng(seed=42)
    n = 500

    log_m1 = rng.uniform(np.log10(1e5), np.log10(1e10), size=n)
    m1 = 10.0**log_m1

    q = rng.uniform(0.1, 1.0, size=n)
    m2 = m1 * q

    scale_factor = rng.uniform(0.3, 1.0, size=n)

    return catalog_from_arrays(
        m1_msun=m1,
        m2_msun=m2,
        scale_factor=scale_factor,
        simulation="TNG100-1-synthetic",
        hubble_h=0.6774,
    )
