"""TNG ``blackhole_details.hdf5`` supplementary catalog: pre-merger mass lookup.

Purpose
-------
The TNG public ``blackhole_mergers.hdf5`` file has a documented data bug:
``mass_out`` is the dynamical cell mass, not the BH mass.  This module reads
the *other* supplementary file, ``blackhole_details.hdf5`` (~5.2 GB for
TNG100-1), whose per-timestep BH mass records are unmodified from the raw
simulation and are therefore the physical BH mass.

File schema (TNG100-1, ``blackhole_details.hdf5``)
--------------------------------------------------
Root attributes::

    @num_blackholes  = 64944        (unique BH particles)
    @num_entries     = 195529001    (total timestep records)
    @target_times    = (M,) float64 scale factors of detail snapshots

Root datasets (all shape ``(num_entries,)``)::

    /id      uint64    BH particle ID
    /time    float64   scale factor a of the record
    /mass    float64   BH mass [TNG internal units = 10^10 M_sun / h]
    /mdot    float64   accretion rate
    /rho     float64   local gas density
    /cs      float64   sound speed

``/unique`` subgroup (shape ``(num_blackholes,)``) — O(1) lookup index::

    /unique/id           uint64    unique BH particle ID (sorted ascending)
    /unique/first_index  int64     offset into /id, /time, /mass, ...
    /unique/num_entries  int64     length of this BH's run of records

To look up a given BH's full time series: the rows of the root datasets in
``[first_index : first_index + num_entries]`` are that BH's contiguous
history (sorted by time ascending, per TNG convention).

Public API
----------
:class:`DetailsIndex`
    In-memory index (loads only the ``/unique`` subgroup, ~1.5 MB).
:func:`load_details_index`
    Builds a :class:`DetailsIndex` from an HDF5 path.
:func:`lookup_premerger_mass`
    Returns the primary's mass from the last record *before* a merger time.
:func:`batch_lookup_premerger_masses`
    Vectorized lookup for a whole merger catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import h5py as _h5py  # noqa: F401


_TNG_MASS_UNIT_OVER_H: float = 1.0e10  # M_sun / h


@dataclass(frozen=True)
class DetailsIndex:
    """In-memory index of a ``blackhole_details.hdf5`` file's ``/unique`` group.

    All three arrays have the same length (``num_blackholes``) and are
    aligned: ``id[i]`` is the particle ID of the *i*-th unique BH, whose
    per-timestep records occupy rows
    ``first_index[i] : first_index[i] + num_entries[i]`` of the root-level
    ``/id``, ``/time``, ``/mass`` datasets.

    ``id`` is guaranteed sorted-ascending (verified in
    :func:`load_details_index`), which lets callers use
    :func:`numpy.searchsorted` for O(log N) ID→index resolution.

    Attributes
    ----------
    id : npt.NDArray[np.uint64]
        Unique BH particle IDs, sorted ascending.
    first_index : npt.NDArray[np.int64]
        Row offset of each BH's first record in the root datasets.
    num_entries : npt.NDArray[np.int64]
        Number of timestep records per BH.
    path : Path
        Filesystem path the index was built from.
    num_blackholes : int
        ``len(id)``, convenience scalar.
    """

    id: npt.NDArray[np.uint64]
    first_index: npt.NDArray[np.int64]
    num_entries: npt.NDArray[np.int64]
    path: Path
    num_blackholes: int


def load_details_index(path: str | Path) -> DetailsIndex:
    """Load the ``/unique`` subgroup of a ``blackhole_details.hdf5`` file.

    Reads ~1.5 MB (three ``(num_blackholes,)`` arrays) into memory.  Does not
    touch the root-level bulk datasets.

    Parameters
    ----------
    path : str or Path
        Path to the ``blackhole_details.hdf5`` file.

    Returns
    -------
    DetailsIndex

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If ``/unique`` is missing, its arrays are inconsistent, or
        ``/unique/id`` is not sorted ascending.
    """
    import h5py  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"blackhole_details file not found: '{path}'.")

    with h5py.File(path, "r") as f:
        if "unique" not in f:
            raise ValueError(
                f"'{path}' has no '/unique' subgroup; not a TNG "
                f"blackhole_details file."
            )
        grp = f["unique"]
        ids = np.asarray(grp["id"], dtype=np.uint64)
        first = np.asarray(grp["first_index"], dtype=np.int64)
        nent = np.asarray(grp["num_entries"], dtype=np.int64)

    if not (ids.shape == first.shape == nent.shape):
        raise ValueError(
            f"/unique arrays have inconsistent shapes: "
            f"id={ids.shape}, first_index={first.shape}, num_entries={nent.shape}."
        )
    if ids.size == 0:
        raise ValueError("/unique/id is empty.")
    if np.any(np.diff(ids.astype(np.int64)) < 0):
        raise ValueError("/unique/id is not sorted ascending.")
    if np.any(nent <= 0):
        raise ValueError("/unique/num_entries contains non-positive values.")
    if np.any(first < 0):
        raise ValueError("/unique/first_index contains negative values.")

    return DetailsIndex(
        id=ids,
        first_index=first,
        num_entries=nent,
        path=path,
        num_blackholes=int(ids.size),
    )


def _resolve_index(index: DetailsIndex, bh_id: int) -> int | None:
    """Return the row index in *index* for particle ID *bh_id*, or None."""
    pos = int(np.searchsorted(index.id, np.uint64(bh_id)))
    if pos >= index.num_blackholes or int(index.id[pos]) != int(bh_id):
        return None
    return pos


def lookup_premerger_mass(
    index: DetailsIndex,
    bh_id: int,
    merger_time: float,
    *,
    hubble_h: float = 0.6774,
) -> float | None:
    """Return the primary BH mass just before *merger_time*, in M_sun.

    Picks the last record of particle *bh_id* whose scale factor is strictly
    less than *merger_time*; this is the pre-merger primary mass (the
    record at ``t == merger_time`` reflects the post-merger state with the
    secondary already absorbed).  If no record exists strictly before
    *merger_time* (e.g. the BH was just seeded), falls back to the earliest
    available record.

    Parameters
    ----------
    index : DetailsIndex
    bh_id : int
        Particle ID of the surviving ("out") BH.
    merger_time : float
        Scale factor *a* of the merger event.
    hubble_h : float, optional
        Dimensionless Hubble parameter for the TNG internal-unit conversion.
        Default 0.6774 (Planck15 / TNG).

    Returns
    -------
    float or None
        Pre-merger BH mass in M_sun, or ``None`` if *bh_id* is not present
        in the index.

    Notes
    -----
    Opens the HDF5 file and reads two contiguous slices (``time`` and
    ``mass``, each ``num_entries[i]`` entries).  For batch workloads prefer
    :func:`batch_lookup_premerger_masses`, which keeps the file open.
    """
    import h5py  # noqa: PLC0415

    pos = _resolve_index(index, bh_id)
    if pos is None:
        return None

    start = int(index.first_index[pos])
    n = int(index.num_entries[pos])
    stop = start + n

    with h5py.File(index.path, "r") as f:
        times = np.asarray(f["time"][start:stop], dtype=np.float64)
        masses_internal = np.asarray(f["mass"][start:stop], dtype=np.float64)

    j = _pick_premerger_row(times, merger_time)
    mass_tng = float(masses_internal[j])
    return mass_tng * (_TNG_MASS_UNIT_OVER_H / hubble_h)


def _pick_premerger_row(times: npt.NDArray[np.float64], merger_time: float) -> int:
    """Return the row index of the last record with ``time < merger_time``.

    If every record has ``time >= merger_time`` (shouldn't happen for a
    genuine pre-merger BH, but guard against it), fall back to row 0.
    """
    mask = times < merger_time
    if not np.any(mask):
        return 0
    # searchsorted on ascending times: rightmost index where time < merger_time
    idx = int(np.searchsorted(times, merger_time, side="left")) - 1
    return max(idx, 0)


def batch_lookup_premerger_masses(
    index: DetailsIndex,
    bh_ids: npt.ArrayLike,
    merger_times: npt.ArrayLike,
    *,
    hubble_h: float = 0.6774,
    progress_every: int = 1000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Vectorized pre-merger mass lookup for a whole merger catalog.

    Sorts queries by ``first_index`` so HDF5 reads proceed monotonically
    forward through the file (roughly sequential on disk), then opens the
    file once for the whole batch.

    Parameters
    ----------
    index : DetailsIndex
    bh_ids : array_like of uint64
        Particle IDs to look up.
    merger_times : array_like of float64
        Scale factor *a* of each merger event, same length as *bh_ids*.
    hubble_h : float, optional
        TNG Hubble parameter.  Default 0.6774.
    progress_every : int, optional
        Print a progress message every this many queries.  Set to 0 to
        disable.  Default 1000.

    Returns
    -------
    masses_msun : npt.NDArray[np.float64]
        Pre-merger BH masses in M_sun, shape ``(N,)``.  Entries where the
        BH ID was not found are ``np.nan``.
    found : npt.NDArray[np.bool_]
        Boolean mask, ``True`` where the BH ID was resolved, shape ``(N,)``.

    Notes
    -----
    For 18,374 TNG100-1 mergers on a local SSD this runs in ~1–5 minutes
    depending on disk speed.  The dominant cost is HDF5 I/O, not the
    searchsorted lookups.
    """
    import h5py  # noqa: PLC0415

    ids = np.asarray(bh_ids, dtype=np.uint64).ravel()
    times = np.asarray(merger_times, dtype=np.float64).ravel()
    if ids.shape != times.shape:
        raise ValueError(
            f"bh_ids and merger_times must have the same length; "
            f"got {ids.shape} vs {times.shape}."
        )
    n = int(ids.size)

    # Resolve ID -> index-row (vectorized binary search).
    positions = np.searchsorted(index.id, ids)
    in_bounds = positions < index.num_blackholes
    found = np.zeros(n, dtype=bool)
    found[in_bounds] = index.id[positions[in_bounds]] == ids[in_bounds]

    masses_out = np.full(n, np.nan, dtype=np.float64)
    if not np.any(found):
        return masses_out, found

    # Order queries by start offset -> mostly-sequential file reads.
    found_ix = np.flatnonzero(found)
    first_starts = index.first_index[positions[found_ix]]
    order = np.argsort(first_starts, kind="stable")
    ordered_qix = found_ix[order]

    scale = _TNG_MASS_UNIT_OVER_H / hubble_h

    with h5py.File(index.path, "r") as f:
        time_ds = f["time"]
        mass_ds = f["mass"]
        for count, q in enumerate(ordered_qix, start=1):
            pos_i = positions[q]
            start = int(index.first_index[pos_i])
            length = int(index.num_entries[pos_i])
            stop = start + length
            t_run = np.asarray(time_ds[start:stop], dtype=np.float64)
            m_run = np.asarray(mass_ds[start:stop], dtype=np.float64)
            j = _pick_premerger_row(t_run, float(times[q]))
            masses_out[q] = float(m_run[j]) * scale
            if progress_every and count % progress_every == 0:
                print(
                    f"  [bh_details] resolved {count}/{int(found.sum())} "
                    f"pre-merger masses",
                    flush=True,
                )

    return masses_out, found
