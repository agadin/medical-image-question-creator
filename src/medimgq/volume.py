"""Load CT volumes and masks, extract 2D slices in radiological orientation, apply windows."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

from .config import TS_ROOT

AXIS_IDS = {"sagittal": 0, "coronal": 1, "axial": 2}

# (width, level) in HU
WINDOWS: dict[str, tuple[float, float]] = {
    "soft_tissue": (400, 40),
    "abdomen": (350, 50),
    "liver": (150, 30),
    "mediastinum": (350, 50),
    "lung": (1500, -600),
    "bone": (1800, 400),
    "brain": (80, 40),
    "angio": (600, 150),
}


def case_dir(case_id: str, root: Path = TS_ROOT) -> Path:
    return root / case_id


@lru_cache(maxsize=8)
def load_ct(case_id: str) -> np.ndarray:
    return np.asanyarray(nib.load(case_dir(case_id) / "ct.nii.gz").dataobj).astype(np.int16)


@lru_cache(maxsize=256)
def load_mask(case_id: str, structure: str) -> np.ndarray:
    return np.asanyarray(nib.load(case_dir(case_id) / "segmentations" / f"{structure}.nii.gz").dataobj) > 0


def take_slice(vol: np.ndarray, axis: str | int, index: int) -> np.ndarray:
    """Return a 2D slice as displayed on a monitor (row 0 = top of image).

    Volumes are RAS. Radiological convention: patient's right on the image left.
    - axial:    rows = anterior->posterior, cols = patient right->left
    - coronal:  rows = superior->inferior,  cols = patient right->left
    - sagittal: rows = superior->inferior,  cols = anterior->posterior
    """
    ax = AXIS_IDS[axis] if isinstance(axis, str) else axis
    if ax == 2:
        s = vol[:, :, index]  # (x, y)
        return s[::-1, ::-1].T  # x: R->L means x decreasing; y: A->P means y decreasing
    if ax == 1:
        s = vol[:, index, :]  # (x, z)
        return s[::-1, ::-1].T
    s = vol[index, :, :]  # (y, z)
    return s[::-1, ::-1].T


def to_display_coords(axis: str | int, shape: tuple[int, int, int], u: float, v: float) -> tuple[float, float]:
    """Map index-space in-plane coordinates (cu, cv) from the indexer to (col, row) in a take_slice image."""
    ax = AXIS_IDS[axis] if isinstance(axis, str) else axis
    others = [a for a in range(3) if a != ax]
    nu, nv = shape[others[0]], shape[others[1]]
    return (nu - 1 - u, nv - 1 - v)


def apply_window(hu: np.ndarray, window: str | tuple[float, float]) -> np.ndarray:
    w, l = WINDOWS[window] if isinstance(window, str) else window
    lo = l - w / 2
    # Pixels outside the scanner's reconstruction field of view are stored far below
    # air (e.g. -3024 HU); treat them as air so wide windows show no FOV rectangle.
    hu = np.maximum(hu.astype(np.float32), -1024)
    return (np.clip((hu - lo) / w, 0, 1) * 255).astype(np.uint8)
