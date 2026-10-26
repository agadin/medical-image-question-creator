"""Image-quality index: noise, contrast phase estimate, body truncation, arms in field.

Outputs (Parquet, in ``INDEX_DIR``):

- ``quality.parquet``       one row per case: noise_hu, aorta_hu, portal_hu, phase.
- ``slice_quality.parquet`` one row per (case, axis, slice): body_truncated, arms_in_fov.

``phase`` is a heuristic from mean HU of the eroded aorta and portal vein:
non_contrast (aorta < 100), arterial (aorta >= 100 and portal well below aorta),
venous (portal close to aorta), or unknown. Good enough to filter on, not a diagnosis.

Run with: ``uv run python -m medimgq.quality [--workers N]``
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .config import INDEX_DIR, TS_ROOT

BODY_HU = -300  # voxels above this are body (or table)
NOISE_SOURCES = ("autochthon_left", "autochthon_right", "gluteus_maximus_left", "gluteus_maximus_right")


def _mask(case_dir: Path, name: str) -> np.ndarray:
    return np.asanyarray(nib.load(case_dir / "segmentations" / f"{name}.nii.gz").dataobj) > 0


def _eroded_values(ct: np.ndarray, m: np.ndarray, it: int = 2) -> np.ndarray:
    if m.sum() < 50:
        return np.array([])
    e = ndimage.binary_erosion(m, iterations=it)
    return ct[e] if e.sum() >= 30 else ct[m]


def classify_phase(aorta_hu: float, portal_hu: float) -> str:
    if np.isnan(aorta_hu):
        return "unknown"
    if aorta_hu < 100:
        return "non_contrast"
    if np.isnan(portal_hu):
        return "contrast_unknown_phase"
    return "arterial" if portal_hu < aorta_hu - 70 else "venous"


def quality_case(case_id: str, root: str) -> tuple[dict, pd.DataFrame]:
    d = Path(root) / case_id
    ct = np.asanyarray(nib.load(d / "ct.nii.gz").dataobj).astype(np.int16)

    # Noise: robust SD (MAD) of HU inside eroded paraspinal / gluteal muscle.
    vals = np.concatenate([_eroded_values(ct, _mask(d, n)) for n in NOISE_SOURCES])
    noise = float(1.4826 * np.median(np.abs(vals - np.median(vals)))) if vals.size >= 200 else np.nan
    a = _eroded_values(ct, _mask(d, "aorta"))
    p = _eroded_values(ct, _mask(d, "portal_vein_and_splenic_vein"), it=1)
    aorta_hu = float(np.median(a)) if a.size else np.nan
    portal_hu = float(np.median(p)) if p.size else np.nan

    # Body truncation per slice: largest body component touches the in-plane border.
    body = ct > BODY_HU
    lab, n = ndimage.label(body)
    if n:
        sizes = ndimage.sum_labels(body, lab, index=np.arange(1, n + 1))
        body = lab == (int(np.argmax(sizes)) + 1)
    arms = _mask(d, "humerus_left") | _mask(d, "humerus_right")
    rows = []
    for axis in range(3):
        others = [x for x in range(3) if x != axis]
        # Only side-to-side edges count; every coronal/sagittal slice touches the
        # top/bottom of the scan range, which is not field-of-view truncation.
        edge = np.zeros(ct.shape, bool)
        for o in (x for x in others if x != 2):
            sl = [slice(None)] * 3
            sl[o] = 0
            edge[tuple(sl)] = True
            sl[o] = -1
            edge[tuple(sl)] = True
        trunc = (body & edge).sum(axis=tuple(others)) > 20
        arm = arms.sum(axis=tuple(others)) > 0
        rows.append(pd.DataFrame(dict(case_id=case_id, axis=np.int8(axis),
                                      slice=np.arange(ct.shape[axis], dtype=np.int16),
                                      body_truncated=trunc, arms_in_fov=arm)))
    return (dict(case_id=case_id, noise_hu=noise, aorta_hu=aorta_hu, portal_hu=portal_hu,
                 phase=classify_phase(aorta_hu, portal_hu)),
            pd.concat(rows, ignore_index=True))


def build(workers: int = 8, limit: int | None = None, root: Path = TS_ROOT, out: Path = INDEX_DIR) -> None:
    case_ids = sorted(pd.read_parquet(out / "cases.parquet").case_id)[:limit]
    t0 = time.time()
    cases, slices = [], []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(quality_case, c, str(root)) for c in case_ids]
        for i, fut in enumerate(as_completed(futs), 1):
            c, s = fut.result()
            cases.append(c)
            slices.append(s)
            if i % 100 == 0 or i == len(case_ids):
                print(f"{i}/{len(case_ids)} cases  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(cases).sort_values("case_id").to_parquet(out / "quality.parquet", index=False)
    sq = pd.concat(slices, ignore_index=True)
    sq["case_id"] = sq.case_id.astype("category")
    sq.to_parquet(out / "slice_quality.parquet", index=False)
    print(f"wrote quality index in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    build(a.workers, a.limit)


if __name__ == "__main__":
    main()
