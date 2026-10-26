"""Build the dataset index used for slice search.

Outputs (Parquet, in ``INDEX_DIR``):

- ``cases.parquet``      one row per case: metadata from meta.csv + volume shape.
- ``structures.parquet`` one row per (case, structure) present: volume, bbox,
                         3D connected components, whether it touches the volume edge.
- ``slices.parquet``     one row per (case, structure, axis, slice) with area > 0:
                         area in voxels and 2D centroid (in-plane coordinates).
- ``levels.parquet``     one row per (case, axial slice): the vertebra with the
                         largest area on that slice (the vertebral level).

Axis convention (volumes are RAS): axis 0 = sagittal (x), 1 = coronal (y),
2 = axial (z). For a slice on axis k, the centroid is given in the two
remaining axes in increasing order, i.e. axial -> (x, y).

Run with: ``uv run python -m medimgq.indexer [--workers N] [--limit N]``
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .catalog import STRUCTURES, VERTEBRAE
from .config import INDEX_DIR, TS_ROOT

AXES = ("sagittal", "coronal", "axial")


def read_meta(root: Path = TS_ROOT) -> pd.DataFrame:
    with open(root / "meta.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    df = pd.DataFrame(rows).rename(columns={"image_id": "case_id"})
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["kvp"] = pd.to_numeric(df["kvp"], errors="coerce")
    for c in ("study_type", "pathology", "pathology_location", "gender"):
        df[c] = df[c].str.strip().replace("", "unknown")
    return df


def index_case(case_id: str, root: str) -> dict:
    case_dir = Path(root) / case_id
    ct = nib.load(case_dir / "ct.nii.gz")
    shape = ct.shape
    structures, slices = [], []
    vert_area = {}  # vertebra -> per-z area
    for name in STRUCTURES:
        full = np.asanyarray(nib.load(case_dir / "segmentations" / f"{name}.nii.gz").dataobj)
        # Crop to the bounding box first; everything below runs on the crop.
        spans = []
        for axis in range(3):
            hit = np.nonzero(full.any(axis=tuple(a for a in range(3) if a != axis)))[0]
            if hit.size == 0:
                break
            spans.append((int(hit[0]), int(hit[-1])))
        if len(spans) < 3:
            continue
        lo = np.array([s[0] for s in spans])
        hi = np.array([s[1] for s in spans])
        m = full[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] > 0
        vol = int(m.sum())
        _, ncomp = ndimage.label(m)
        touches = bool((lo == 0).any() or (hi == np.array(shape) - 1).any())
        structures.append(
            dict(case_id=case_id, structure=name, volume_vox=vol, n_components=int(ncomp),
                 touches_edge=touches,
                 bbox_x0=int(lo[0]), bbox_y0=int(lo[1]), bbox_z0=int(lo[2]),
                 bbox_x1=int(hi[0]), bbox_y1=int(hi[1]), bbox_z1=int(hi[2]))
        )
        coords = [np.arange(lo[a], hi[a] + 1, dtype=np.float64) for a in range(3)]
        for axis in range(3):
            others = tuple(a for a in range(3) if a != axis)
            area = m.sum(axis=others)
            nz = np.nonzero(area)[0]
            # in-plane centroid: weighted mean of the two remaining coordinates
            cents = []
            for o in others:
                rest = tuple(a for a in range(3) if a not in (axis, o))
                proj = m.sum(axis=rest)  # 2D over (axis, o) in sorted order
                if axis > o:
                    proj = proj.T
                cents.append((proj * coords[o][None, :]).sum(1)[nz] / area[nz])
            slices.append(pd.DataFrame(dict(
                case_id=case_id, structure=name, axis=np.int8(axis), slice=(nz + lo[axis]).astype(np.int16),
                area=area[nz].astype(np.int32),
                cu=cents[0].astype(np.float32), cv=cents[1].astype(np.float32))))
            if axis == 2 and name.startswith("vertebrae_"):
                za = np.zeros(shape[2], dtype=np.int64)
                za[lo[2]:hi[2] + 1] = area
                vert_area[name.removeprefix("vertebrae_")] = za
    levels = None
    if vert_area:
        order = [v for v in VERTEBRAE if v in vert_area]
        stack = np.stack([vert_area[v] for v in order])
        best = stack.argmax(0)
        has = stack.max(0) > 0
        z = np.nonzero(has)[0]
        levels = pd.DataFrame(dict(case_id=case_id, slice=z.astype(np.int16),
                                   level=[order[i] for i in best[z]]))
    return dict(
        case=dict(case_id=case_id, shape_x=shape[0], shape_y=shape[1], shape_z=shape[2],
                  n_structures=len(structures)),
        structures=structures,
        slices=pd.concat(slices, ignore_index=True) if slices else None,
        levels=levels,
    )


def build(workers: int = 8, limit: int | None = None, root: Path = TS_ROOT, out: Path = INDEX_DIR) -> None:
    meta = read_meta(root)
    case_ids = sorted(meta["case_id"])
    if limit:
        case_ids = case_ids[:limit]
    out.mkdir(parents=True, exist_ok=True)
    cases, structures, slices, levels = [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(index_case, c, str(root)): c for c in case_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            cases.append(r["case"])
            structures += r["structures"]
            if r["slices"] is not None:
                slices.append(r["slices"])
            if r["levels"] is not None:
                levels.append(r["levels"])
            if i % 25 == 0 or i == len(case_ids):
                el = time.time() - t0
                print(f"{i}/{len(case_ids)} cases  {el:.0f}s elapsed  ~{el / i * (len(case_ids) - i):.0f}s left", flush=True)
    cases_df = meta.merge(pd.DataFrame(cases), on="case_id", how="inner").sort_values("case_id")
    cases_df.to_parquet(out / "cases.parquet", index=False)
    pd.DataFrame(structures).sort_values(["case_id", "structure"]).to_parquet(out / "structures.parquet", index=False)
    sl = pd.concat(slices, ignore_index=True)
    for c in ("case_id", "structure"):
        sl[c] = sl[c].astype("category")
    sl.to_parquet(out / "slices.parquet", index=False)
    pd.concat(levels, ignore_index=True).to_parquet(out / "levels.parquet", index=False)
    print(f"wrote index for {len(cases_df)} cases, {len(sl):,} slice rows to {out} in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="index only the first N cases (for testing)")
    p.add_argument("--out", type=Path, default=INDEX_DIR)
    a = p.parse_args()
    build(a.workers, a.limit, out=a.out)


if __name__ == "__main__":
    main()
