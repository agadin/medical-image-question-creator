"""Findings index: measurements that reveal real pathology in the segmentations.

Each finding is a measurement taken on every case plus a threshold that flags it. Thresholds are
screening values for finding teaching cases, not diagnostic criteria: always look at the image.

Outputs ``findings.parquet`` in ``INDEX_DIR``: one row per (case, finding) with the measured value
and the axial slice that best shows it.

Run with: ``uv run python -m medimgq.findings [--workers N]``
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import INDEX_DIR
from .volume import load_mask

VOXEL_MM = 1.5
VOXEL_ML = VOXEL_MM**3 / 1000


@dataclass(frozen=True)
class Finding:
    name: str
    title: str
    measure: str
    unit: str
    threshold: float
    normal: str
    structures: tuple[str, ...]  # what to label or click when showing it
    note: str = ""
    sex: str | None = None


FINDINGS: dict[str, Finding] = {f.name: f for f in [
    Finding("thoracic_aortic_aneurysm", "Thoracic aortic aneurysm", "widest thoracic aortic diameter", "mm", 50,
            "ascending < 40 mm, descending < 30 mm", ("aorta",),
            "Diameter is the widest inscribed circle on an axial slice, so oblique cuts through the arch "
            "are not overestimated. 40-49 mm is dilation."),
    Finding("abdominal_aortic_aneurysm", "Abdominal aortic aneurysm", "widest abdominal aortic diameter", "mm", 30,
            "< 30 mm", ("aorta",), "30-35 mm is borderline; look for mural thrombus in larger ones."),
    Finding("goiter", "Goiter (enlarged thyroid)", "thyroid volume", "mL", 35, "10-20 mL", ("thyroid_gland",),
            "Look for retrosternal extension and tracheal narrowing."),
    Finding("renal_cyst", "Renal cyst", "largest renal cyst diameter", "mm", 10, "none", ("kidney_cyst_left",
            "kidney_cyst_right"), "Diameter of a sphere with the cyst's volume. TotalSegmentator segments cysts "
            "separately from the kidneys."),
    Finding("cardiomegaly", "Cardiomegaly", "axial cardiothoracic ratio", "ratio", 0.6, "< 0.5",
            ("heart",), "Widest heart width over the chest width (outer lung edges) on the same axial slice. "
            "Similar to, but not the same as, the radiograph ratio; pericardial effusion also raises it."),
    Finding("splenomegaly", "Splenomegaly", "spleen volume", "mL", 450, "about 150-250 mL", ("spleen",)),
    Finding("hepatomegaly", "Hepatomegaly", "liver volume", "mL", 2400, "about 1,200-1,800 mL", ("liver",)),
    Finding("enlarged_prostate", "Enlarged prostate", "prostate volume", "mL", 40, "20-30 mL", ("prostate",),
            sex="m"),
]}

# Volume findings come straight from the structure index (whole structure inside the scan only).
_VOLUME = {"goiter": "thyroid_gland", "splenomegaly": "spleen", "hepatomegaly": "liver",
           "enlarged_prostate": "prostate"}


def _aorta(case_id: str) -> list[dict]:
    """Widest inscribed-circle diameter of the aorta per axial slice, split thoracic vs abdominal."""
    m = load_mask(case_id, "aorta")
    if not m.any():
        return []
    rows = []
    for z in np.nonzero(m.any((0, 1)))[0]:
        sl = m[:, :, z]
        lab, n = ndimage.label(sl)
        d = max(2 * ndimage.distance_transform_edt(lab == k).max() * VOXEL_MM for k in range(1, n + 1))
        rows.append((int(z), float(d)))
    return [dict(case_id=case_id, kind="aorta", slice=z, value=d) for z, d in rows]


def _cysts(case_id: str) -> list[dict]:
    out = []
    for side in ("left", "right"):
        m = load_mask(case_id, f"kidney_cyst_{side}")
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        sizes = ndimage.sum_labels(m, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        d = (6 * sizes[k - 1] * VOXEL_MM**3 / np.pi) ** (1 / 3)
        z = int(np.argmax((lab == k).sum((0, 1))))
        out.append(dict(case_id=case_id, kind="renal_cyst", slice=z, value=float(d)))
    return out


def _ctr(case_id: str) -> list[dict]:
    heart = load_mask(case_id, "heart")
    if not heart.any():
        return []
    # Only slices through the heart's mid-section (T5-T11), using its largest piece, so stray
    # mask voxels near the diaphragm or great vessels can't inflate the width.
    from . import selection as sel

    lv = sel.levels()
    lv = lv[lv.case_id.astype(str) == case_id]
    mid = set(lv[lv.level.isin([f"T{i}" for i in range(5, 12)])].slice.astype(int))
    lungs = np.zeros_like(heart)
    for lobe in ("lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"):
        lungs |= load_mask(case_id, lobe)
    best = None
    for z in np.nonzero(heart.any((0, 1)))[0]:
        if int(z) not in mid:
            continue
        lab, n = ndimage.label(heart[:, :, z])
        sizes = ndimage.sum_labels(lab > 0, lab, index=np.arange(1, n + 1))
        hx = np.nonzero((lab == int(np.argmax(sizes)) + 1).any(1))[0]
        lx = np.nonzero(lungs[:, :, z].any(1))[0]
        if len(lx) < 2 or len(hx) < 2:
            continue
        chest = lx.max() - lx.min() + 1
        # Skip slices through the lung tips, and small-field-of-view scans (e.g. cardiac CT)
        # where the lungs run off the image and the chest width would be underestimated.
        if chest * VOXEL_MM < 200 or lx.min() <= 2 or lx.max() >= heart.shape[0] - 3:
            continue
        r = (hx.max() - hx.min() + 1) / chest
        if best is None or r > best[1]:
            best = (int(z), float(r))
    return [dict(case_id=case_id, kind="cardiomegaly", slice=best[0], value=best[1])] if best else []


def measure_case(case_id: str) -> list[dict]:
    return _aorta(case_id) + _cysts(case_id) + _ctr(case_id)


def build(workers: int = 8, out=INDEX_DIR) -> None:
    from . import selection as sel

    t0 = time.time()
    cases, structs, levels = sel.cases(), sel.structures(), sel.levels()
    lv = levels.assign(case_id=levels.case_id.astype(str)).set_index(["case_id", "slice"]).level
    ids = sorted(cases.case_id)
    with ProcessPoolExecutor(workers) as ex:
        raw = pd.DataFrame([r for rows in ex.map(measure_case, ids, chunksize=8) for r in rows])
    raw["level"] = [lv.get((c, s)) for c, s in zip(raw.case_id, raw.slice)]
    rows = []

    # Aorta: thoracic = cervical/thoracic levels, abdominal = lumbar/sacral levels.
    a = raw[raw.kind == "aorta"].dropna(subset=["level"])
    for name, regions in (("thoracic_aortic_aneurysm", "CT"), ("abdominal_aortic_aneurysm", "LS")):
        part = a[a.level.str[0].isin(list(regions))]
        best = part.loc[part.groupby("case_id").value.idxmax()]
        rows += [dict(case_id=r.case_id, finding=name, value=round(r.value, 1), slice=int(r.slice),
                      level=r.level) for r in best.itertuples()]
    for kind in ("renal_cyst", "cardiomegaly"):
        part = raw[raw.kind == kind]
        best = part.loc[part.groupby("case_id").value.idxmax()]
        rows += [dict(case_id=r.case_id, finding=kind, value=round(r.value, 3 if kind == "cardiomegaly" else 1),
                      slice=int(r.slice), level=r.level) for r in best.itertuples()]

    # Volumes from the structure index; the best slice is the one with the largest axial area.
    sl = sel.slice_rows(list(_VOLUME.values()), 2)
    for name, struct in _VOLUME.items():
        s = structs[(structs.structure == struct) & ~structs.touches_edge]
        area = sl[sl.structure == struct]
        top = area.loc[area.groupby("case_id").area.idxmax()].set_index("case_id").slice
        for r in s.itertuples():
            z = int(top.get(r.case_id, -1))
            rows.append(dict(case_id=r.case_id, finding=name, value=round(r.volume_vox * VOXEL_ML, 1),
                             slice=z, level=lv.get((r.case_id, z))))

    df = pd.DataFrame(rows)
    df["flagged"] = [v >= FINDINGS[f].threshold for f, v in zip(df.finding, df.value)]
    male = set(cases.case_id[cases.gender == "m"])
    df.loc[(df.finding == "enlarged_prostate") & ~df.case_id.isin(male), "flagged"] = False
    df.to_parquet(out / "findings.parquet", index=False)
    print(df[df.flagged].finding.value_counts().to_string())
    print(f"wrote findings for {df.case_id.nunique()} cases in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    build(p.parse_args().workers)


if __name__ == "__main__":
    main()
