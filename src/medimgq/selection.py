"""Choose (case, view, slice) for a set of target structures.

Each strategy scores candidate slices; all of them share the same hard filters
(case filters, minimum visible area per target, edge truncation).

Strategies
----------
- ``classical``  restrict each target to its textbook view and vertebral-level
                 window (see ``CLASSICAL``), then pick the slice where every
                 target is most visible. Falls back to ``max_visibility``.
- ``max_visibility`` the slice maximizing the *least* visible target, where
                 visibility = area on slice / that structure's max area on this axis.
- ``max_area``   the slice maximizing summed absolute area of targets.
- ``random``     uniformly random among slices passing the filters.
- ``level``      a specific vertebral level (e.g. "L1"); best slice within it.
- ``index``      an explicit slice index (requires a case_id).

Scores are comparable within a strategy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from .catalog import VERTEBRAE
from .config import BLOCKLIST, INDEX_DIR

VOXEL_MM2 = 1.5 * 1.5  # TotalSegmentator v2 is resampled to 1.5 mm isotropic

# Textbook presentation per structure: view and the vertebral levels where it is
# classically shown. Levels are inclusive and refer to the axial slice's vertebra;
# for coronal/sagittal views the level window is not applied.
CLASSICAL: dict[str, tuple[str, tuple[str, str] | None]] = {
    "liver": ("axial", ("T10", "T12")),
    "spleen": ("axial", ("T11", "L1")),
    "stomach": ("axial", ("T10", "L1")),
    "gallbladder": ("axial", ("T12", "L2")),
    "pancreas": ("axial", ("T12", "L2")),
    "duodenum": ("axial", ("L1", "L3")),
    "kidney_left": ("axial", ("T12", "L2")),
    "kidney_right": ("axial", ("L1", "L2")),
    "adrenal_gland_left": ("axial", ("T11", "L1")),
    "adrenal_gland_right": ("axial", ("T11", "L1")),
    "portal_vein_and_splenic_vein": ("axial", ("T12", "L1")),
    "inferior_vena_cava": ("axial", None),
    "aorta": ("axial", None),
    "small_bowel": ("axial", ("L3", "L5")),
    "colon": ("axial", ("L2", "L5")),
    "iliopsoas_left": ("axial", ("L3", "L5")),
    "iliopsoas_right": ("axial", ("L3", "L5")),
    "urinary_bladder": ("axial", ("S1", "S1")),
    "prostate": ("axial", None),
    "esophagus": ("axial", None),
    "trachea": ("axial", ("C6", "T4")),
    "heart": ("axial", ("T6", "T9")),
    "superior_vena_cava": ("axial", ("T3", "T5")),
    "brachiocephalic_trunk": ("axial", ("T1", "T3")),
    "brachiocephalic_vein_left": ("axial", ("T1", "T3")),
    "brachiocephalic_vein_right": ("axial", ("T1", "T3")),
    "pulmonary_vein": ("axial", ("T6", "T8")),
    "atrial_appendage_left": ("axial", ("T5", "T8")),
    "thyroid_gland": ("axial", ("C6", "T1")),
    "common_carotid_artery_left": ("axial", ("C4", "C7")),
    "common_carotid_artery_right": ("axial", ("C4", "C7")),
    "lung_middle_lobe_right": ("axial", ("T6", "T9")),
    "lung_upper_lobe_left": ("coronal", None),
    "lung_lower_lobe_left": ("coronal", None),
    "lung_upper_lobe_right": ("coronal", None),
    "lung_lower_lobe_right": ("coronal", None),
    "spinal_cord": ("sagittal", None),
    "sacrum": ("sagittal", None),
    "sternum": ("sagittal", None),
    "brain": ("axial", None),
    "femur_left": ("coronal", None),
    "femur_right": ("coronal", None),
    "hip_left": ("coronal", None),
    "hip_right": ("coronal", None),
    **{f"vertebrae_{v}": ("sagittal", None) for v in VERTEBRAE},
}

# Tubular structures: visibility is judged against the structure's typical
# cross-section in the case (median), not its maximum (e.g. the aortic arch).
TUBULAR = {
    "aorta", "inferior_vena_cava", "superior_vena_cava", "esophagus", "trachea", "spinal_cord",
    "portal_vein_and_splenic_vein", "brachiocephalic_trunk", "brachiocephalic_vein_left",
    "brachiocephalic_vein_right", "common_carotid_artery_left", "common_carotid_artery_right",
    "subclavian_artery_left", "subclavian_artery_right", "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right", "pulmonary_vein",
}

_VORDER = {v: i for i, v in enumerate(VERTEBRAE)}
_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


@dataclass
class Filters:
    case_id: str | None = None
    pathology: list[str] | None = None  # e.g. ["no_pathology"], ["tumor"]
    pathology_location: list[str] | None = None  # substring match, e.g. ["abdomen"]
    study_type: list[str] | None = None  # substring match, e.g. ["abdomen"]
    gender: str | None = None
    age_min: float | None = None
    age_max: float | None = None
    exclude_cases: list[str] = field(default_factory=list)
    min_area_mm2: float | None = None  # per target on the slice; None = 30% of that structure's typical max, >= 20
    min_visibility: float = 0.4  # per target: area / its max area on this axis
    allow_edge_truncated: bool = False  # target structure cut off by the scan volume
    allow_body_truncated: bool = False  # patient's body cut off by the field of view on this slice
    allow_arms_in_fov: bool = False  # axial slices at T10 or below: arms down beside the trunk
    end_margin_mm: float = 12.0  # skip axial slices this close to either end of the scan
    use_blocklist: bool = True  # skip cases listed in qc/blocklist.csv
    max_noise_hu: float | None = 45.0  # robust SD of HU in muscle; None disables
    phase: list[str] | None = None  # non_contrast | arterial | venous | contrast_unknown_phase


@dataclass
class Candidate:
    case_id: str
    axis: str
    slice: int
    score: float
    level: str | None
    areas_mm2: dict[str, float]
    visibility: dict[str, float]  # area fraction of own max, scaled by size typicality
    strategy: str


@lru_cache(maxsize=1)
def cases() -> pd.DataFrame:
    return pd.read_parquet(INDEX_DIR / "cases.parquet")


@lru_cache(maxsize=1)
def structures() -> pd.DataFrame:
    return pd.read_parquet(INDEX_DIR / "structures.parquet")


@lru_cache(maxsize=1)
def levels() -> pd.DataFrame:
    return pd.read_parquet(INDEX_DIR / "levels.parquet")


@lru_cache(maxsize=1)
def blocklist() -> frozenset[str]:
    if not BLOCKLIST.exists():
        return frozenset()
    return frozenset(pd.read_csv(BLOCKLIST, comment="#").case_id.astype(str).str.strip())


@lru_cache(maxsize=1)
def quality() -> pd.DataFrame:
    return pd.read_parquet(INDEX_DIR / "quality.parquet")


@lru_cache(maxsize=1)
def slice_quality() -> pd.DataFrame:
    sq = pd.read_parquet(INDEX_DIR / "slice_quality.parquet")
    sq["case_id"] = sq.case_id.astype(str)
    return sq.set_index(["case_id", "axis", "slice"]).sort_index()


@lru_cache(maxsize=1)
def _slices_ds() -> pads.Dataset:
    return pads.dataset(INDEX_DIR / "slices.parquet")


def slice_rows(names: list[str], axis: int, case_ids: list[str] | None = None) -> pd.DataFrame:
    expr = pads.field("structure").isin(names) & (pads.field("axis") == axis)
    if case_ids is not None:
        expr &= pads.field("case_id").isin(case_ids)
    df = _slices_ds().to_table(filter=expr).to_pandas()
    for c in ("case_id", "structure"):
        df[c] = df[c].astype(str)
    return df


def eligible_cases(f: Filters, required: list[str]) -> list[str]:
    c = cases()
    m = pd.Series(True, index=c.index)
    if f.case_id:
        m &= c.case_id == f.case_id
    if f.pathology:
        m &= c.pathology.isin(f.pathology)
    if f.pathology_location:
        m &= c.pathology_location.apply(lambda s: any(p in s for p in f.pathology_location))
    if f.study_type:
        m &= c.study_type.apply(lambda s: any(p in s for p in f.study_type))
    if f.gender:
        m &= c.gender == f.gender
    if f.age_min is not None:
        m &= c.age >= f.age_min
    if f.age_max is not None:
        m &= c.age <= f.age_max
    if f.exclude_cases:
        m &= ~c.case_id.isin(f.exclude_cases)
    if f.use_blocklist:
        m &= ~c.case_id.isin(blocklist())
    ids = set(c.case_id[m])
    if f.max_noise_hu is not None or f.phase:
        q = quality()
        qm = pd.Series(True, index=q.index)
        if f.max_noise_hu is not None:
            qm &= q.noise_hu.fillna(np.inf) <= f.max_noise_hu
        if f.phase:
            qm &= q.phase.isin(f.phase)
        ids &= set(q.case_id[qm])
    s = structures()
    s = s[s.case_id.isin(ids) & s.structure.isin(required)]
    if not f.allow_edge_truncated:
        s = s[~s.touches_edge]
    have = s.groupby("case_id").structure.nunique()
    return sorted(have[have == len(set(required))].index)


def classical_view(names: list[str]) -> str:
    """Most common classical view among the targets (axial wins ties)."""
    views = [CLASSICAL.get(n, ("axial", None))[0] for n in names]
    return max(("axial", "coronal", "sagittal"), key=views.count)


def _level_window(names: list[str]) -> tuple[int, int] | None:
    """Intersection of the targets' classical level windows, or their union if disjoint."""
    wins = [CLASSICAL[n][1] for n in names if n in CLASSICAL and CLASSICAL[n][1]]
    if not wins:
        return None
    lo = [_VORDER[a] for a, _ in wins]
    hi = [_VORDER[b] for _, b in wins]
    if max(lo) <= min(hi):
        return max(lo), min(hi)
    return min(lo), max(hi)


def candidates(
    targets: list[str],
    *,
    view: str = "auto",
    strategy: str = "classical",
    filters: Filters | None = None,
    level: str | None = None,
    slice_index: int | None = None,
    rng: np.random.Generator | None = None,
    limit: int = 50,
) -> list[Candidate]:
    """Rank candidate slices showing every structure in ``targets`` (canonical names)."""
    f = filters or Filters()
    rng = rng or np.random.default_rng()
    targets = list(dict.fromkeys(targets))
    view = classical_view(targets) if view == "auto" else view
    axis = _AXIS[view]
    ids = eligible_cases(f, targets)
    if not ids:
        return []

    df = slice_rows(targets, axis, ids)
    df["area_mm2"] = df.area * VOXEL_MM2
    g = df.groupby(["case_id", "structure"]).area
    case_max = g.transform("max")
    ref = case_max.where(~df.structure.isin(TUBULAR), g.transform("median"))
    df["vis"] = (df.area / ref).clip(upper=1.0)
    # Typicality: penalize cases where the structure is unusually small (e.g. a
    # contracted gallbladder) relative to the median case's largest cross-section.
    med = g.max().groupby("structure").median()
    typ = (case_max / df.structure.map(med)).clip(upper=1.0)
    df["vis"] = df.vis * typ
    if f.min_area_mm2 is None:
        df["min_area"] = (0.3 * df.structure.map(med) * VOXEL_MM2).clip(lower=20.0)
    else:
        df["min_area"] = f.min_area_mm2
    df["area_ok"] = df.area_mm2 >= df.min_area
    area = df.pivot_table(index=["case_id", "slice"], columns="structure", values="area_mm2", fill_value=0.0)
    vis = df.pivot_table(index=["case_id", "slice"], columns="structure", values="vis", fill_value=0.0)
    area, vis = area.reindex(columns=targets, fill_value=0.0), vis.reindex(columns=targets, fill_value=0.0)
    area_ok = df.pivot_table(index=["case_id", "slice"], columns="structure", values="area_ok",
                             fill_value=False, aggfunc="max").reindex(index=area.index, columns=targets, fill_value=False)
    ok = area_ok.all(axis=1) & (vis.min(axis=1) >= f.min_visibility)
    sq = slice_quality()
    key = pd.MultiIndex.from_arrays([area.index.get_level_values(0), np.full(len(area), axis, np.int8),
                                     area.index.get_level_values(1)])
    if not f.allow_body_truncated:
        ok &= ~sq.body_truncated.reindex(key, fill_value=True).to_numpy()
    lv = None
    if axis == 2:
        lv = levels().set_index(["case_id", "slice"]).level
        lv = lv.reindex(area.index)
        lv_ord = lv.map(_VORDER)
        if not f.allow_arms_in_fov:
            # shoulders always contain the humeral heads; only flag arms beside the abdomen/pelvis
            low = (lv_ord >= _VORDER["T10"]).to_numpy()
            ok &= ~(sq.arms_in_fov.reindex(key, fill_value=False).to_numpy() & low)
        if f.end_margin_mm:
            nz = cases().set_index("case_id").shape_z
            sl = area.index.get_level_values("slice").to_numpy()
            n = nz.reindex(area.index.get_level_values("case_id")).to_numpy()
            mg = int(round(f.end_margin_mm / 1.5))
            ok &= (sl >= mg) & (sl < n - mg)

    used = strategy
    if strategy == "index":
        if slice_index is None or not f.case_id:
            raise ValueError("strategy 'index' needs slice_index and filters.case_id")
        ok &= area.index.get_level_values("slice") == slice_index
    elif strategy == "level" or (strategy == "classical" and axis == 2):
        if strategy == "level":
            if axis != 2 or not level:
                raise ValueError("strategy 'level' needs an axial view and a vertebral level")
            win = (_VORDER[level], _VORDER[level])
        else:
            win = _level_window(targets)
        if win is not None:
            in_win = ok & lv_ord.between(*win)
            if in_win.any():
                ok = in_win
            elif strategy == "classical":
                used = "max_visibility (classical window empty)"
            else:
                ok = in_win

    if not ok.any():
        return []
    a, v = area[ok], vis[ok]
    if strategy == "max_area":
        score = a.sum(axis=1)
    elif strategy == "random":
        score = pd.Series(rng.random(len(a)), index=a.index)
    else:
        # least-visible target dominates; small tie-breaker on mean visibility
        score = v.min(axis=1) + 0.1 * v.mean(axis=1)
    # random tie-breaking so equally good cases aren't always returned in case-id order
    score = score + rng.random(len(score)) * 1e-6

    # keep the best slice per case, then rank cases
    best = score.groupby(level="case_id").idxmax()
    ranked = score.loc[best.values].sort_values(ascending=False)
    if strategy == "random":
        ranked = ranked.sample(frac=1, random_state=int(rng.integers(2**31)))
    out = []
    for (cid, sl), sc in ranked.head(limit).items():
        out.append(Candidate(
            case_id=cid, axis=view, slice=int(sl), score=float(sc),
            level=(lv.loc[(cid, sl)] if lv is not None and pd.notna(lv.loc[(cid, sl)]) else None),
            areas_mm2={t: round(float(a.loc[(cid, sl), t]), 1) for t in targets},
            visibility={t: round(float(v.loc[(cid, sl), t]), 3) for t in targets},
            strategy=used,
        ))
    return out
