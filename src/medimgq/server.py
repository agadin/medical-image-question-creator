"""MCP server: labeled CT images for practice-exam questions.

The calling AI writes the questions; these tools give it the pieces:

- ``list_structures`` / ``dataset_info``  what can be labeled and filtered on
- ``find_slices``                         rank (case, view, slice) showing given structures
- ``slice_contents``                      what else is visible on a slice (for distractors)
- ``render_image``                        render a specific slice with labels (+ hotspot masks)
- ``labeled_image``                       find + render in one call
- ``batch_labeled_images``                many ``labeled_image`` requests -> folder + manifest.jsonl

Run: ``uv run medimgq-mcp`` (stdio).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from mcp.server import MCPServer

from . import selection as sel
from .catalog import GROUPS, STRUCTURES, UnknownStructureError, resolve
from .config import OUTPUT_DIR, TS_ATTRIBUTION
from .render import PRESETS, LabelSpec, RenderOptions, render
from .volume import WINDOWS

mcp = MCPServer(
    name="medimgq",
    instructions=(
        "Generate labeled CT images (TotalSegmentator, 117 structures, 1.5 mm) for medical exam questions. "
        "Typical flow: labeled_image(labels=[{label:'A', structures:['pancreas']}, ...]) -> image + answer key. "
        "For control: find_slices -> slice_contents (pick distractors) -> render_image. "
        "Structure names accept synonyms ('IVC'), unsided names ('kidney' = both), and groups ('lumbar spine'). "
        "Labels with several structures share one letter. A label of '' draws an arrow with no text. "
        "Images are in radiological convention (patient's right on image left). "
        "Case pathology metadata describes the whole scan, not a labeled structure."
    ),
)

ViewT = Literal["auto", "axial", "coronal", "sagittal"]
StrategyT = Literal["classical", "max_visibility", "max_area", "random", "level", "index"]


class Label(BaseModel):
    label: str = Field(description='Text drawn on the image, e.g. "A" or "1". "" draws an arrow with no text.')
    structures: list[str] = Field(description="One or more structure names; all share this label.")


class HotspotZone(BaseModel):
    id: str = Field(description="Zone id, e.g. 'correct' or 'right_bcv'.")
    structures: list[str]


class CaseFilters(BaseModel):
    case_id: str | None = None
    pathology: list[str] | None = Field(None, description="e.g. ['no_pathology'] or ['tumor']; see dataset_info")
    pathology_location: list[str] | None = Field(None, description="substring match, e.g. ['abdomen']")
    study_type: list[str] | None = Field(None, description="substring match, e.g. ['abdomen']")
    gender: Literal["m", "f"] | None = None
    age_min: float | None = None
    age_max: float | None = None
    phase: list[str] | None = Field(None, description="contrast phase filter (off by default): non_contrast, arterial, venous")
    exclude_cases: list[str] = Field(default_factory=list)
    min_visibility: float = 0.4
    max_noise_hu: float | None = 45.0
    allow_edge_truncated: bool = False
    allow_body_truncated: bool = False
    allow_arms_in_fov: bool = False


class RenderSettings(BaseModel):
    preset: str = Field("arrow_short", description=f"one of {list(PRESETS)}")
    marker: str | None = Field(None, description="override: arrow_short | arrow_margin | on_structure | none")
    tip: Literal["center", "edge"] | None = None
    outline: bool | None = None
    shared_mode: Literal["multi_arrow", "repeat"] | None = None
    text_style: Literal["halo", "boxed", "boxed_dark"] | None = None
    marker_color: tuple[int, int, int] | None = None
    text_color: tuple[int, int, int] | None = None
    font_size: int | None = None
    size: int | None = Field(None, description="long side of the image in px (default 768)")
    crop: Literal["auto", "neck", "chest", "abdomen", "body", "targets", "full"] | None = None
    window: str | list[float] | None = Field(None, description=f"auto | {list(WINDOWS)} | [width, level]")
    format: Literal["png", "webp", "jpeg"] = "png"

    def options(self) -> RenderOptions:
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose from {list(PRESETS)}")
        kw: dict[str, Any] = dict(PRESETS[self.preset])
        for f in ("marker", "tip", "outline", "shared_mode", "text_style", "marker_color", "text_color",
                  "font_size", "size", "crop"):
            v = getattr(self, f)
            if v is not None:
                kw[f] = v
        if self.window is not None:
            kw["window"] = tuple(self.window) if isinstance(self.window, list) else self.window
        return RenderOptions(**kw)


# ---------------------------------------------------------------- helpers


def _resolve_all(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        try:
            out += [x for x in resolve(n) if x not in out]
        except UnknownStructureError as e:
            raise ValueError(str(e)) from None
    return out


def _filters(f: CaseFilters | None) -> sel.Filters:
    return sel.Filters(**(f.model_dump() if f else {}))


def _case_meta(case_id: str) -> dict:
    c = sel.cases().set_index("case_id").loc[case_id]
    q = sel.quality().set_index("case_id")
    meta = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in dict(
        age=float(c.age) if not np.isnan(c.age) else None, sex=c.gender, study_type=c.study_type,
        pathology=c.pathology, pathology_location=c.pathology_location).items()}
    if case_id in q.index:
        meta["contrast_phase_estimate"] = q.loc[case_id, "phase"]
    return meta


def _out_path(output_dir: str | None, name: str | None, fmt: str, seed_text: str) -> Path:
    d = Path(output_dir) if output_dir else OUTPUT_DIR / "mcp"
    d.mkdir(parents=True, exist_ok=True)
    stem = name or f"img_{hashlib.sha1(seed_text.encode()).hexdigest()[:10]}"
    return d / f"{stem}.{fmt}"


def _render_and_save(case_id: str, view: str, slice_index: int, labels: list[Label],
                     settings: RenderSettings, hotspot_zones: list[HotspotZone] | None,
                     output_dir: str | None, name: str | None, include_base64: bool,
                     level: str | None = None, extra: dict | None = None,
                     context_structures: list[str] | None = None) -> dict:
    specs = [LabelSpec(l.label, _resolve_all(l.structures)) for l in labels]
    o = settings.options()
    # Hotspot zones and context structures also drive cropping and the auto window
    # (a bare hotspot image has no labels).
    context = _resolve_all([s for z in hotspot_zones or [] for s in z.structures] + (context_structures or []))
    r = render(case_id, view, slice_index, specs, o, level=level, context=context)
    seed_text = json.dumps([case_id, view, slice_index, [l.model_dump() for l in labels],
                            settings.model_dump(), time.time()])
    path = _out_path(output_dir, name, settings.format, seed_text)
    img = r.image.convert("RGB") if settings.format == "jpeg" else r.image
    img.save(path, quality=92) if settings.format in ("jpeg", "webp") else img.save(path)

    masks = []
    for z in hotspot_zones or []:
        structs = _resolve_all(z.structures)
        mp = path.with_name(f"{path.stem}_mask_{z.id}.png")
        r.mask_image(structs).save(mp)
        masks.append(dict(id=z.id, structures=structs, path=str(mp)))

    key = r.key_dicts()
    result = dict(
        image_path=str(path),
        image_size=list(r.image.size),
        answer_key={k["label"] or f"(arrow {i + 1})": ", ".join(k["display_names"]) for i, k in enumerate(key)},
        key=key,
        hotspot_masks=masks,
        source=dict(dataset="TotalSegmentator v2.0.1", case_id=case_id, view=view, slice=slice_index,
                    vertebral_level=level, window=r.window if isinstance(r.window, str) else list(r.window),
                    crop_region=r.region, crop_box_slice_px=list(r.crop_box), case=_case_meta(case_id),
                    attribution=TS_ATTRIBUTION),
    )
    if extra:
        result.update(extra)
    if include_base64:
        buf = io.BytesIO()
        img.save(buf, format=settings.format.upper())
        result["image_base64"] = base64.b64encode(buf.getvalue()).decode()
        result["mime_type"] = f"image/{settings.format}"
    return result


# ---------------------------------------------------------------- tools


@mcp.tool()
def list_structures(query: str | None = None, region: str | None = None, system: str | None = None) -> dict:
    """List labelable structures. ``query`` resolves a name/synonym or substring-searches; filter by
    region (abdomen, thorax, neck, pelvis, head, spine, upper_limb, lower_limb) or system."""
    items = list(STRUCTURES.values())
    resolved = None
    if query:
        try:
            resolved = list(resolve(query))
            items = [STRUCTURES[n] for n in resolved]
        except UnknownStructureError as e:
            q = query.lower()
            items = [s for s in items if q in s.display.lower() or any(q in x.lower() for x in s.synonyms)]
            if not items:
                return dict(error=str(e), suggestions=e.suggestions)
    if region:
        items = [s for s in items if s.region == region]
    if system:
        items = [s for s in items if s.system == system]
    return dict(
        resolved=resolved,
        structures=[dict(name=s.name, display=s.display, region=s.region, system=s.system, side=s.side,
                         synonyms=list(s.synonyms)) for s in items],
        groups={g: list(m) for g, m in GROUPS.items()} if not query else None,
    )


@mcp.tool()
def dataset_info() -> dict:
    """Filter values (pathology, study type, contrast phase), windows, presets, and option names."""
    c, q = sel.cases(), sel.quality()
    return dict(
        cases=len(c),
        pathology=c.pathology.value_counts().to_dict(),
        pathology_location=c.pathology_location.value_counts().head(15).to_dict(),
        study_type=c.study_type.value_counts().head(20).to_dict(),
        contrast_phase_estimate=q.phase.value_counts().to_dict(),
        windows={k: dict(width=w, level=l) for k, (w, l) in WINDOWS.items()},
        presets=PRESETS,
        views=["auto", "axial", "coronal", "sagittal"],
        strategies=["classical", "max_visibility", "max_area", "random", "level", "index"],
        crop=["auto", "neck", "chest", "abdomen", "body", "targets", "full"],
        notes=["Pathology labels are coarse and describe the scan, not a structure.",
               "Contrast phase is a heuristic estimate from aortic/portal HU."],
    )


@mcp.tool()
def find_slices(structures: list[str], view: ViewT = "auto", strategy: StrategyT = "classical",
                level: str | None = None, slice_index: int | None = None,
                filters: CaseFilters | None = None, limit: int = 10, seed: int | None = None) -> dict:
    """Rank (case, view, slice) candidates where every structure is clearly visible.
    ``classical`` = textbook view and vertebral level; ``level`` = a vertebral level like 'L1'."""
    names = _resolve_all(structures)
    cands = sel.candidates(names, view=view, strategy=strategy, filters=_filters(filters), level=level,
                           slice_index=slice_index, rng=np.random.default_rng(seed), limit=limit)
    return dict(structures=names, candidates=[c.__dict__ for c in cands])


@mcp.tool()
def slice_contents(case_id: str, view: Literal["axial", "coronal", "sagittal"], slice_index: int,
                   min_area_mm2: float = 50.0) -> dict:
    """Structures visible on a slice (area, visibility), largest first. Use to choose distractors."""
    ax = {"sagittal": 0, "coronal": 1, "axial": 2}[view]
    ds = sel._slices_ds()
    import pyarrow.dataset as pads

    t = ds.to_table(filter=(pads.field("case_id") == case_id) & (pads.field("axis") == ax)).to_pandas()
    t["structure"] = t.structure.astype(str)
    mx = t.groupby("structure").area.max()
    row = t[t.slice == slice_index].copy()
    row["area_mm2"] = row.area * sel.VOXEL_MM2
    row = row[row.area_mm2 >= min_area_mm2].sort_values("area_mm2", ascending=False)
    lv = None
    if ax == 2:
        l = sel.levels()
        hit = l[(l.case_id.astype(str) == case_id) & (l.slice == slice_index)]
        lv = hit.level.iloc[0] if len(hit) else None
    return dict(case_id=case_id, view=view, slice=slice_index, vertebral_level=lv, structures=[
        dict(name=r.structure, display=STRUCTURES[r.structure].display, area_mm2=round(float(r.area_mm2), 1),
             visibility=round(float(r.area / mx[r.structure]), 3)) for r in row.itertuples()])


@mcp.tool()
def render_image(case_id: str, view: Literal["axial", "coronal", "sagittal"], slice_index: int,
                 labels: list[Label], render: RenderSettings | None = None,
                 hotspot_zones: list[HotspotZone] | None = None, output_dir: str | None = None,
                 name: str | None = None, include_base64: bool = True) -> dict:
    """Render a specific slice with labels. Returns image path (+ base64), answer key with arrow/label
    geometry, and optional hotspot masks (transparent PNGs, pixel-aligned with the image)."""
    lv = None
    if view == "axial":
        l = sel.levels()
        hit = l[(l.case_id.astype(str) == case_id) & (l.slice == slice_index)]
        lv = hit.level.iloc[0] if len(hit) else None
    return _render_and_save(case_id, view, slice_index, labels, render or RenderSettings(), hotspot_zones,
                            output_dir, name, include_base64, level=lv)


@mcp.tool()
def labeled_image(labels: list[Label], view: ViewT = "auto", strategy: StrategyT = "classical",
                  level: str | None = None, filters: CaseFilters | None = None,
                  render: RenderSettings | None = None, hotspot_zones: list[HotspotZone] | None = None,
                  context_structures: list[str] | None = None, top_k: int = 10, seed: int | None = None,
                  output_dir: str | None = None, name: str | None = None, include_base64: bool = True) -> dict:
    """Find a slice where every labeled structure (plus ``context_structures``) is visible, then render it.
    Picks randomly among the ``top_k`` best cases so repeated calls vary (fix ``seed`` to reproduce)."""
    names = _resolve_all([s for l in labels for s in l.structures] + (context_structures or [])
                         + [s for z in hotspot_zones or [] for s in z.structures])
    rng = np.random.default_rng(seed)
    cands = sel.candidates(names, view=view, strategy=strategy, filters=_filters(filters), level=level,
                           rng=rng, limit=max(top_k, 1))
    if not cands:
        raise ValueError("No slice shows all requested structures with the current filters. Try fewer "
                         "structures, another view, or relax filters (min_visibility, allow_* flags).")
    c = cands[int(rng.integers(len(cands)))]
    sel_info = dict(selection=dict(strategy=c.strategy, score=round(c.score, 3), visibility=c.visibility,
                                   areas_mm2=c.areas_mm2, candidates_considered=len(cands)))
    return _render_and_save(c.case_id, c.axis, c.slice, labels, render or RenderSettings(), hotspot_zones,
                            output_dir, name, include_base64, level=c.level, extra=sel_info,
                            context_structures=context_structures)


class BatchRequest(BaseModel):
    labels: list[Label]
    view: ViewT = "auto"
    strategy: StrategyT = "classical"
    level: str | None = None
    filters: CaseFilters | None = None
    render: RenderSettings | None = None
    hotspot_zones: list[HotspotZone] | None = None
    context_structures: list[str] | None = None
    seed: int | None = None
    name: str | None = None


@mcp.tool()
def batch_labeled_images(requests: list[BatchRequest], output_dir: str | None = None,
                         avoid_repeat_cases: bool = True) -> dict:
    """Run many labeled_image requests into one folder with a manifest.jsonl (no base64 in the reply).
    With ``avoid_repeat_cases`` each image comes from a different case when possible."""
    d = Path(output_dir) if output_dir else OUTPUT_DIR / "batches" / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    used: list[str] = []
    results = []
    with open(d / "manifest.jsonl", "w") as mf:
        for i, rq in enumerate(requests):
            f = rq.filters or CaseFilters()
            if avoid_repeat_cases:
                f = f.model_copy(update=dict(exclude_cases=[*f.exclude_cases, *used]))
            try:
                r = labeled_image(rq.labels, rq.view, rq.strategy, rq.level, f, rq.render, rq.hotspot_zones,
                                  rq.context_structures, seed=rq.seed, output_dir=str(d),
                                  name=rq.name or f"q{i + 1:03d}", include_base64=False)
                used.append(r["source"]["case_id"])
            except ValueError as e:
                r = dict(error=str(e), request_index=i)
            mf.write(json.dumps(r) + "\n")
            results.append(r)
    return dict(output_dir=str(d), manifest=str(d / "manifest.jsonl"), results=results)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
