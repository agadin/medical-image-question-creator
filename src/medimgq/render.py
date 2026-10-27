"""Render a labeled CT slice and the geometry of every label.

Markers
-------
- ``arrow_short``   (default) short arrow next to each structure, direction chosen
                    to avoid covering other labeled structures and other labels
- ``arrow_margin``  labels in side margins, arrows run in to each structure
                    (classic atlas style)
- ``on_structure``  label text placed directly on the structure
- ``none``          no label markers (use with ``outline`` or for a bare image)

``outline`` is independent and draws each labeled structure's contour.
``tip`` puts arrow tips at the structure's deepest point (``center``) or just
inside its boundary on the side facing the label (``edge``).

``crop="auto"`` crops to the anatomical region: neck (excludes shoulders), chest
(lungs + chest wall, excludes arms), or the whole trunk for abdomen/pelvis.
A label text of ``""`` draws the marker without text (e.g. a lone arrow).

Structures sharing a label use ``shared_mode``: ``"multi_arrow"`` (one label,
an arrow to each) or ``"repeat"`` (the same label next to each structure).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import measure

from .catalog import STRUCTURES
from .config import TS_ATTRIBUTION
from .volume import apply_window, load_ct, load_mask, take_slice

Color = tuple[int, int, int]

PALETTE: list[Color] = [(255, 214, 10), (0, 200, 255), (255, 90, 90), (110, 230, 110),
                        (220, 130, 255), (255, 160, 60), (90, 140, 255), (255, 255, 255)]


@dataclass
class LabelSpec:
    label: str  # "A", "1", or "" for a marker without text
    structures: list[str]  # canonical names


@dataclass
class RenderOptions:
    marker: str = "arrow_short"  # arrow_short | arrow_margin | on_structure | none
    tip: str = "center"  # center (deepest point) | edge (just inside the boundary facing the label)
    outline: bool = False
    outline_colors: str = "per_label"  # per_label | single
    shared_mode: str = "multi_arrow"  # multi_arrow | repeat
    text_style: str = "halo"  # halo | boxed | boxed_dark
    marker_color: Color = (255, 255, 255)
    text_color: Color = (255, 255, 255)
    outline_color: Color = (255, 214, 10)
    font_size: int | None = None  # default scales with size
    line_width: int | None = None
    size: int = 768  # long side of the image area (excludes margins / footer)
    crop: str = "auto"  # auto (by region) | neck | chest | abdomen | body | targets | full
    window: str | tuple[float, float] = "auto"  # auto | a preset in volume.WINDOWS | (width, level)
    attribution: bool = True
    footer_text: str = TS_ATTRIBUTION


PRESETS: dict[str, dict] = {
    "arrow_margin": dict(marker="arrow_margin"),
    "arrow_short": dict(marker="arrow_short"),
    "on_structure": dict(marker="on_structure"),
    "outline_labeled": dict(marker="on_structure", outline=True),
    "outline_arrows": dict(marker="arrow_short", outline=True),
    "outline_only": dict(marker="none", outline=True),
    "boxed_margin": dict(marker="arrow_margin", text_style="boxed"),
    "yellow_short": dict(marker="arrow_short", marker_color=(255, 214, 10), text_color=(255, 214, 10)),
    "edge_arrows": dict(marker="arrow_short", tip="edge"),
    "bare": dict(marker="none"),
}


@dataclass
class KeyEntry:
    label: str
    structures: list[str]
    display_names: list[str]
    anchors_px: list[tuple[float, float]]  # (x, y) arrow tips / text centers in output image
    label_box_px: tuple[float, float, float, float] | None  # x0, y0, x1, y1
    arrows_px: list[tuple[tuple[float, float], tuple[float, float]]]  # (tail, tip)
    outlines_px: list[list[tuple[float, float]]] = field(default_factory=list)
    area_mm2: dict[str, float] = field(default_factory=dict)


@dataclass
class Rendered:
    image: Image.Image
    key: list[KeyEntry]
    crop_box: tuple[int, int, int, int]  # in slice pixels: x0, y0, x1, y1
    scale: float  # output px per slice px
    offset: tuple[int, int]  # top-left of image area within the output canvas
    case_id: str = ""
    axis: str = ""
    index: int = 0
    window: str | tuple[float, float] = ""
    region: str | None = None

    def key_dicts(self) -> list[dict]:
        return [asdict(k) for k in self.key]

    def mask_image(self, structures: list[str], color: Color = (0, 0, 0)) -> Image.Image:
        """RGBA mask the size of ``image``: the structures' pixels painted, the rest transparent.

        Pixel-aligned with the rendered image, so it can serve as a hotspot answer region.
        """
        x0, y0, x1, y1 = self.crop_box
        m = np.zeros((y1 - y0, x1 - x0), bool)
        for n in structures:
            m |= take_slice(load_mask(self.case_id, n), self.axis, self.index)[y0:y1, x0:x1]
        W = round((x1 - x0) * self.scale)
        H = round((y1 - y0) * self.scale)
        small = Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.Resampling.BILINEAR)
        alpha = np.zeros((self.image.height, self.image.width), np.uint8)
        a = (np.asarray(small) >= 128).astype(np.uint8) * 255
        alpha[self.offset[1]:self.offset[1] + H, self.offset[0]:self.offset[0] + W] = a
        rgba = np.zeros((*alpha.shape, 4), np.uint8)
        rgba[..., :3] = color
        rgba[..., 3] = alpha
        return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------- geometry


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size, index=1) if path.endswith(".ttc") else ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _body_bbox(hu: np.ndarray) -> tuple[int, int, int, int]:
    body = hu > -300
    lab, n = ndimage.label(body)
    if n:
        sizes = ndimage.sum_labels(body, lab, index=np.arange(1, n + 1))
        body = lab == int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(body)
    if not len(xs):
        return 0, 0, hu.shape[1], hu.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# Structures whose extent on the slice defines each crop region.
REGION_ANCHORS: dict[str, tuple[str, ...]] = {
    "neck": ("trachea", "esophagus", "thyroid_gland", "common_carotid_artery_left", "common_carotid_artery_right",
             *(f"vertebrae_C{i}" for i in range(1, 8))),
    "chest": ("lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
              "lung_lower_lobe_right", "heart", "sternum"),
}
REGION_PAD_MM = {"neck": 35.0, "chest": 18.0}


def auto_window(targets: list[str]) -> str:
    """Lung window if any lung lobe is labeled (lobes are only distinguishable there),
    bone when bones dominate, else soft tissue."""
    systems = [STRUCTURES[t].system for t in targets]
    if any(t.startswith("lung_") for t in targets):
        return "lung"
    if systems and systems.count("skeletal") * 2 > len(systems):
        return "bone"
    return "soft_tissue"


def auto_region(axis: str, level: str | None, targets: list[str], lung_px: int) -> str:
    """Pick a crop region for ``crop="auto"`` from the targets and the slice's vertebral level."""
    regions = [STRUCTURES[t].region for t in targets]
    if regions and regions.count("neck") * 2 > len(regions):
        return "neck"
    if axis == "axial":
        if level and level[0] == "C":
            return "neck"
        if level and level[0] == "T" and lung_px > 400:
            return "chest"
        return "abdomen"
    if regions and regions.count("thorax") * 2 > len(regions):
        return "chest"
    return "abdomen"


def _region_box(region_masks: list[np.ndarray], pad_px: int):
    if not region_masks:
        return None
    ys, xs = np.nonzero(np.any(np.stack(region_masks), axis=0))
    if not len(xs):
        return None
    return xs.min() - pad_px, ys.min() - pad_px, xs.max() + 1 + pad_px, ys.max() + 1 + pad_px


def _crop_box(hu: np.ndarray, masks: list[np.ndarray], axis: str, mode: str,
              region_masks: list[np.ndarray] = (), region: str | None = None) -> tuple[int, int, int, int]:
    H, W = hu.shape
    if mode == "full":
        return 0, 0, W, H
    x0, y0, x1, y1 = _body_bbox(hu)
    tbox = _region_box(list(masks), 0) if masks else None
    if mode == "targets" and tbox:
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        side = max(tw, th) * 2.2 + 40  # +60 mm context at 1.5 mm/px
        cx, cy = (tbox[0] + tbox[2]) / 2, (tbox[1] + tbox[3]) / 2
        return (max(x0, int(cx - side / 2)), max(y0, int(cy - side / 2)),
                min(x1, int(cx + side / 2)), min(y1, int(cy + side / 2)))
    if region in REGION_PAD_MM:
        rbox = _region_box(list(region_masks), int(REGION_PAD_MM[region] / 1.5))
        if rbox:
            if tbox:  # never crop away a labeled structure
                rbox = (min(rbox[0], tbox[0] - 8), min(rbox[1], tbox[1] - 8),
                        max(rbox[2], tbox[2] + 8), max(rbox[3], tbox[3] + 8))
            rx0, ry0, rx1, ry1 = (max(x0, rbox[0]), max(y0, rbox[1]), min(x1, rbox[2]), min(y1, rbox[3]))
            # keep the aspect ratio reasonable (at most 1.6:1)
            rw, rh = rx1 - rx0, ry1 - ry0
            if rw > 1.6 * rh:
                grow = (rw / 1.6 - rh) / 2
                ry0, ry1 = max(y0, int(ry0 - grow)), min(y1, int(ry1 + grow))
            elif rh > 1.6 * rw:
                grow = (rh / 1.6 - rw) / 2
                rx0, rx1 = max(x0, int(rx0 - grow)), min(x1, int(rx1 + grow))
            return int(rx0), int(ry0), int(rx1), int(ry1)
    if axis != "axial" and masks:
        # Coronal/sagittal scans may span head to toe: keep the body's width but
        # limit height to the targets plus generous context.
        ys = np.nonzero(np.any(np.stack(masks), axis=(0, 2)))[0]
        ty0, ty1 = int(ys.min()), int(ys.max()) + 1
        width = x1 - x0
        half = max((ty1 - ty0) * 0.5 + 0.35 * width, 0.6 * width)
        cy = (ty0 + ty1) / 2
        y0, y1 = max(y0, int(cy - half)), min(y1, int(cy + half))
    pad = int(0.04 * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def _anchor(mask: np.ndarray) -> tuple[float, float] | None:
    """Deepest point of the largest component (always inside the structure)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
    comp = lab == int(np.argmax(sizes)) + 1
    dt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return float(x), float(y)


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    cs = measure.find_contours(np.pad(mask.astype(float), 1), 0.5)
    return [measure.approximate_polygon(c, 0.6)[:, ::-1] - 1 for c in cs if len(c) >= 6]  # (x, y)


def _text_box(draw: ImageDraw.ImageDraw, text: str, font, center: tuple[float, float], pad: int):
    l, t, r, b = draw.textbbox((0, 0), text or " ", font=font)
    w, h = r - l, b - t
    cx, cy = center
    return (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _box_edge_point(box, toward: tuple[float, float]) -> tuple[float, float]:
    """Point where the segment from the box center toward ``toward`` leaves the box."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == dy == 0:
        return cx, cy
    hw, hh = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * t, cy + dy * t


# ---------------------------------------------------------------- drawing


def _draw_arrow(d: ImageDraw.ImageDraw, tail, tip, color: Color, width: int, head: bool = True) -> None:
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    hl, hw = width * 4.0, width * 2.6
    base = (tip[0] - ux * hl, tip[1] - uy * hl)
    poly = [tip, (base[0] - uy * hw, base[1] + ux * hw), (base[0] + uy * hw, base[1] - ux * hw)]
    end = base if head else tip
    d.line([tail, end], fill=(0, 0, 0), width=width + 3)
    if head:
        d.polygon(poly, fill=(0, 0, 0), outline=(0, 0, 0), width=2)
    d.line([tail, end], fill=color, width=width)
    if head:
        d.polygon(poly, fill=color)


def _draw_text(d: ImageDraw.ImageDraw, box, text: str, font, o: RenderOptions) -> None:
    if not text:
        return
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if o.text_style == "boxed":
        d.rectangle(box, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        d.text((cx, cy), text, font=font, fill=(0, 0, 0), anchor="mm")
    elif o.text_style == "boxed_dark":
        d.rectangle(box, fill=(0, 0, 0), outline=o.text_color, width=1)
        d.text((cx, cy), text, font=font, fill=o.text_color, anchor="mm")
    else:
        d.text((cx, cy), text, font=font, fill=o.text_color, anchor="mm",
               stroke_width=max(2, font.size // 9), stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------- main


def render(case_id: str, axis: str, index: int, labels: list[LabelSpec],
           options: RenderOptions | None = None, *, context: list[str] = (), level: str | None = None) -> Rendered:
    """Render one slice. ``context`` names extra structures used only for cropping;
    ``level`` (the slice's vertebral level) helps ``crop="auto"`` choose a region."""
    o = options or RenderOptions()
    hu = take_slice(load_ct(case_id), axis, index)
    names = [n for spec in labels for n in spec.structures]
    masks = {n: take_slice(load_mask(case_id, n), axis, index) for n in dict.fromkeys([*names, *context])}
    present = [m for m in masks.values() if m.any()]
    region = None
    region_masks: list[np.ndarray] = []
    if o.crop in ("auto", "neck", "chest", "abdomen"):
        lungs = [take_slice(load_mask(case_id, n), axis, index) for n in REGION_ANCHORS["chest"][:5]] if axis == "axial" else []
        region = o.crop if o.crop != "auto" else auto_region(axis, level, names or list(context), int(sum(m.sum() for m in lungs)))
        if region in REGION_ANCHORS:
            region_masks = [take_slice(load_mask(case_id, n), axis, index) for n in REGION_ANCHORS[region]]
    x0, y0, x1, y1 = _crop_box(hu, present, axis, o.crop, region_masks, region)
    hu = hu[y0:y1, x0:x1]
    masks = {n: m[y0:y1, x0:x1] for n, m in masks.items()}

    s = o.size / max(hu.shape)
    W, H = round(hu.shape[1] * s), round(hu.shape[0] * s)
    window = auto_window(names or list(context)) if o.window == "auto" else o.window
    base = Image.fromarray(apply_window(hu, window)).resize((W, H), Image.Resampling.LANCZOS).convert("RGB")

    fs = o.font_size or max(14, round(o.size * 0.042))
    lw = o.line_width or max(2, round(o.size / 300))
    font = _font(fs)
    pad = max(3, fs // 6)
    scratch = ImageDraw.Draw(base)

    # Per-structure anchors in output pixels.
    anchors: dict[str, tuple[float, float]] = {}
    for n in names:
        a = _anchor(masks[n])
        if a is not None:
            anchors[n] = ((a[0] + 0.5) * s, (a[1] + 0.5) * s)

    owner = {v: n for n, v in anchors.items()}

    def tip_for(tail, center):
        """Arrow tip (image-area coords) for an arrow from ``tail`` to a structure's anchor."""
        if o.tip != "edge" or center not in owner:
            return _pull_back(tail, center, lw * 0.5)
        m = masks[owner[center]]
        n = 64
        for t in np.linspace(0, 1, n):
            x, y = tail[0] + (center[0] - tail[0]) * t, tail[1] + (center[1] - tail[1]) * t
            ix, iy = int(x / s), int(y / s)
            if 0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and m[iy, ix]:
                # step a little inside the boundary, but never past the anchor
                inset = min(3 * s, math.hypot(center[0] - x, center[1] - y))
                L = math.hypot(center[0] - tail[0], center[1] - tail[1]) or 1.0
                return (x + (center[0] - tail[0]) / L * inset, y + (center[1] - tail[1]) / L * inset)
        return _pull_back(tail, center, lw * 0.5)

    # Margins (arrow_margin only) and footer.
    margin = 0
    if o.marker == "arrow_margin" and any(spec.label for spec in labels):
        widest = max(scratch.textbbox((0, 0), spec.label or " ", font=font)[2] for spec in labels)
        margin = int(widest + 2 * pad + fs * 1.2)
    footer = max(14, round(o.size * 0.022)) + 8 if o.attribution else 0
    canvas = Image.new("RGB", (W + 2 * margin, H + footer), (0, 0, 0))
    canvas.paste(base, (margin, 0))
    d = ImageDraw.Draw(canvas)
    off = (margin, 0)

    def shift(p):
        return (p[0] + off[0], p[1] + off[1])

    key: list[KeyEntry] = []
    for i, spec in enumerate(labels):
        pts = [shift(anchors[n]) for n in spec.structures if n in anchors]
        key.append(KeyEntry(
            label=spec.label, structures=list(spec.structures),
            display_names=[STRUCTURES[n].display for n in spec.structures],
            anchors_px=pts, label_box_px=None, arrows_px=[],
            area_mm2={n: round(float(masks[n].sum()) * 2.25, 1) for n in spec.structures},
        ))

    # Outlines first so markers sit on top.
    if o.outline:
        for i, (spec, k) in enumerate(zip(labels, key)):
            col = PALETTE[i % len(PALETTE)] if o.outline_colors == "per_label" else o.outline_color
            for n in spec.structures:
                for c in _contours(masks[n]):
                    poly = [shift(((x + 0.5) * s, (y + 0.5) * s)) for x, y in c]
                    d.line(poly + [poly[0]], fill=col, width=lw)
                    k.outlines_px.append([(round(x, 1), round(y, 1)) for x, y in poly])

    if o.marker == "on_structure":
        _place_on_structure(d, labels, key, anchors, shift, font, pad, o)
    elif o.marker == "arrow_short":
        def edge_base(center, ang):
            """Where a ray from the structure's anchor at angle ``ang`` leaves the structure."""
            m = masks[owner[center]]
            x, y = center
            for _ in range(400):
                nx, ny = x + math.cos(ang) * s, y + math.sin(ang) * s
                ix, iy = int(nx / s), int(ny / s)
                if not (0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and m[iy, ix]):
                    break
                x, y = nx, ny
            return x, y

        _place_short(d, labels, key, anchors, masks, shift, s, font, pad, lw, o, (W, H), tip_for,
                     edge_base if o.tip == "edge" else None)
    elif o.marker == "arrow_margin":
        _place_margin(d, labels, key, anchors, shift, font, pad, lw, o, (W, H), margin, tip_for)

    if o.attribution:
        fsz = max(10, round(o.size * 0.017))
        d.text((canvas.width / 2, H + footer / 2), o.footer_text, font=_font(fsz) if fsz >= 12 else ImageFont.load_default(fsz),
               fill=(150, 150, 150), anchor="mm")
    return Rendered(canvas, key, (x0, y0, x1, y1), s, off, case_id, axis, index, window, region)


def _place_on_structure(d, labels, key, anchors, shift, font, pad, o) -> None:
    for spec, k in zip(labels, key):
        for p in (shift(anchors[n]) for n in spec.structures if n in anchors):
            box = _text_box(d, spec.label, font, p, pad)
            k.label_box_px = k.label_box_px or tuple(round(v, 1) for v in box)
            if spec.label:
                _draw_text(d, box, spec.label, font, o)
            else:  # no text: a small dot marks the structure
                r = font.size / 5
                d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=o.marker_color, outline=(0, 0, 0), width=2)


def _segment_mask_cost(p0, p1, masks_px: list[np.ndarray], s: float) -> float:
    """Fraction of samples along the segment lying on any of the given masks."""
    if not masks_px:
        return 0.0
    hits = 0
    N = 24
    h, w = masks_px[0].shape
    for t in np.linspace(0, 1, N):
        x = int((p0[0] + (p1[0] - p0[0]) * t) / s)
        y = int((p0[1] + (p1[1] - p0[1]) * t) / s)
        if 0 <= x < w and 0 <= y < h and any(m[y, x] for m in masks_px):
            hits += 1
    return hits / N


def _place_short(d, labels, key, anchors, masks, shift, s, font, pad, lw, o, size, tip_for, edge_base=None) -> None:
    W, H = size
    L = max(font.size * 2.2, 0.08 * max(W, H))
    placed_boxes: list[tuple] = []
    placed_segs: list[tuple] = []
    angles = [math.radians(a) for a in range(0, 360, 15)]
    # Fill a cost map of all labeled structures to avoid covering them.
    for spec, k in zip(labels, key):
        own = set(spec.structures)
        others = [masks[n] for n in masks if n not in own and n in anchors]
        tips = [anchors[n] for n in spec.structures if n in anchors]
        if not tips:
            continue
        groups = [tips] if o.shared_mode == "multi_arrow" else [[t] for t in tips]
        for tg in groups:
            cx = sum(t[0] for t in tg) / len(tg)
            cy = sum(t[1] for t in tg) / len(tg)
            best = None
            for scale in (1.0, 1.4, 1.9):
                for a in angles:
                    # with edge tips, measure the arrow from where it meets the structure
                    bx0, by0 = edge_base(tg[0], a) if edge_base and len(tg) == 1 else (cx, cy)
                    lc = (bx0 + math.cos(a) * L * scale, by0 + math.sin(a) * L * scale)
                    box = _text_box(d, spec.label, font, lc, pad)
                    cost = 0.0
                    if box[0] < 2 or box[1] < 2 or box[2] > W - 2 or box[3] > H - 2:
                        cost += 1000
                    cost += sum(_overlap(box, b) for b in placed_boxes) * 5
                    cost += 60 * sum(_box_hits_segment(box, q0, q1) for q0, q1 in placed_segs)
                    tails = [_box_edge_point(box, t) for t in tg]
                    for tail, tip in zip(tails, tg):
                        cost += 30 * _segment_mask_cost(tail, tip, others, s)
                        for q0, q1 in placed_segs:
                            if _segments_cross(tail, tip, q0, q1):
                                cost += 40
                    # label text should not sit on the structure it points to
                    cost += 15 * _box_mask_frac(box, [masks[n] for n in spec.structures], s)
                    cost += 10 * _box_mask_frac(box, others, s)
                    cost += scale * 2  # prefer short arrows
                    if best is None or cost < best[0]:
                        best = (cost, box, tails)
                if best and best[0] < 5:
                    break
            _, box, tails = best
            placed_boxes.append(box)
            for tail, tip in zip(tails, tg):
                tip_s = tip_for(tail, tip)
                placed_segs.append((tail, tip))
                _draw_arrow(d, shift(tail), shift(tip_s), o.marker_color, lw)
                k.arrows_px.append((_r(shift(tail)), _r(shift(tip_s))))
            sb = (box[0] + shift((0, 0))[0], box[1], box[2] + shift((0, 0))[0], box[3])
            k.label_box_px = k.label_box_px or tuple(round(v, 1) for v in sb)
            _draw_text(d, sb, spec.label, font, o)


def _place_margin(d, labels, key, anchors, shift, font, pad, lw, o, size, margin, tip_for) -> None:
    W, H = size
    items = []  # (side, desired_y, spec, key, tips)
    for spec, k in zip(labels, key):
        tips = [anchors[n] for n in spec.structures if n in anchors]
        if not tips:
            continue
        groups = [tips] if o.shared_mode == "multi_arrow" else [[t] for t in tips]
        for tg in groups:
            mx = sum(t[0] for t in tg) / len(tg)
            my = sum(t[1] for t in tg) / len(tg)
            items.append(("left" if mx < W / 2 else "right", my, spec, k, tg))
    lh = font.size + 2 * pad
    gap = lh * 1.35
    for side in ("left", "right"):
        col = sorted([it for it in items if it[0] == side], key=lambda it: it[1])
        ys = [it[1] for it in col]
        for i in range(1, len(ys)):  # push down to keep spacing
            ys[i] = max(ys[i], ys[i - 1] + gap)
        overflow = (ys[-1] + lh / 2 - (H - 4)) if ys else 0
        if overflow > 0:  # shift up, then push up from the bottom
            ys[-1] -= overflow
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = min(ys[i], ys[i + 1] - gap)
        for (_, _, spec, k, tg), y in zip(col, ys):
            cx = margin / 2 if side == "left" else margin + W + margin / 2
            box = _text_box(d, spec.label, font, (cx, max(lh / 2 + 2, y)), pad)
            for tip in tg:
                tail = (box[2] + 3, (box[1] + box[3]) / 2) if side == "left" else (box[0] - 3, (box[1] + box[3]) / 2)
                tip_s = shift(tip_for((tail[0] - margin, tail[1]), tip))
                _draw_arrow(d, tail, tip_s, o.marker_color, lw)
                k.arrows_px.append((_r(tail), _r(tip_s)))
            k.label_box_px = k.label_box_px or tuple(round(v, 1) for v in box)
            _draw_text(d, box, spec.label, font, o)


def _pull_back(tail, tip, dist):
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    L = math.hypot(dx, dy) or 1.0
    return (tip[0] - dx / L * dist, tip[1] - dy / L * dist)


def _r(p):
    return (round(p[0], 1), round(p[1], 1))


def _box_mask_frac(box, masks_px: list[np.ndarray], s: float) -> float:
    """Fraction of a 3x3 grid of points in ``box`` lying on any of the masks."""
    if not masks_px:
        return 0.0
    h, w = masks_px[0].shape
    hits = 0
    for fx in (0.15, 0.5, 0.85):
        for fy in (0.15, 0.5, 0.85):
            x = int((box[0] + (box[2] - box[0]) * fx) / s)
            y = int((box[1] + (box[3] - box[1]) * fy) / s)
            if 0 <= x < w and 0 <= y < h and any(m[y, x] for m in masks_px):
                hits += 1
    return hits / 9


def _box_hits_segment(box, p0, p1) -> bool:
    for t in np.linspace(0, 1, 16):
        x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return True
    return False


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0) and (orient(q1, q2, p1) * orient(q1, q2, p2) < 0)
