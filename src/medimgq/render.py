"""Render a labeled CT slice and the geometry of every label.

Markers
-------
- ``arrow_margin``  labels in side margins, arrows run in to each structure
                    (classic atlas style; lines rarely cross the anatomy twice)
- ``arrow_short``   short arrow next to each structure, direction chosen to avoid
                    covering other labeled structures and other labels
- ``on_structure``  label text placed directly on the structure
- ``none``          no label markers (use with ``outline`` or for a bare image)

``outline`` is independent and draws each labeled structure's contour.
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
    marker: str = "arrow_margin"  # arrow_margin | arrow_short | on_structure | none
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
    crop: str = "auto"  # auto (body, zoomed for small target clusters) | body | targets | full
    window: str | tuple[float, float] = "soft_tissue"
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

    def key_dicts(self) -> list[dict]:
        return [asdict(k) for k in self.key]


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


def _crop_box(hu: np.ndarray, masks: list[np.ndarray], axis: str, mode: str) -> tuple[int, int, int, int]:
    H, W = hu.shape
    if mode == "full":
        return 0, 0, W, H
    x0, y0, x1, y1 = _body_bbox(hu)
    if mode in ("auto", "targets") and masks:
        ys, xs = np.nonzero(np.any(np.stack(masks), axis=0))
        tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        bw, bh = x1 - x0, y1 - y0
        tw, th = tx1 - tx0, ty1 - ty0
        # Zoom when the labeled structures occupy a small part of the body
        # (e.g. neck structures on a slice that includes the shoulders).
        if mode == "targets" or max(tw / bw, th / bh) < 0.4:
            side = max(tw, th) * 2.2 + 40  # +60 mm context at 1.5 mm/px
            cx, cy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
            nx0, nx1 = max(x0, int(cx - side / 2)), min(x1, int(cx + side / 2))
            ny0, ny1 = max(y0, int(cy - side / 2)), min(y1, int(cy + side / 2))
            if mode == "targets" or (nx1 - nx0) < 0.75 * bw:
                return nx0, ny0, nx1, ny1
    if mode == "auto" and axis != "axial" and masks:
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
           options: RenderOptions | None = None, *, context: list[str] = ()) -> Rendered:
    """Render one slice. ``context`` names extra structures used only for cropping."""
    o = options or RenderOptions()
    hu = take_slice(load_ct(case_id), axis, index)
    names = [n for spec in labels for n in spec.structures]
    masks = {n: take_slice(load_mask(case_id, n), axis, index) for n in dict.fromkeys([*names, *context])}
    present = [m for m in masks.values() if m.any()]
    x0, y0, x1, y1 = _crop_box(hu, present, axis, o.crop)
    hu = hu[y0:y1, x0:x1]
    masks = {n: m[y0:y1, x0:x1] for n, m in masks.items()}

    s = o.size / max(hu.shape)
    W, H = round(hu.shape[1] * s), round(hu.shape[0] * s)
    base = Image.fromarray(apply_window(hu, o.window)).resize((W, H), Image.Resampling.LANCZOS).convert("RGB")

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
        _place_short(d, labels, key, anchors, masks, shift, s, font, pad, lw, o, (W, H))
    elif o.marker == "arrow_margin":
        _place_margin(d, labels, key, anchors, shift, font, pad, lw, o, (W, H), margin)

    if o.attribution:
        fsz = max(10, round(o.size * 0.017))
        d.text((canvas.width / 2, H + footer / 2), o.footer_text, font=_font(fsz) if fsz >= 12 else ImageFont.load_default(fsz),
               fill=(150, 150, 150), anchor="mm")
    return Rendered(canvas, key, (x0, y0, x1, y1), s, off)


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


def _place_short(d, labels, key, anchors, masks, shift, s, font, pad, lw, o, size) -> None:
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
                    lc = (cx + math.cos(a) * L * scale, cy + math.sin(a) * L * scale)
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
                    bx = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    cost += 15 * _segment_mask_cost(bx, bx, [masks[n] for n in spec.structures], s)
                    cost += 10 * _segment_mask_cost(bx, bx, others, s)
                    cost += scale * 2  # prefer short arrows
                    if best is None or cost < best[0]:
                        best = (cost, box, tails)
                if best and best[0] < 5:
                    break
            _, box, tails = best
            placed_boxes.append(box)
            for tail, tip in zip(tails, tg):
                tip_s = _pull_back(tail, tip, lw * 0.5)
                placed_segs.append((tail, tip))
                _draw_arrow(d, shift(tail), shift(tip_s), o.marker_color, lw)
                k.arrows_px.append((_r(shift(tail)), _r(shift(tip_s))))
            sb = (box[0] + shift((0, 0))[0], box[1], box[2] + shift((0, 0))[0], box[3])
            k.label_box_px = k.label_box_px or tuple(round(v, 1) for v in sb)
            _draw_text(d, sb, spec.label, font, o)


def _place_margin(d, labels, key, anchors, shift, font, pad, lw, o, size, margin) -> None:
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
                tip_c = shift(tip)
                tail = (box[2] + 3, (box[1] + box[3]) / 2) if side == "left" else (box[0] - 3, (box[1] + box[3]) / 2)
                tip_s = _pull_back(tail, tip_c, lw * 0.5)
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
