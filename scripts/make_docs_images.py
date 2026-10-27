"""Render the images used in README.md into docs/images/ (fixed cases, so output is reproducible).

    uv run python scripts/make_docs_images.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medimgq.catalog import resolve
from medimgq.render import PRESETS, LabelSpec, RenderOptions, render

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
BG = (246, 247, 249)
INK = (28, 32, 38)
MUTED = (96, 104, 116)
TILE = 360

ABDOMEN = ("s1361", "axial", 262, "L1")  # pancreas, left adrenal, kidneys, IVC, aorta
CORONAL = ("s0680", "coronal", 106, None)
CHEST = ("s0423", "axial", 302, "T9")
NECK = ("s0763", "axial", 422, "C7")
ANEURYSM = ("s1400", "axial", 315, "T9")
GOITER = ("s0522", "axial", 173, "T3")
ANEURYSM_HOT = ("s1365", "axial", 313, "T9")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


def labels(*groups: str) -> list[LabelSpec]:
    return [LabelSpec(str(i + 1), list(resolve(g))) for i, g in enumerate(groups)]


def shot(case, specs, size=TILE * 2, **opts) -> Image.Image:
    cid, axis, idx, level = case
    return render(cid, axis, idx, specs, RenderOptions(size=size, **opts), level=level).image


def sheet(title: str, tiles: list[tuple[str, str, Image.Image]], cols: int, path: str, tile: int = TILE) -> None:
    """Grid of captioned tiles: (caption, sub-caption, image)."""
    rows = (len(tiles) + cols - 1) // cols
    head, cap, pad = 56, 52, 16
    W = cols * (tile + pad) + pad
    H = head + rows * (tile + cap + pad) + pad
    s = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(s)
    d.text((pad, 16), title, font=font(26, True), fill=INK)
    for i, (c1, c2, im) in enumerate(tiles):
        x = pad + (i % cols) * (tile + pad)
        y = head + (i // cols) * (tile + cap + pad)
        im = im.copy()
        im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        d.rounded_rectangle((x - 1, y - 1, x + tile, y + tile + cap), radius=8, fill=(255, 255, 255),
                            outline=(222, 225, 230))
        box = Image.new("RGB", (tile, tile), (0, 0, 0))
        box.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
        s.paste(box, (x, y))
        d.text((x + 10, y + tile + 7), c1, font=font(17, True), fill=INK)
        d.text((x + 10, y + tile + 29), c2, font=font(13), fill=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    s.save(OUT / path, quality=88, method=6)
    print(OUT / path)


def overlay(img: Image.Image, masks: list[tuple[Image.Image, tuple[int, int, int]]]) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(float)
    for m, col in masks:
        al = np.asarray(m)[..., 3] > 0
        a[al] = a[al] * 0.45 + np.array(col) * 0.55
    return Image.fromarray(a.astype(np.uint8))


def main() -> None:
    abd = labels("pancreas", "left adrenal gland", "kidneys", "IVC", "aorta")

    # Hero
    shot(ABDOMEN, abd, size=900).save(OUT / "hero.webp", quality=90, method=6)

    # 1. Label presets
    desc = {
        "arrow_short": "default: short arrows, placed to avoid overlaps",
        "arrow_margin": "labels in side margins, atlas style",
        "on_structure": "number sits on the structure",
        "outline_labeled": "outlines + numbers on structures",
        "outline_arrows": "outlines + short arrows",
        "outline_only": "outlines, no labels",
        "boxed_margin": "boxed labels in margins",
        "yellow_short": "yellow arrows and text",
        "edge_arrows": "tips stop just inside the edge",
        "bare": "no markers (hotspot questions)",
    }
    sheet("Label presets  (render.preset)",
          [(p, desc[p], shot(ABDOMEN, abd, **PRESETS[p])) for p in PRESETS], 5, "presets.webp")

    # 2. Views
    sag_specs = labels("L1", "L3", "L5", "sacrum")
    tiles = [
        ("axial", "pancreas level (L1)", shot(ABDOMEN, abd)),
        ("coronal", "liver, spleen, psoas, bladder",
         shot(CORONAL, labels("liver", "spleen", "psoas", "urinary bladder"))),
    ]
    from medimgq.selection import candidates

    c = candidates([n for s in sag_specs for n in s.structures], view="sagittal",
                   rng=np.random.default_rng(2))[0]
    tiles.append(("sagittal", "lumbar spine and sacrum (bone window)",
                  shot((c.case_id, "sagittal", c.slice, None), sag_specs)))
    sheet("Views  (view: auto | axial | coronal | sagittal)", tiles, 3, "views.webp")

    # 3. Windows
    sheet("Windows  (render.window: auto | preset | [width, level])", [
        (w, sub, shot(CHEST, [], marker="none", window=w))
        for w, sub in [("soft_tissue", "W400 / L40 (default)"), ("lung", "W1500 / L-600, auto for lobes"),
                       ("bone", "W1800 / L400, auto for bones"), ("mediastinum", "W350 / L50")]
    ], 4, "windows.webp")

    # 4. Crop
    neck = labels("thyroid", "trachea", "esophagus", "common carotid arteries")
    sheet("Crop  (render.crop)", [
        ("auto", "picks neck / chest / abdomen", shot(NECK, neck, crop="auto")),
        ("body", "whole body cross-section", shot(NECK, neck, crop="body")),
        ("targets", "tight zoom on labeled structures", shot(NECK, neck, crop="targets")),
        ("full", "entire scan field of view", shot(NECK, neck, crop="full")),
    ], 4, "crop.webp")

    # 5. Arrow tips and shared labels
    kid = [LabelSpec("1", list(resolve("kidneys"))), LabelSpec("2", list(resolve("liver")))]
    big = labels("liver", "stomach", "left kidney")
    sheet("Arrow tips and shared labels", [
        ("tip: center", "deepest point of the structure", shot(ABDOMEN, big, tip="center")),
        ("tip: edge", "just inside the boundary", shot(ABDOMEN, big, tip="edge")),
        ("shared_mode: multi_arrow", "one label, an arrow to each", shot(ABDOMEN, kid, shared_mode="multi_arrow")),
        ("shared_mode: repeat", "label repeated at each", shot(ABDOMEN, kid, shared_mode="repeat")),
    ], 4, "tips_shared.webp")

    # 5b. Which piece to point at
    t8 = ("s0651", "axial", 216, "T8")
    sheet("Which piece to point at  (labels[].component)", [
        ("component: largest", "default: the aortic root here",
         shot(t8, [LabelSpec("1", ["aorta"], "largest"), LabelSpec("2", ["esophagus"])])),
        ("component: posterior", "the descending aorta",
         shot(t8, [LabelSpec("1", ["aorta"], "posterior"), LabelSpec("2", ["esophagus"])])),
    ], 2, "component.webp", tile=440)

    # 6. Hotspot masks
    cid, axis, idx, level = ANEURYSM_HOT
    zones = [("esophagus", (60, 200, 90)), ("aorta", (230, 70, 70)), ("heart", (70, 140, 240)),
             ("spinal_cord", (240, 190, 40))]
    r = render(cid, axis, idx, [], RenderOptions(size=TILE * 2, marker="none", window="soft_tissue"),
               level=level, context=[z for z, _ in zones])
    over = overlay(r.image, [(r.mask_image([z]), col) for z, col in zones])
    single = Image.new("RGBA", r.image.size, (0, 0, 0, 255))
    single.alpha_composite(r.mask_image(["esophagus"], color=(255, 255, 255)))
    sheet("Hotspot masks  (hotspot_zones)", [
        ("image", "bare render the student clicks", r.image),
        ("zones", "esophagus, aorta, heart, cord", over),
        ("mask: esophagus", "transparent PNG, pixel-aligned", single.convert("RGB")),
    ], 3, "hotspot.webp")

    # 7. Real pathology
    sheet("Real pathology found by measuring the segmentations", [
        ("thoracic aortic aneurysm", "s1400: ~9 cm, mural thrombus",
         shot(ANEURYSM, labels("aorta", "esophagus", "heart"), window="soft_tissue")),
        ("retrosternal goiter", "s0522: thyroid 63 mL vs 13 mL median",
         shot(GOITER, labels("thyroid", "trachea", "esophagus"), crop="neck")),
    ], 2, "pathology.webp", tile=440)


if __name__ == "__main__":
    main()
