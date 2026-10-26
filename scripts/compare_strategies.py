"""Render a contact sheet comparing slice-selection strategies.

Rows = target sets, columns = strategies. Each cell shows the top candidate with
targets outlined, captioned with case / slice / vertebral level / min visibility.

    uv run python scripts/compare_strategies.py --out outputs/strategies.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

from medimgq.catalog import resolve
from medimgq.selection import candidates
from medimgq.volume import apply_window, load_ct, load_mask, take_slice

COLORS = [(255, 80, 80), (80, 200, 255), (255, 210, 60), (120, 255, 120), (230, 120, 255)]
DEFAULT_SETS = [
    "pancreas",
    "left adrenal gland",
    "gallbladder",
    "pancreas; left adrenal gland; kidneys",
    "aorta; IVC; portal vein",
    "liver; spleen; stomach",
    "esophagus; trachea; aorta",
    "psoas; L3",
]
STRATEGIES = ["classical", "max_visibility", "max_area", "random"]
CELL = 320


def render_cell(cand, targets: list[str], window: str = "soft_tissue") -> Image.Image:
    img = apply_window(take_slice(load_ct(cand.case_id), cand.axis, cand.slice), window)
    rgb = np.stack([img] * 3, -1)
    for i, t in enumerate(targets):
        m = take_slice(load_mask(cand.case_id, t), cand.axis, cand.slice)
        rgb[find_boundaries(m, mode="outer")] = COLORS[i % len(COLORS)]
    im = Image.fromarray(rgb)
    im.thumbnail((CELL, CELL - 30))
    cell = Image.new("RGB", (CELL, CELL), "black")
    cell.paste(im, ((CELL - im.width) // 2, 0))
    d = ImageDraw.Draw(cell)
    vis = min(cand.visibility.values())
    d.text((4, CELL - 28), f"{cand.case_id} {cand.axis} #{cand.slice} {cand.level or ''}", fill="white",
           font=ImageFont.load_default(13))
    d.text((4, CELL - 14), f"min vis {vis:.2f}  [{cand.strategy}]", fill=(180, 180, 180),
           font=ImageFont.load_default(11))
    return cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="*", default=DEFAULT_SETS, help='target sets, structures separated by ";"')
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/strategies.png"))
    a = p.parse_args()

    sheet = Image.new("RGB", (220 + CELL * len(a.strategies), 24 + CELL * len(a.sets)), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for j, s in enumerate(a.strategies):
        d.text((220 + j * CELL + 8, 4), s, fill="white", font=ImageFont.load_default(15))
    for i, spec in enumerate(a.sets):
        groups = [resolve(x.strip()) for x in spec.split(";")]
        targets = [n for g in groups for n in g]
        y = 24 + i * CELL
        d.multiline_text((8, y + 8), spec.replace("; ", "\n"), fill="white", font=ImageFont.load_default(14))
        for j, strat in enumerate(a.strategies):
            c = candidates(targets, strategy=strat, rng=np.random.default_rng(a.seed + i))
            if not c:
                d.text((220 + j * CELL + 8, y + CELL // 2), "no candidate", fill="red")
                continue
            sheet.paste(render_cell(c[0], targets), (220 + j * CELL, y))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
