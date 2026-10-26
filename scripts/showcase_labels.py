"""Render the same scenes with every label preset, for judging label styles.

    uv run python scripts/showcase_labels.py --out outputs/label_presets.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medimgq.catalog import resolve
from medimgq.render import PRESETS, LabelSpec, RenderOptions, render
from medimgq.selection import candidates

# (title, view, label groups); each group is one label, ";" separates structures in a group
SCENES = [
    ("Upper abdomen, 5 options", "auto", ["pancreas", "left adrenal gland", "kidneys", "IVC", "aorta"]),
    ("Coronal abdomen, 4 options", "coronal", ["liver", "spleen", "psoas", "urinary bladder"]),
    ("Neck, 4 options", "auto", ["thyroid", "trachea", "esophagus", "common carotid arteries"]),
]
CELL = 460


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--presets", nargs="*", default=list(PRESETS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("outputs/label_presets.png"))
    a = p.parse_args()

    sheet = Image.new("RGB", (CELL * len(a.presets), 30 + (CELL + 30) * len(SCENES)), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for j, pr in enumerate(a.presets):
        d.text((j * CELL + 8, 6), pr, fill="white", font=ImageFont.load_default(18))
    for i, (title, view, groups) in enumerate(SCENES):
        resolved = [list(resolve(g)) for g in groups]
        targets = [n for g in resolved for n in g]
        rng = np.random.default_rng(a.seed + i)
        cands = candidates(targets, view=view, rng=rng)
        y = 30 + i * (CELL + 30)
        if not cands:
            d.text((8, y + 8), f"{title}: no candidate", fill="red")
            continue
        c = cands[int(rng.integers(min(10, len(cands))))]
        d.text((8, y + 6), f"{title} — {c.case_id} {c.axis} #{c.slice} {c.level or ''}", fill="white",
               font=ImageFont.load_default(16))
        letters = "ABCDEFGH"
        specs = [LabelSpec(letters[k], g) for k, g in enumerate(resolved)]
        for j, pr in enumerate(a.presets):
            r = render(c.case_id, c.axis, c.slice, specs, RenderOptions(**PRESETS[pr], size=640))
            im = r.image
            im.thumbnail((CELL - 8, CELL - 8))
            sheet.paste(im, (j * CELL + 4 + (CELL - 8 - im.width) // 2, y + 30))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
