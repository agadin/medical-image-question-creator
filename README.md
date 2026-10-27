# medimgq — medical image question creator

An MCP server that another AI can call to generate **labeled medical images** for
medical-school practice exams, together with a machine-readable **answer key**.
CT first (TotalSegmentator, 117 structures); X-ray and MRI later.

> Status: early development. The MCP server, slice search, and renderer work. Question writing is
> left to the calling AI: the server provides images, answer keys, and hotspot masks.

## Question type: `labeled`

The calling AI asks for an image with one or more structures as answer options, e.g.
*"axial CT with the pancreas, left adrenal gland, and both kidneys (as one label)"*.
The tool:

- picks a case and the most classical view/slice for the targets (overridable),
- finds a slice where every requested structure is clearly visible,
- adds distractor labels (optional, confusable by default),
- labels with letters, numbers, or nothing (bare image), using arrows, on-structure
  markers, or outlines,
- returns the PNG (file path + base64) and a key mapping each label to its structure(s).

## MCP server

```bash
claude mcp add medimgq -- uv run --directory /path/to/medical_image_question_creator medimgq-mcp
```

| Tool | Purpose |
|---|---|
| `list_structures` | labelable structures, synonyms, groups |
| `dataset_info` | filter values (pathology, study type, contrast phase), windows, presets |
| `find_slices` | rank (case, view, slice) candidates showing given structures |
| `slice_contents` | structures visible on a slice, for picking distractors |
| `render_image` | render a specific slice with labels and optional hotspot masks |
| `labeled_image` | find + render in one call |
| `batch_labeled_images` | many requests into a folder with `manifest.jsonl` |

Every image comes back with a file path (and base64 if requested) and a key: label → structures,
arrow tail/tip and label-box coordinates, outline polygons, and per-structure area. Hotspot masks are
transparent PNGs pixel-aligned with the image.

## Data

Datasets are **not** in this repo. Download them yourself and place them in the project root
(or point `MEDIMGQ_TS_ROOT` at them):

- [TotalSegmentator dataset v2.0.1](https://zenodo.org/records/10047292) → `Totalsegmentator_dataset_v201/` (CC BY 4.0)
- [AMOS22](https://amos22.grand-challenge.org/) → `amos22/` (CC BY-SA 4.0; optional, future MRI source)

Every generated image carries the dataset attribution in a footer.

## Setup

```bash
uv sync
uv run python -m medimgq.indexer --workers 12   # one-time (~4 min), structure/slice index
uv run python -m medimgq.quality --workers 12   # one-time, noise / contrast phase / FOV index
uv run python scripts/compare_strategies.py     # contact sheet comparing slice strategies
```

## Layout

- `src/medimgq/catalog.py` — canonical structure names, display names, synonyms, groups, name resolution
- `src/medimgq/indexer.py` — builds the per-case / per-structure / per-slice index used for slice search
- `src/medimgq/quality.py` — per-case noise and contrast-phase estimate; per-slice body truncation and arms-in-field
- `src/medimgq/volume.py` — CT/mask loading, radiological slice extraction, window presets
- `src/medimgq/selection.py` — slice-selection strategies (classical, max_visibility, max_area, random, level, index) and filters
- `src/medimgq/render.py` — label rendering (margin arrows, short arrows, on-structure, outlines; presets) and key geometry
- `scripts/compare_strategies.py` — renders a strategy-comparison contact sheet
- `scripts/showcase_labels.py` — renders the same scenes with every label preset
- `qc/blocklist.csv` — cases excluded after human review
- `src/medimgq/server.py` — MCP server (stdio)
- `src/medimgq/config.py` — paths (override with `MEDIMGQ_TS_ROOT`, `MEDIMGQ_INDEX_DIR`, `MEDIMGQ_OUTPUT_DIR`)

## Intended use

Educational use only. Not for clinical decision-making.
