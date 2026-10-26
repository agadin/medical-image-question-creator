# medimgq — medical image question creator

An MCP server that another AI can call to generate **labeled medical images** for
medical-school practice exams, together with a machine-readable **answer key**.
CT first (TotalSegmentator, 117 structures); X-ray and MRI later.

> Status: early development. Catalog, dataset/quality index, slice selection, and label rendering work;
> the question generator and MCP server are in progress.

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
- `src/medimgq/config.py` — paths (override with `MEDIMGQ_TS_ROOT`, `MEDIMGQ_INDEX_DIR`, `MEDIMGQ_OUTPUT_DIR`)

## Intended use

Educational use only. Not for clinical decision-making.
