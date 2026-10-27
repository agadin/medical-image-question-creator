# ai_read_this.md: guide for AI agents using medimgq

You are connected to **medimgq**, an MCP server that returns real CT images with numbered labels, plus an
answer key and optional click-region masks. **You** write the exam question. medimgq gives you the image
and the facts about it. Humans should read `README.md` instead.

## Core facts

- Data: TotalSegmentator v2.0.1. 1,228 CT scans, 1.5 mm isotropic, 117 segmented structures.
- Orientation: radiological. The patient's right is on the image's left; anterior is at the top of axial
  images; superior is at the top of coronal and sagittal images.
- Every image has an attribution footer. Keep it.
- Labels are drawn exactly as you give them. Use **numbers** (`"1"`…`"9"`) so they never collide with the
  exam's A–E answer letters.
- You can only label segmented structures. If a name fails, the error lists close matches; call
  `list_structures` to see everything.

## Tools

| Tool | Required args | Use it to |
|---|---|---|
| `list_structures` | none (`query`, `region`, `system` optional) | resolve a name, or list valid names, synonyms and groups |
| `dataset_info` | none | get filter values, window names and presets |
| `find_slices` | `structures: string[]` | rank candidate (case_id, view, slice) where all structures are visible |
| `slice_contents` | `case_id`, `view`, `slice_index` | list everything visible on a slice (area, visibility), to pick distractors |
| `render_image` | `case_id`, `view`, `slice_index`, `labels` | render a slice you chose |
| `labeled_image` | `labels` | find + render in one call (most common) |
| `batch_labeled_images` | `requests: BatchRequest[]` | many images into one folder + `manifest.jsonl` (no base64) |

### Argument shapes

```text
Label        = { label: string, structures: string[] }      # label "" = arrow with no text
HotspotZone  = { id: string, structures: string[] }         # becomes <image>_mask_<id>.png
CaseFilters  = { case_id?, pathology?: string[], pathology_location?: string[], study_type?: string[],
                 gender?: "m"|"f", age_min?, age_max?, phase?: string[], exclude_cases?: string[],
                 min_visibility?=0.4, max_noise_hu?=45, allow_edge_truncated?=false,
                 allow_body_truncated?=false, allow_arms_in_fov?=false }
RenderSettings = { preset?="arrow_short", marker?, tip?: "center"|"edge", outline?: bool,
                   shared_mode?: "multi_arrow"|"repeat", text_style?: "halo"|"boxed"|"boxed_dark",
                   marker_color?: [r,g,b], text_color?: [r,g,b], font_size?, size?=768,
                   crop?: "auto"|"neck"|"chest"|"abdomen"|"body"|"targets"|"full",
                   window?: "auto"|preset|[width, level], format?: "png"|"webp"|"jpeg" }

labeled_image(labels, view?="auto", strategy?="classical", level?, filters?: CaseFilters,
              render?: RenderSettings, hotspot_zones?: HotspotZone[], context_structures?: string[],
              top_k?=10, seed?, output_dir?, name?, include_base64?=true)
render_image(case_id, view, slice_index, labels, render?, hotspot_zones?, output_dir?, name?, include_base64?=true)
find_slices(structures, view?="auto", strategy?="classical", level?, slice_index?, filters?, limit?=10, seed?)
batch_labeled_images(requests: [{labels, view?, strategy?, level?, filters?, render?, hotspot_zones?,
                     context_structures?, seed?, name?}], output_dir?, avoid_repeat_cases?=true)
```

Enumerations:

- `view`: `auto` | `axial` | `coronal` | `sagittal`
- `strategy`: `classical` (textbook view and level) | `max_visibility` | `max_area` | `random` | `level`
  (needs `level`, axial) | `index` (needs `slice_index` and `filters.case_id`)
- `preset`: `arrow_short`, `arrow_margin`, `on_structure`, `outline_labeled`, `outline_arrows`,
  `outline_only`, `boxed_margin`, `yellow_short`, `edge_arrows`, `bare`
- `window` presets: `soft_tissue`, `abdomen`, `liver`, `mediastinum`, `lung`, `bone`, `brain`, `angio`
- `pathology`: `no_pathology`, `tumor`, `vascular`, `trauma`, `inflammation`, `bleeding`, `other`, `unclear`
- `phase`: `non_contrast`, `arterial`, `venous`, `contrast_unknown_phase` (a heuristic estimate)

### Names you can use

The canonical names are TotalSegmentator's (`kidney_left`, `vertebrae_L1`, `rib_right_7`). You can also
use:

- display names: "left kidney"
- synonyms: "IVC", "LUL", "RML", "L1", "atlas", "psoas"
- unsided names, which mean both sides: "kidney", "adrenal gland", "carotids"
- plurals: "common carotid arteries"
- groups: "lungs", "left lung", "ribs", "lumbar spine", "kidneys", "psoas muscles"

A group in one `Label` gives a single shared number.

## Response

```text
image_path, image_size [w, h], image_base64?, mime_type?
answer_key   { "<label>": "<display names, comma-joined>" }       # "(arrow N)" for "" labels
key[]        { label, structures[], display_names[], anchors_px[[x,y]], label_box_px [x0,y0,x1,y1],
               arrows_px[[[tail_x,tail_y],[tip_x,tip_y]]], outlines_px[[[x,y]...]], area_mm2{name: mm2} }
hotspot_masks[] { id, structures[], path }   # RGBA PNG, same size as the image, alpha>0 = inside
source       { dataset, case_id, view, slice, vertebral_level, window, crop_region, crop_box_slice_px,
               case{age, sex, study_type, pathology, pathology_location, contrast_phase_estimate},
               attribution }
selection    { strategy, score, visibility{name: 0..1}, areas_mm2{name: mm2}, candidates_considered }
```

## Recipes

**Labeled multiple-choice question (most common)**

1. Decide the answer structure and 3–4 distractors that plausibly sit near it on the same slice.
2. Shuffle the numbers yourself so the answer isn't always "1".
3. Call `labeled_image` with `labels=[{label:"1", structures:[...]}, ...]`.
4. Look at the returned image, if your client can show it, and confirm each arrow is unambiguous.
5. Use `answer_key` for the correct choice. Write the stem so the student must identify the structure and
   then reason about it (second-order). The choices are the numbers; never name the structures in them.

**Choosing distractors that are definitely on the slice**

`find_slices([answer])`, then `slice_contents(case_id, view, slice)`, pick from the returned names, then
`render_image(...)`.

**Click-the-structure (hotspot) question**

`labeled_image(labels=[], render={preset:"bare"}, hotspot_zones=[{id:"answer", structures:[X]},
{id:"<wrong1>", structures:[Y]}, ...])`. The zone structures also drive slice search, crop and window.
Add the likely wrong clicks as zones so feedback can name what was clicked. Avoid answers smaller than
about 150 mm² (check `slice_contents`), since they are hard to click.

**Arrow-only question** ("What is indicated by the arrow?"): `labels=[{label:"", structures:[X]}]`.

**A specific vertebral level:** `strategy:"level", level:"T8"` (axial only).

**Many images:** `batch_labeled_images`. It uses a different case per image (`avoid_repeat_cases`) and
writes `manifest.jsonl`.

**Reproducibility:** pass `seed`. Without it, `labeled_image` picks randomly among the `top_k` best cases.

## Rules and pitfalls

- If you get **"No slice shows all requested structures"**: drop a structure, try another `view`, use
  `strategy:"max_visibility"`, lower `filters.min_visibility`, or set `allow_*` flags. Structures far apart
  (e.g. thyroid and kidney) can never share an axial slice.
- **Some names are unknown** (SMA, azygos, internal jugular vein, nerves). Pick something else.
- **A structure that appears twice on a slice gets its arrow on the larger piece.** The aorta can show its
  root and its descending part on one axial slice. Check `anchors_px` or the image, or choose a level where
  it appears once.
- **The window switches to lung whenever a lung lobe is labeled.** Pass `render.window:"soft_tissue"` to
  override.
- **`source.case.pathology` describes the scan, not your structure.** Don't claim a lesion unless you can see
  it. Most images are normal; say "representative image" in stems that describe disease on a normal scan.
- **Real disease can be found by measuring**, e.g. aortic aneurysm in `s1400` or `s1365`, and retrosternal
  goiter in `s0522`. Use `render_image` with the case and slice.
- **Stated patient ages and sexes are your fiction** unless you copy `source.case`.
- **Keep the attribution footer.** It is required by CC BY 4.0.
- **The quality filters are on by default.** Leave them on unless a request fails.
