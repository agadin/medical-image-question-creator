"""Unit tests on synthetic volumes: no dataset or index needed."""

from __future__ import annotations

import numpy as np
import pytest

from medimgq import findings, render as rmod
from medimgq.catalog import UnknownStructureError, display, resolve
from medimgq.render import LabelSpec, RenderOptions, _anchor, auto_region, auto_window
from medimgq.selection import _level_window, classical_view
from medimgq.volume import apply_window, take_slice

# ---------------------------------------------------------------- catalog


@pytest.mark.parametrize("query, expected", [
    ("IVC", ("inferior_vena_cava",)),
    ("postcava", ("inferior_vena_cava",)),
    ("arota", ("aorta",)),  # AMOS's spelling
    ("left kidney", ("kidney_left",)),
    ("kidney", ("kidney_left", "kidney_right")),  # unsided = both
    ("common carotid arteries", ("common_carotid_artery_left", "common_carotid_artery_right")),  # plural
    ("L1", ("vertebrae_L1",)),
    ("left 7th rib", ("rib_left_7",)),
    ("RML", ("lung_middle_lobe_right",)),
    ("The Pancreas", ("pancreas",)),  # case and articles ignored
])
def test_resolve(query, expected):
    assert resolve(query) == expected


def test_resolve_groups():
    assert len(resolve("lumbar spine")) == 5
    assert len(resolve("ribs")) == 24


def test_unknown_structure_suggests():
    with pytest.raises(UnknownStructureError) as e:
        resolve("pancraes")
    assert "pancreas" in e.value.suggestions


def test_display_names_groups():
    assert display(("kidney_left", "kidney_right")) == "kidneys"
    assert display(("pancreas",)) == "pancreas"


# ---------------------------------------------------------------- volume


def test_take_slice_radiological_orientation():
    # RAS volume: +x = patient right, +y = anterior, +z = superior.
    vol = np.zeros((40, 30, 20), np.int16)
    vol[35, 15, 10] = 1  # patient right
    vol[20, 27, 10] = 2  # anterior
    ax = take_slice(vol, "axial", 10)
    r, c = np.argwhere(ax == 1)[0]
    assert c < ax.shape[1] / 2  # patient right -> image left
    r, c = np.argwhere(ax == 2)[0]
    assert r < ax.shape[0] / 2  # anterior -> image top

    vol[:] = 0
    vol[20, 15, 18] = 3  # superior
    cor = take_slice(vol, "coronal", 15)
    assert np.argwhere(cor == 3)[0][0] < cor.shape[0] / 2  # superior -> image top


def test_apply_window_and_fov_clamp():
    hu = np.array([[-3024, -1024, 40, 2000]], np.int16)
    soft = apply_window(hu, "soft_tissue")
    assert soft[0, 2] == 127 or soft[0, 2] == 128  # level maps to mid-gray
    assert soft[0, 3] == 255
    lung = apply_window(hu, "lung")
    assert lung[0, 0] == lung[0, 1]  # outside-FOV pixels look like air


# ---------------------------------------------------------------- anchors and components


def _two_blobs():
    m = np.zeros((60, 60), bool)
    m[5:20, 25:35] = True  # top (anterior on axial), smaller
    m[35:58, 20:45] = True  # bottom (posterior), larger
    return m


def test_anchor_is_inside_mask():
    m = _two_blobs()
    x, y = _anchor(m)
    assert m[int(y), int(x)]


def test_anchor_components():
    m = _two_blobs()
    assert _anchor(m, "largest")[1] > 30  # bigger blob is lower
    assert _anchor(m, "anterior", "axial")[1] < 20
    assert _anchor(m, "posterior", "axial")[1] > 30
    left = np.zeros((40, 60), bool)
    left[10:30, 2:12] = True  # image left = patient right
    left[10:30, 45:58] = True
    assert _anchor(left, "right", "axial")[0] < 20
    assert _anchor(left, "left", "axial")[0] > 40


def test_anchor_component_invalid_for_view():
    with pytest.raises(ValueError, match="not visible"):
        _anchor(_two_blobs(), "superior", "axial")


def test_anchor_ignores_specks():
    m = _two_blobs()
    m[1, 1] = True  # a speck far anterior
    assert _anchor(m, "anterior", "axial")[1] > 4  # still the real anterior blob


# ---------------------------------------------------------------- auto window / region / level windows


def test_auto_window():
    assert auto_window(["heart", "lung_upper_lobe_left"]) == "lung"
    assert auto_window(["vertebrae_L1", "vertebrae_L2", "aorta"]) == "bone"
    assert auto_window(["pancreas", "aorta"]) == "soft_tissue"


def test_auto_region():
    assert auto_region("axial", "C6", ["trachea"], 0) == "neck"
    assert auto_region("axial", "T6", ["heart"], 5000) == "chest"
    assert auto_region("axial", "L1", ["pancreas"], 0) == "abdomen"
    assert auto_region("axial", "T1", ["thyroid_gland", "common_carotid_artery_left"], 5000) == "neck"


def test_level_windows_and_views():
    lo, hi = _level_window(["pancreas", "kidney_left"])  # T12-L2 and T12-L2
    assert lo <= hi
    assert classical_view(["lung_upper_lobe_left", "lung_lower_lobe_left", "heart"]) == "coronal"
    assert classical_view(["pancreas"]) == "axial"


# ---------------------------------------------------------------- full render on a synthetic case


@pytest.fixture
def synthetic_case(monkeypatch):
    shape = (90, 90, 12)
    xx, yy = np.meshgrid(np.arange(90), np.arange(90), indexing="ij")
    body = (xx - 45) ** 2 + (yy - 45) ** 2 < 38**2
    ct = np.full(shape, -1024, np.int16)
    ct[body] = 40
    masks = {
        "liver": np.repeat(((xx - 62) ** 2 + (yy - 45) ** 2 < 12**2)[..., None], 12, 2),
        "spleen": np.repeat(((xx - 28) ** 2 + (yy - 40) ** 2 < 7**2)[..., None], 12, 2),
        "kidney_left": np.repeat(((xx - 30) ** 2 + (yy - 25) ** 2 < 6**2)[..., None], 12, 2),
        "kidney_right": np.repeat(((xx - 60) ** 2 + (yy - 25) ** 2 < 6**2)[..., None], 12, 2),
    }
    monkeypatch.setattr(rmod, "load_ct", lambda cid: ct)
    empty = np.zeros(shape, bool)  # anything else (e.g. lungs used by auto crop) is absent
    monkeypatch.setattr(rmod, "load_mask", lambda cid, n: masks.get(n, empty))
    return masks


def _tip_inside(r, name, tip):
    m = r.mask_image([name])
    return np.asarray(m)[int(tip[1]), int(tip[0]), 3] > 0


def test_render_arrows_land_inside_structures(synthetic_case):
    labels = [LabelSpec("1", ["liver"]), LabelSpec("2", ["spleen"])]
    r = rmod.render("synthetic", "axial", 6, labels, RenderOptions(size=400, crop="full"))
    for k, name in zip(r.key, ["liver", "spleen"]):
        assert len(k.arrows_px) == 1
        assert _tip_inside(r, name, k.arrows_px[0][1])
        assert k.area_mm2[name] > 0


def test_render_shared_label_modes(synthetic_case):
    kid = [LabelSpec("1", ["kidney_left", "kidney_right"])]
    multi = rmod.render("synthetic", "axial", 6, kid, RenderOptions(size=400, shared_mode="multi_arrow"))
    rep = rmod.render("synthetic", "axial", 6, kid, RenderOptions(size=400, shared_mode="repeat"))
    assert len(multi.key[0].arrows_px) == 2 and len(rep.key[0].arrows_px) == 2
    for tail_tip, name in zip(multi.key[0].arrows_px, ["kidney_left", "kidney_right"]):
        assert any(_tip_inside(multi, n, tail_tip[1]) for n in ("kidney_left", "kidney_right"))


def test_render_edge_tip_is_near_boundary(synthetic_case):
    labels = [LabelSpec("1", ["liver"])]
    center = rmod.render("synthetic", "axial", 6, labels, RenderOptions(size=400, crop="full", tip="center"))
    edge = rmod.render("synthetic", "axial", 6, labels, RenderOptions(size=400, crop="full", tip="edge"))
    cx, cy = center.key[0].anchors_px[0]
    ex, ey = edge.key[0].arrows_px[0][1]
    assert _tip_inside(edge, "liver", (ex, ey))
    assert np.hypot(ex - cx, ey - cy) > 10  # tip moved from the center toward the edge


def test_hotspot_mask_is_pixel_aligned(synthetic_case):
    r = rmod.render("synthetic", "axial", 6, [], RenderOptions(size=400, marker="none"), context=["liver"])
    m = np.asarray(r.mask_image(["liver"]))
    assert m.shape[:2] == (r.image.height, r.image.width)
    img = np.asarray(r.image.convert("L")).astype(int)
    inside, outside = img[m[..., 3] > 0], img[(m[..., 3] == 0)]
    assert inside.size > 0 and inside.mean() > 0  # the mask covers body, not background


def test_unlabeled_arrow_has_no_text_box(synthetic_case):
    r = rmod.render("synthetic", "axial", 6, [LabelSpec("", ["liver"])], RenderOptions(size=400))
    assert r.key[0].label == "" and len(r.key[0].arrows_px) == 1


def test_component_option_reaches_render(synthetic_case, monkeypatch):
    two = np.zeros((90, 90, 12), bool)
    two[40:50, 70:80, :] = True  # anterior (high y)
    two[40:50, 10:25, :] = True  # posterior, larger
    synthetic_case["liver"] = two
    ant = rmod.render("synthetic", "axial", 6, [LabelSpec("1", ["liver"], "anterior")], RenderOptions(size=300))
    post = rmod.render("synthetic", "axial", 6, [LabelSpec("1", ["liver"], "posterior")], RenderOptions(size=300))
    assert ant.key[0].anchors_px[0][1] < post.key[0].anchors_px[0][1]


# ---------------------------------------------------------------- server helpers


def test_shuffle_labels_permutes_and_is_reproducible():
    from medimgq.server import Label, _shuffle_labels

    labels = [Label(label=str(i + 1), structures=[s]) for i, s in enumerate(["liver", "spleen", "aorta", "IVC"])]
    labels.append(Label(label="", structures=["pancreas"]))
    a = _shuffle_labels(labels, seed=3)
    b = _shuffle_labels(labels, seed=3)
    assert [(l.label, l.structures) for l in a] == [(l.label, l.structures) for l in b]
    assert sorted(l.label for l in a if l.label) == ["1", "2", "3", "4"]
    assert [l for l in a if not l.label][0].structures == ["pancreas"]
    seen = {tuple(l.structures[0] for l in _shuffle_labels(labels, seed=s) if l.label == "1") for s in range(20)}
    assert len(seen) > 1  # the structure numbered "1" varies


# ---------------------------------------------------------------- findings


def test_aorta_diameter_measures_a_cylinder(monkeypatch):
    m = np.zeros((60, 60, 5), bool)
    xx, yy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    m[(xx - 30) ** 2 + (yy - 30) ** 2 < 10**2] = True  # radius 10 voxels = 15 mm -> 30 mm diameter
    monkeypatch.setattr(findings, "load_mask", lambda cid, n: m)
    rows = findings._aorta("x")
    assert len(rows) == 5
    assert all(27 <= r["value"] <= 31 for r in rows)


def test_findings_catalog_is_consistent():
    from medimgq.catalog import STRUCTURES

    for f in findings.FINDINGS.values():
        assert f.threshold > 0 and f.unit
        assert all(s in STRUCTURES for s in f.structures)
