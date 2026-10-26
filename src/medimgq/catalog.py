"""Structure catalog: canonical names, display names, synonyms, and groups.

Canonical names are the TotalSegmentator v2 mask names (e.g. ``kidney_left``).
Everything user-facing (synonym lookup, laterality, groups) resolves to these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Structure:
    name: str  # canonical TotalSegmentator name
    display: str  # human-readable name used in answer keys
    system: str  # e.g. "digestive", "skeletal"
    region: str  # e.g. "abdomen", "thorax"
    side: str | None = None  # "left" | "right" | None
    synonyms: tuple[str, ...] = field(default_factory=tuple)


# name -> (display, system, region, synonyms)
_BASE: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    # Abdominal / pelvic organs
    "spleen": ("spleen", "lymphatic", "abdomen", ()),
    "liver": ("liver", "digestive", "abdomen", ("hepar",)),
    "gallbladder": ("gallbladder", "digestive", "abdomen", ("gall bladder", "GB")),
    "stomach": ("stomach", "digestive", "abdomen", ("gaster",)),
    "pancreas": ("pancreas", "digestive", "abdomen", ()),
    "duodenum": ("duodenum", "digestive", "abdomen", ()),
    "small_bowel": ("small bowel", "digestive", "abdomen", ("small intestine", "jejunum", "ileum")),
    "colon": ("colon", "digestive", "abdomen", ("large bowel", "large intestine")),
    "esophagus": ("esophagus", "digestive", "thorax", ("oesophagus",)),
    "kidney_left": ("left kidney", "urinary", "abdomen", ()),
    "kidney_right": ("right kidney", "urinary", "abdomen", ()),
    "kidney_cyst_left": ("left renal cyst", "urinary", "abdomen", ("left kidney cyst",)),
    "kidney_cyst_right": ("right renal cyst", "urinary", "abdomen", ("right kidney cyst",)),
    "adrenal_gland_left": ("left adrenal gland", "endocrine", "abdomen", ("left suprarenal gland", "left adrenal")),
    "adrenal_gland_right": ("right adrenal gland", "endocrine", "abdomen", ("right suprarenal gland", "right adrenal")),
    "urinary_bladder": ("urinary bladder", "urinary", "pelvis", ("bladder",)),
    "prostate": ("prostate", "reproductive", "pelvis", ("prostate gland",)),
    # Thorax / neck
    "heart": ("heart", "cardiovascular", "thorax", ()),
    "atrial_appendage_left": ("left atrial appendage", "cardiovascular", "thorax", ("LAA",)),
    "lung_upper_lobe_left": ("left upper lobe of lung", "respiratory", "thorax", ("LUL", "left upper lobe")),
    "lung_lower_lobe_left": ("left lower lobe of lung", "respiratory", "thorax", ("LLL", "left lower lobe")),
    "lung_upper_lobe_right": ("right upper lobe of lung", "respiratory", "thorax", ("RUL", "right upper lobe")),
    "lung_middle_lobe_right": ("right middle lobe of lung", "respiratory", "thorax", ("RML", "right middle lobe", "middle lobe")),
    "lung_lower_lobe_right": ("right lower lobe of lung", "respiratory", "thorax", ("RLL", "right lower lobe")),
    "trachea": ("trachea", "respiratory", "thorax", ("windpipe",)),
    "thyroid_gland": ("thyroid gland", "endocrine", "neck", ("thyroid",)),
    "spinal_cord": ("spinal cord", "nervous", "spine", ("cord",)),
    "brain": ("brain", "nervous", "head", ()),
    # Vessels
    "aorta": ("aorta", "cardiovascular", "thorax", ("arota",)),
    "inferior_vena_cava": ("inferior vena cava", "cardiovascular", "abdomen", ("IVC", "postcava")),
    "superior_vena_cava": ("superior vena cava", "cardiovascular", "thorax", ("SVC",)),
    "portal_vein_and_splenic_vein": ("portal and splenic veins", "cardiovascular", "abdomen", ("portal vein", "splenic vein", "portosplenic confluence")),
    "pulmonary_vein": ("pulmonary veins", "cardiovascular", "thorax", ("pulmonary vein",)),
    "brachiocephalic_trunk": ("brachiocephalic trunk", "cardiovascular", "thorax", ("innominate artery", "brachiocephalic artery")),
    "brachiocephalic_vein_left": ("left brachiocephalic vein", "cardiovascular", "thorax", ("left innominate vein",)),
    "brachiocephalic_vein_right": ("right brachiocephalic vein", "cardiovascular", "thorax", ("right innominate vein",)),
    "common_carotid_artery_left": ("left common carotid artery", "cardiovascular", "neck", ("left CCA", "left carotid")),
    "common_carotid_artery_right": ("right common carotid artery", "cardiovascular", "neck", ("right CCA", "right carotid")),
    "subclavian_artery_left": ("left subclavian artery", "cardiovascular", "thorax", ()),
    "subclavian_artery_right": ("right subclavian artery", "cardiovascular", "thorax", ()),
    "iliac_artery_left": ("left common iliac artery", "cardiovascular", "pelvis", ("left iliac artery",)),
    "iliac_artery_right": ("right common iliac artery", "cardiovascular", "pelvis", ("right iliac artery",)),
    "iliac_vena_left": ("left common iliac vein", "cardiovascular", "pelvis", ("left iliac vein",)),
    "iliac_vena_right": ("right common iliac vein", "cardiovascular", "pelvis", ("right iliac vein",)),
    # Muscles
    "autochthon_left": ("left erector spinae (autochthonous back muscles)", "muscular", "spine", ("left erector spinae", "left paraspinal muscles")),
    "autochthon_right": ("right erector spinae (autochthonous back muscles)", "muscular", "spine", ("right erector spinae", "right paraspinal muscles")),
    "iliopsoas_left": ("left iliopsoas", "muscular", "pelvis", ("left psoas", "left psoas major")),
    "iliopsoas_right": ("right iliopsoas", "muscular", "pelvis", ("right psoas", "right psoas major")),
    "gluteus_maximus_left": ("left gluteus maximus", "muscular", "pelvis", ()),
    "gluteus_maximus_right": ("right gluteus maximus", "muscular", "pelvis", ()),
    "gluteus_medius_left": ("left gluteus medius", "muscular", "pelvis", ()),
    "gluteus_medius_right": ("right gluteus medius", "muscular", "pelvis", ()),
    "gluteus_minimus_left": ("left gluteus minimus", "muscular", "pelvis", ()),
    "gluteus_minimus_right": ("right gluteus minimus", "muscular", "pelvis", ()),
    # Bones (non-serial)
    "skull": ("skull", "skeletal", "head", ("cranium",)),
    "sternum": ("sternum", "skeletal", "thorax", ("breastbone",)),
    "costal_cartilages": ("costal cartilages", "skeletal", "thorax", ("costal cartilage",)),
    "sacrum": ("sacrum", "skeletal", "pelvis", ()),
    "clavicula_left": ("left clavicle", "skeletal", "thorax", ("left collarbone",)),
    "clavicula_right": ("right clavicle", "skeletal", "thorax", ("right collarbone",)),
    "scapula_left": ("left scapula", "skeletal", "thorax", ("left shoulder blade",)),
    "scapula_right": ("right scapula", "skeletal", "thorax", ("right shoulder blade",)),
    "humerus_left": ("left humerus", "skeletal", "upper_limb", ()),
    "humerus_right": ("right humerus", "skeletal", "upper_limb", ()),
    "hip_left": ("left hip bone", "skeletal", "pelvis", ("left os coxae", "left innominate bone", "left ilium")),
    "hip_right": ("right hip bone", "skeletal", "pelvis", ("right os coxae", "right innominate bone", "right ilium")),
    "femur_left": ("left femur", "skeletal", "lower_limb", ()),
    "femur_right": ("right femur", "skeletal", "lower_limb", ()),
}

VERTEBRAE: tuple[str, ...] = (
    *(f"C{i}" for i in range(1, 8)),
    *(f"T{i}" for i in range(1, 13)),
    *(f"L{i}" for i in range(1, 6)),
    "S1",
)

_VERT_REGION = {"C": "neck", "T": "thorax", "L": "abdomen", "S": "pelvis"}
_VERT_WORD = {"C": "cervical", "T": "thoracic", "L": "lumbar", "S": "sacral"}
_VERT_SPECIAL = {"C1": ("atlas",), "C2": ("axis",)}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _build() -> dict[str, Structure]:
    out: dict[str, Structure] = {}
    for name, (display, system, region, syn) in _BASE.items():
        side = "left" if name.endswith("_left") else "right" if name.endswith("_right") else None
        out[name] = Structure(name, display, system, region, side, syn)
    for v in VERTEBRAE:
        word = _VERT_WORD[v[0]]
        num = int(v[1:])
        syn = (v, f"{v} vertebra", f"{v} vertebral body", f"{_ordinal(num)} {word} vertebra") + _VERT_SPECIAL.get(v, ())
        out[f"vertebrae_{v}"] = Structure(f"vertebrae_{v}", f"{v} vertebra", "skeletal", _VERT_REGION[v[0]], None, syn)
    for side in ("left", "right"):
        for i in range(1, 13):
            syn = (f"{side} rib {i}", f"{side} {_ordinal(i)} rib", f"{side[0].upper()}{i} rib")
            out[f"rib_{side}_{i}"] = Structure(f"rib_{side}_{i}", f"{side} {_ordinal(i)} rib", "skeletal", "thorax", side, syn)
    return out


STRUCTURES: dict[str, Structure] = _build()

# Named groups: one label can point at several structures (e.g. "kidneys").
GROUPS: dict[str, tuple[str, ...]] = {
    "kidneys": ("kidney_left", "kidney_right"),
    "adrenal glands": ("adrenal_gland_left", "adrenal_gland_right"),
    "lungs": tuple(n for n in STRUCTURES if n.startswith("lung_")),
    "left lung": ("lung_upper_lobe_left", "lung_lower_lobe_left"),
    "right lung": ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"),
    "common iliac arteries": ("iliac_artery_left", "iliac_artery_right"),
    "common iliac veins": ("iliac_vena_left", "iliac_vena_right"),
    "psoas muscles": ("iliopsoas_left", "iliopsoas_right"),
    "erector spinae": ("autochthon_left", "autochthon_right"),
    "left ribs": tuple(f"rib_left_{i}" for i in range(1, 13)),
    "right ribs": tuple(f"rib_right_{i}" for i in range(1, 13)),
    "ribs": tuple(f"rib_{s}_{i}" for s in ("left", "right") for i in range(1, 13)),
    "cervical spine": tuple(f"vertebrae_{v}" for v in VERTEBRAE if v[0] == "C"),
    "thoracic spine": tuple(f"vertebrae_{v}" for v in VERTEBRAE if v[0] == "T"),
    "lumbar spine": tuple(f"vertebrae_{v}" for v in VERTEBRAE if v[0] == "L"),
}


def _norm(s: str) -> str:
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\bthe\b|\bof\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_lookup() -> dict[str, tuple[str, ...]]:
    lookup: dict[str, tuple[str, ...]] = {}
    for s in STRUCTURES.values():
        for key in (s.name, s.display, *s.synonyms):
            lookup[_norm(key)] = (s.name,)
    for g, members in GROUPS.items():
        lookup.setdefault(_norm(g), members)
    # Unsided names ("kidney", "adrenal gland") resolve to both sides.
    for s in STRUCTURES.values():
        if s.side and not s.name.startswith("rib_"):
            for key in (s.display, *s.synonyms):
                n = _norm(key)
                if n.startswith(s.side + " "):
                    bare = n[len(s.side) + 1 :]
                    other = s.name.replace(f"_{s.side}", "_right" if s.side == "left" else "_left")
                    if other in STRUCTURES:
                        lookup.setdefault(bare, tuple(sorted((s.name, other))))
    return lookup


_LOOKUP = _build_lookup()


class UnknownStructureError(ValueError):
    def __init__(self, query: str, suggestions: list[str]):
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"Unknown structure {query!r}. Did you mean: {', '.join(suggestions) or 'n/a'}? "
                         "Call list_structures for the full list.")


def resolve(query: str) -> tuple[str, ...]:
    """Resolve a free-text name (synonym, group, or unsided name) to canonical structure names."""
    key = _norm(query)
    if key in _LOOKUP:
        return _LOOKUP[key]
    import difflib

    close = difflib.get_close_matches(key, _LOOKUP.keys(), n=5, cutoff=0.6)
    raise UnknownStructureError(query, close)


def display(names: tuple[str, ...] | list[str]) -> str:
    """Human-readable name for one structure or a group of them."""
    names = tuple(sorted(names))
    for g, members in GROUPS.items():
        if tuple(sorted(members)) == names:
            return g
    if len(names) == 2:
        a, b = (STRUCTURES[n] for n in names)
        if a.side and b.side and a.display.split(" ", 1)[1] == b.display.split(" ", 1)[1]:
            return "bilateral " + a.display.split(" ", 1)[1] + "s"
    return ", ".join(STRUCTURES[n].display for n in names)
