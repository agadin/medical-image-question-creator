"""Example MCP client: the image requests behind the exam-sim CT anatomy bank.

Usage: uv run python examples/examsim_ct_bank.py OUT_DIR [question ids...]
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

OUT = Path(sys.argv[1])
ONLY = set(sys.argv[2:])

L = lambda label, *s: {"label": label, "structures": list(s)}  # noqa: E731

REQUESTS = {
    # card 66/141: superior mediastinum, catheter via left IJV
    "ct1": dict(answer=('left brachiocephalic vein', "1"), labels=[L("A", "left brachiocephalic vein"), L("B", "right brachiocephalic vein"),
                        L("C", "brachiocephalic trunk"), L("D", "trachea"), L("E", "esophagus")]),
    # card 63: aortic arch aneurysm -> hoarseness
    "ct2": dict(labels=[L("", "aorta")], view="axial", strategy="max_area",
                context_structures=["trachea"]),
    # card 26: enlarged left atrium compresses esophagus
    "ct3": dict(answer=('esophagus', "2"), labels=[L("A", "heart"), L("B", "esophagus"), {"label": "C", "structures": ["aorta"], "component": "posterior"},
                        L("D", "spinal cord")],
                level="T8", strategy="level"),
    # card 46/164: diaphragmatic openings
    "ct4": dict(answer=('aorta', "3"), labels=[L("A", "inferior vena cava"), L("B", "esophagus"), L("C", "aorta"),
                        L("D", "liver"), L("E", "stomach")]),
    # card 135/72: silhouette of left heart border -> lingula (LUL)
    "ct5": dict(answer=('left upper lobe', "5"), labels=[L("A", "left upper lobe"), L("B", "left lower lobe"), L("C", "right middle lobe"),
                        L("D", "right lower lobe"), L("E", "heart")]),
    # card 49: penetrating injury right 6th ICS midaxillary line
    "ct6": dict(answer=('right lower lobe', "2"), labels=[L("A", "right upper lobe"), L("B", "right middle lobe"), L("C", "right lower lobe"),
                        L("D", "left upper lobe"), L("E", "left lower lobe")], view="coronal"),
    # card 110: recurrent laryngeal nerve in the tracheoesophageal groove
    "ct7": dict(answer=('common carotid arteries', "4"), labels=[L("A", "thyroid"), L("B", "trachea"), L("C", "esophagus"),
                        L("D", "common carotid arteries")], render={"shared_mode": "repeat"},
                filters={"max_noise_hu": 30}),
    # card 140/17: right subclavian artery and the right recurrent laryngeal nerve
    "ct8": dict(answer=('right subclavian artery', "3"), labels=[L("A", "right subclavian artery"), L("B", "right common carotid artery"),
                        L("C", "left common carotid artery"), L("D", "left subclavian artery"), L("E", "trachea")],
                filters={"max_noise_hu": 30}, strategy="level", level="T2"),
    # disease (real pathology; fixed case/slice via render_image)
    "d1": dict(answer=('esophagus', "4"), tool="render_image", case_id="s1400", view="axial", slice_index=315,
               labels=[L("A", "aorta"), L("B", "esophagus"), L("C", "heart"), L("D", "left lower lobe"),
                       L("E", "spinal cord")], render={"window": "soft_tissue"}),
    "d2": dict(answer=('trachea', "5"), tool="render_image", case_id="s0522", view="axial", slice_index=173,
               labels=[L("A", "thyroid"), L("B", "trachea"), L("C", "esophagus"),
                       L("D", "left common carotid artery"), L("E", "sternum")], render={"crop": "neck"}),
    "d3": dict(tool="render_image", case_id="s1365", view="axial", slice_index=313, labels=[],
               render={"preset": "bare", "window": "soft_tissue"}, hotspot_zones=[
                   {"id": "esophagus", "structures": ["esophagus"]},
                   {"id": "aorta", "structures": ["aorta"]},
                   {"id": "heart", "structures": ["heart"]},
                   {"id": "cord", "structures": ["spinal cord"]}]),
    # hotspots (bare images)
    "h1": dict(labels=[], render={"preset": "bare"}, hotspot_zones=[
        {"id": "lbcv", "structures": ["left brachiocephalic vein"]},
        {"id": "rbcv", "structures": ["right brachiocephalic vein"]},
        {"id": "trunk", "structures": ["brachiocephalic trunk"]},
        {"id": "trachea", "structures": ["trachea"]},
        {"id": "esophagus", "structures": ["esophagus"]}]),
    "h2": dict(labels=[], render={"preset": "bare"}, level="T8", strategy="level", hotspot_zones=[
        {"id": "esophagus", "structures": ["esophagus"]},
        {"id": "aorta", "structures": ["aorta"]},
        {"id": "heart", "structures": ["heart"]},
        {"id": "cord", "structures": ["spinal cord"]}]),
    "h3": dict(labels=[], render={"preset": "bare"}, hotspot_zones=[
        {"id": "lul", "structures": ["left upper lobe"]},
        {"id": "lll", "structures": ["left lower lobe"]},
        {"id": "rml", "structures": ["right middle lobe"]},
        {"id": "rll", "structures": ["right lower lobe"]},
        {"id": "rul", "structures": ["right upper lobe"]},
        {"id": "heart", "structures": ["heart"]}],
        context_structures=["heart"]),
}


def shuffle_letters(rq: dict, seed: int) -> dict:
    """Number the labels in random order. Numbers on the image keep them distinct from the exam's
    A-E answer letters. An ``answer`` entry pins the correct structure's number, so a bank can
    spread its correct answers evenly across A-E (the choices are listed 1..n, so number = letter)."""
    import random

    labeled = [l for l in rq["labels"] if l["label"]]
    numbers = [str(i + 1) for i in range(len(labeled))]
    answer = rq.pop("answer", None)  # (structure, number): pin the correct answer's number
    if answer:
        target = next(l for l in labeled if l["structures"][0] == answer[0])
        target["label"] = answer[1]
        numbers.remove(answer[1])
        labeled = [l for l in labeled if l is not target]
    random.Random(seed).shuffle(numbers)
    for l, ch in zip(labeled, numbers):
        l["label"] = ch
    rq["labels"] = sorted(rq["labels"], key=lambda l: l["label"])
    return rq


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(command=sys.executable, args=["-m", "medimgq.server"])
    results = {}
    prev_used = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    used: list[str] = [r["source"]["case_id"] for q, r in prev_used.items() if not ONLY or q not in ONLY]
    async with Client(params) as c:
        for qid, rq in REQUESTS.items():
            if ONLY and qid not in ONLY:
                continue
            rq = shuffle_letters(rq, seed=sum(map(ord, qid)))
            rq = dict(rq)
            tool = rq.pop("tool", "labeled_image")
            if tool == "render_image":
                args = dict(rq, output_dir=str(OUT), name=qid, include_base64=False)
            else:
                args = dict(rq, seed=rq.get("seed", 7), output_dir=str(OUT), name=qid, include_base64=False,
                            filters={"exclude_cases": used, **rq.get("filters", {})})
            res = await c.call_tool(tool, args)
            if res.is_error:
                print(qid, "ERROR", res.content[0].text)
                continue
            sc = res.structured_content
            data = sc.get("result", sc) if sc else json.loads(res.content[0].text)
            used.append(data["source"]["case_id"])
            results[qid] = data
            s = data["source"]
            print(qid, s["case_id"], s["view"], s["slice"], s["vertebral_level"], s["window"], s["crop_region"],
                  data["answer_key"])
    prev = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
    prev.update(results)
    (OUT / "results.json").write_text(json.dumps(prev, indent=1))


asyncio.run(main())
