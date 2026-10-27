"""End-to-end MCP tests over the real stdio transport (needs the dataset + index)."""

from __future__ import annotations

import json
import sys

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from medimgq.config import INDEX_DIR

pytestmark = pytest.mark.skipif(not (INDEX_DIR / "slices.parquet").exists(), reason="index not built")

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "medimgq.server"])


def _data(result) -> dict:
    if getattr(result, "structured_content", None):
        sc = result.structured_content
        return sc.get("result", sc)
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_tools_end_to_end(tmp_path):
    async with Client(SERVER) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert {"list_structures", "dataset_info", "find_slices", "slice_contents", "render_image",
                "labeled_image", "batch_labeled_images", "list_findings", "find_findings"} <= names

        r = _data(await c.call_tool("list_structures", {"query": "IVC"}))
        assert r["resolved"] == ["inferior_vena_cava"]

        bad = await c.call_tool("find_slices", {"structures": ["superior mesenteric artery"]})
        assert bad.is_error

        r = _data(await c.call_tool("labeled_image", {
            "labels": [{"label": "A", "structures": ["pancreas"]}, {"label": "B", "structures": ["kidneys"]}],
            "hotspot_zones": [{"id": "pancreas", "structures": ["pancreas"]}],
            "seed": 3, "output_dir": str(tmp_path), "include_base64": True,
        }))
        assert r["answer_key"] == {"A": "pancreas", "B": "left kidney, right kidney"}
        assert r["image_base64"] and (tmp_path / r["image_path"].split("/")[-1]).exists()
        assert r["hotspot_masks"][0]["path"].endswith("_mask_pancreas.png")
        assert len(r["key"][1]["arrows_px"]) == 2  # shared label -> one arrow per kidney

        src = r["source"]
        contents = _data(await c.call_tool("slice_contents", {
            "case_id": src["case_id"], "view": src["view"], "slice_index": src["slice"]}))
        assert "pancreas" in {s["name"] for s in contents["structures"]}

        b = _data(await c.call_tool("batch_labeled_images", {
            "requests": [{"labels": [{"label": "A", "structures": ["liver"]}], "seed": 1},
                         {"labels": [{"label": "A", "structures": ["liver"]}], "seed": 1}],
            "output_dir": str(tmp_path / "batch")}))
        cases = [x["source"]["case_id"] for x in b["results"]]
        assert len(set(cases)) == 2  # avoid_repeat_cases
        assert (tmp_path / "batch" / "manifest.jsonl").read_text().count("\n") == 2

        lf = _data(await c.call_tool("list_findings", {}))
        assert {f["name"] for f in lf["measured_findings"]} >= {"thoracic_aortic_aneurysm", "goiter", "renal_cyst"}
        ff = _data(await c.call_tool("find_findings", {"finding": "thoracic_aortic_aneurysm", "limit": 2}))
        top = ff["cases"][0]
        assert top["value"] >= ff["threshold"]

        # render the finding, pointing at the descending (posterior) aorta, with shuffled numbers
        r = _data(await c.call_tool("render_image", {
            "case_id": top["case_id"], "view": "axial", "slice_index": top["slice"],
            "labels": [{"label": "1", "structures": ["aorta"], "component": "posterior"},
                       {"label": "2", "structures": ["heart"]}, {"label": "3", "structures": ["esophagus"]}],
            "shuffle_labels": True, "seed": 5, "output_dir": str(tmp_path), "include_base64": False}))
        assert sorted(r["answer_key"]) == ["1", "2", "3"]
        assert set(r["answer_key"].values()) == {"aorta", "heart", "esophagus"}

