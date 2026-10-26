"""Filesystem locations, overridable with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TS_ROOT = Path(os.environ.get("MEDIMGQ_TS_ROOT", PROJECT_ROOT / "Totalsegmentator_dataset_v201"))
INDEX_DIR = Path(os.environ.get("MEDIMGQ_INDEX_DIR", PROJECT_ROOT / "data" / "index"))
BLOCKLIST = Path(os.environ.get("MEDIMGQ_BLOCKLIST", PROJECT_ROOT / "qc" / "blocklist.csv"))
OUTPUT_DIR = Path(os.environ.get("MEDIMGQ_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

TS_ATTRIBUTION = (
    "TotalSegmentator dataset v2.0.1 (Wasserthal et al., Radiology: AI 2023), CC BY 4.0"
)
