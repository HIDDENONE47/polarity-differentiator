"""
regenerate_csv.py
Rebuilds the flattened CSV deliverable from the cleaned JSONL, so the two
files can never drift out of sync. Reuses the exact same flattening logic
as graph_orchestrator.py's export_node — imported, not duplicated, so a
future change to the CSV schema only has to happen in one place.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow importing graph_orchestrator

import pandas as pd
from graph_orchestrator import _flatten_record_for_csv
from config import FamilyOfficeRecord

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CLEANED_JSONL = DOCS_DIR / "family_offices_dataset_BACKUP.jsonl"
OUTPUT_CSV = DOCS_DIR / "family_offices_dataset.csv"


def regenerate_csv():
    if not CLEANED_JSONL.exists():
        raise FileNotFoundError(f"{CLEANED_JSONL} not found. Run clean_dataset.py first.")

    rows = []
    with open(CLEANED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = FamilyOfficeRecord(**json.loads(line))
            rows.append(_flatten_record_for_csv(record))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Regenerated {OUTPUT_CSV.name} from {CLEANED_JSONL.name}: {len(rows)} records.")


if __name__ == "__main__":
    regenerate_csv()