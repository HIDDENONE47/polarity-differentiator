"""
clean_dataset.py
One-off cleanup pass: removes manually-identified contaminated records
(false positives that passed the automated actionability bar but were
confirmed on manual review to be vendors/wealth-managers, not real family
offices) from the final dataset JSONL. Run this ONCE the pipeline run is
finished or paused — not while it's actively writing to the backup file.

Removed records are logged to rejected_log.json with a reason, never just
silently deleted — this preserves the evidence trail for the methodology
writeup.
"""

import json
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SOURCE_FILE = DOCS_DIR / "family_offices_dataset_BACKUP.jsonl"  # adjust if your resume system uses a different filename
OUTPUT_FILE = DOCS_DIR / "family_offices_dataset_CLEANED.jsonl"
REJECTED_LOG = DOCS_DIR / "rejected_log.json"

# Entity names confirmed via manual research to be vendors/wealth-managers
# marketing TO family offices, not family offices themselves.
EXCLUDED_ENTITIES = {
    "qp global family offices": "Confirmed via manual research: SEC-registered RIA (CRD #294879) that manages single-family offices for multiple wealthy families as an outsourced service — a vendor, not a family office.",
    "century private wealth": "Confirmed via manual research: DFSA-regulated wealth manager serving family offices, corporates, and institutions as clients; launched a public Nasdaq-licensed fund. Not a private family office.",
}


def clean_dataset():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"{SOURCE_FILE} not found. Check the actual filename your resume system writes to.")

    kept = []
    removed = []

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get("entity_name", "").strip().lower()
            if key in EXCLUDED_ENTITIES:
                removed.append({
                    "entity_name": record.get("entity_name"),
                    "reason": f"manually_removed_post_hoc: {EXCLUDED_ENTITIES[key]}",
                    "source": record.get("discovery_source"),
                })
            else:
                kept.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record) + "\n")

    existing_rejected = []
    if REJECTED_LOG.exists():
        with open(REJECTED_LOG, "r", encoding="utf-8") as f:
            existing_rejected = json.load(f)
    existing_rejected.extend(removed)
    with open(REJECTED_LOG, "w", encoding="utf-8") as f:
        json.dump(existing_rejected, f, indent=2)

    print(f"Kept: {len(kept)} records -> {OUTPUT_FILE.name}")
    print(f"Removed: {len(removed)} records -> logged to {REJECTED_LOG.name}")
    for r in removed:
        print(f"  - {r['entity_name']}")


if __name__ == "__main__":
    clean_dataset()