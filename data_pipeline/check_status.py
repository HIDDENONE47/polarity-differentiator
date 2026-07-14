"""
check_stats.py

Run this any time to see the real verification state of your dataset --
no more eyeballing terminal logs for supported=True/False lines.

Reads from family_offices_dataset_BACKUP.jsonl, the crash-safe, always-current
source of truth (not family_offices_dataset.jsonl/.csv, which only refresh when
the pipeline reaches natural completion).

Run from either the repo root or from data_pipeline/ -- it checks both locations.
"""
import json
from pathlib import Path

for candidate in [
    Path("docs/family_offices_dataset_BACKUP.jsonl"),
    Path("../docs/family_offices_dataset_BACKUP.jsonl"),
]:
    if candidate.exists():
        path = candidate
        break
else:
    raise FileNotFoundError(
        "Can't find family_offices_dataset_BACKUP.jsonl -- "
        "run this from the repo root or from data_pipeline/"
    )

recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
print(f"Total accepted records: {len(recs)}\n")

entity_fields = ["investing_thesis", "investing_mandate", "background_info", "aum", "corporate_linkedin"]
print("Entity-level fields:")
for f in entity_fields:
    v = sum(1 for r in recs if r[f]["status"] == "verified")
    print(f"  {f:20s} {v:3d} / {len(recs)} verified")

principal_fields = ["linkedin_url", "direct_phone", "work_email"]
total_principals = sum(len(r["principals"]) for r in recs)
print("\nPrincipal-level fields:")
for f in principal_fields:
    v = sum(1 for r in recs for p in r["principals"] if p[f]["status"] == "verified")
    print(f"  {f:20s} {v:3d} / {total_principals} verified")

sig_total = sum(len(r["signals"]) for r in recs)
sig_v = sum(1 for r in recs for s in r["signals"] if s["source"]["status"] == "verified")
print(f"\nSignals:               {sig_v:3d} / {sig_total} verified")

zero_fact_records = sum(
    1 for r in recs if not any(r[f]["status"] == "verified" for f in entity_fields)
)
print(f"\nRecords with ZERO verified entity-level facts: {zero_fact_records} / {len(recs)}")