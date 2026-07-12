import json
import os
import sys

# --- FIX: Tell Python to also look inside the data_pipeline folder for imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "data_pipeline")))

from config import FamilyOfficeRecord
from agent_auditor import audit_record_deep

# 1. Load your existing dataset
DATASET_PATH = "docs/family_offices_dataset_BACKUP.jsonl"
OUTPUT_PATH = "docs/spotlight_records.json"

def run_spotlight():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Let the main script run a bit longer!")
        return

    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(FamilyOfficeRecord(**json.loads(line)))
                
    if len(records) < 3:
        print("Not enough records yet. We need at least 3 to run the spotlight.")
        return

    # 2. Pick the first 3 records for the deep audit
    spotlight_candidates = records[:3]
    spotlight_results = []

    print("Running Deep Audit (Cross-Source Corroboration) on 3 records...")
    for i, record in enumerate(spotlight_candidates):
        print(f"\nDeep Auditing [{i+1}/3]: {record.entity_name}")
        # This runs your special function that checks for a SECOND independent source
        deep_audited_record = audit_record_deep(record)
        spotlight_results.append(deep_audited_record.dict())
        print(f"  -> Finished auditing {record.entity_name}")

    # 3. Save the results as a pretty JSON artifact for Brian to grade
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(spotlight_results, f, indent=2)
        
    print(f"\nSuccess! Spotlight artifact saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_spotlight()