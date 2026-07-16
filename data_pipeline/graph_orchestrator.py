"""
graph_orchestrator.py
LangGraph state-machine wiring: discover -> extract -> audit -> filter -> export.

Responsibility: run the full pipeline end-to-end, looping over discovery
queries and candidate entities until either 50 actionable records are
accepted or the discovery/safety budget runs out. This is the ONLY file
that decides which candidates are attempted and in what order — extraction,
auditing, and filtering logic all live in their own modules and are just
called from here.
"""

import os
import json
from pathlib import Path
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from config import FamilyOfficeRecord, VerifiableField, DEFAULT_CONFIG
from agent_extraction import tavily_client, raw_extract_entity_from_source, rate_limited_search
from agent_auditor import audit_record, passes_actionability_bar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

DISCOVERY_QUERIES = [
    # Broad single/multi family office footprint queries
    '"single family office" "investment mandate" "portfolio" -directory -list -top',
    '"single family office" "AUM" "co-investment" -news -wiki -directory',
    '"family investment firm" "managing director" "contact us"',
    '"multi-family office" "principals" "SEC filings" "wealth management"',
    
    # Platform-specific emulated footprint queries
    'site:linkedin.com/company "single family office" "investment firm"',
    'site:crunchbase.com "family office" "investments"',
    
    # Sector & Geographic strategic focus footprints
    '"family office" "healthcare" OR "biotech" "direct investment" "portfolio"',
    '"family office" "real estate portfolio" "United States" "investments"',
    '"family office" "fintech" OR "venture capital" London "contact"',
    '"family office" "Texas" OR "California" "investment strategy" "AUM"',
    
    # Direct transactional & news announcements
    '"family office" "direct investment" startup announcement "seed" OR "series"',
    '"family office" "press release" fund launch OR fund commitment 2025',

    # SEC Form ADV filings -- structured, public disclosure of AUM, client
    # types, and strategy/mandate language for any family office registered
    # as an RIA.
    'site:adviserinfo.sec.gov "family office"',
    '"single family office" "Form ADV" "assets under management"',
    '"family office" "SEC registered investment adviser" "AUM"',
]

# Safety cap on total candidates attempted, regardless of how many get accepted.
MAX_CANDIDATES_SAFETY_CAP = 500  
RESULTS_PER_QUERY = 25  # Expanded search depth to pull deep listings per query loop


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    queries: List[str]
    current_query_results: List[dict]
    raw_record: Optional[FamilyOfficeRecord]
    audited_record: Optional[FamilyOfficeRecord]
    accepted_records: List[FamilyOfficeRecord]
    rejected_log: List[dict]
    seen_entity_names: List[str]
    target_count: int
    total_processed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_full_text_for_extraction(url: str) -> Optional[str]:
    """
    Fetches fuller page content (beyond the short search snippet) to give the
    extraction LLM more to work with. Deliberately separate from the auditor's
    own re-fetch later — extraction-time fetching and audit-time re-fetching
    are different concerns and this keeps the modules independent, at the
    minor cost of duplicating one small fetch call.
    """
    if not url:
        return None
    try:
        result = tavily_client.extract(urls=[url])
        pages = result.get("results", [])
        return pages[0].get("raw_content") if pages else None
    except Exception:
        return None


def _flatten_field(field: VerifiableField, prefix: str) -> dict:
    return {
        f"{prefix}_value": field.value or "",
        f"{prefix}_status": field.status.value,
        f"{prefix}_source": field.source_url or "",
    }


def _flatten_record_for_csv(record: FamilyOfficeRecord) -> dict:
    """
    Flattens one record into a single CSV row. Only the primary (first)
    principal gets top-level columns, since that's the main decision-maker
    contact a client would act on first — the full principal list and full
    signal list are preserved as JSON in sidecar columns so no data is lost,
    just not all of it surfaced at top level.
    """
    row = {
        "entity_name": record.entity_name,
        "entity_type": record.entity_type or "",
        "location": record.location or "",
        "website": record.website or "",
        "discovery_source": record.discovery_source or "",
    }
    row.update(_flatten_field(record.investing_thesis, "investing_thesis"))
    row.update(_flatten_field(record.investing_mandate, "investing_mandate"))
    row.update(_flatten_field(record.background_info, "background_info"))
    row.update(_flatten_field(record.aum, "aum"))
    row.update(_flatten_field(record.corporate_linkedin, "corporate_linkedin"))

    primary = record.principals[0] if record.principals else None
    if primary:
        row["principal_name"] = primary.name
        row["principal_title"] = primary.title or ""
        row.update(_flatten_field(primary.linkedin_url, "principal_linkedin"))
        row.update(_flatten_field(primary.work_email, "principal_email"))
        row.update(_flatten_field(primary.direct_phone, "principal_phone"))
    else:
        row["principal_name"] = ""
        row["principal_title"] = ""
        for p in ["principal_linkedin", "principal_email", "principal_phone"]:
            row.update({f"{p}_value": "", f"{p}_status": "", f"{p}_source": ""})

    row["all_principals_json"] = json.dumps([p.model_dump(mode="json") for p in record.principals])
    row["signals_json"] = json.dumps([s.model_dump(mode="json") for s in record.signals])
    return row


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def discover_node(state: PipelineState) -> dict:
    """Refills current_query_results by advancing through DISCOVERY_QUERIES until a query returns hits."""
    if state["current_query_results"]:
        return {}  # already have candidates queued — nothing to do

    queries = list(state["queries"])
    hits: List[dict] = []
    while queries and not hits:
        next_query = queries.pop(0)
        print(f"[discover] searching: {next_query!r}")
        hits = rate_limited_search(next_query, max_results=RESULTS_PER_QUERY, delay=1.0)
        print(f"[discover] {len(hits)} candidate(s) found")

    return {"queries": queries, "current_query_results": hits}


def extract_node(state: PipelineState) -> dict:
    """Pops one candidate and runs raw extraction on it. Never lets a bad extraction crash the run."""
    candidates = list(state["current_query_results"])
    candidate = candidates.pop(0)
    entity_name = candidate.get("title", "Unknown Entity")
    source_url = candidate.get("url", "")
    print(f"[extract] attempting: {entity_name!r} ({source_url})")

    raw_text = _fetch_full_text_for_extraction(source_url) or candidate.get("content", "")

    try:
        raw_record = raw_extract_entity_from_source(entity_name, source_url, raw_text)
    except ValueError as e:
        print(f"[extract] rejected: {e}")
        return {
            "current_query_results": candidates,
            "raw_record": None,
            "total_processed": state["total_processed"] + 1,
            "rejected_log": state["rejected_log"] + [{
                "entity_name": entity_name,
                "reason": f"extraction_failed: {e}",
                "source": source_url,
            }],
        }

    print(f"[extract] passed — entering audit: {raw_record.entity_name!r}")
    return {
        "current_query_results": candidates,
        "raw_record": raw_record,
        "total_processed": state["total_processed"] + 1,
    }


def audit_node(state: PipelineState) -> dict:
    """Runs the full adversarial audit on the raw record. This is where VERIFIED/COULD_NOT_VERIFY get decided."""
    print(f"[audit] auditing {state['raw_record'].entity_name!r} — several LLM + Tavily calls, may take a bit")
    audited = audit_record(state["raw_record"], rate_limit_delay=2.5)
    print(f"[audit] done: {audited.entity_name!r}")
    return {"audited_record": audited, "raw_record": None}


def filter_node(state: PipelineState) -> dict:
    """
    Applies the actionability bar and dedupes by entity name. Rejected
    candidates are logged with a reason, never silently dropped —
    this log becomes evidence for the methodology writeup.
    """
    audited = state["audited_record"]
    entity_key = audited.entity_name.strip().lower()

    if entity_key in state["seen_entity_names"]:
        print(f"[filter] rejected (duplicate): {audited.entity_name!r}")
        return {
            "audited_record": None,
            "rejected_log": state["rejected_log"] + [{
                "entity_name": audited.entity_name,
                "reason": "duplicate_entity",
                "source": audited.discovery_source,
            }],
        }

    if not passes_actionability_bar(audited):
        print(f"[filter] rejected (no verified contact): {audited.entity_name!r}")
        principal_diagnostic = [
            {
                "name": p.name,
                "email_status": p.work_email.status.value,
                "phone_status": p.direct_phone.status.value,
                "linkedin_status": p.linkedin_url.status.value,
            }
            for p in audited.principals
        ]
        return {
            "audited_record": None,
            "seen_entity_names": state["seen_entity_names"] + [entity_key],
            "rejected_log": state["rejected_log"] + [{
                "entity_name": audited.entity_name,
                "reason": "failed_actionability_bar_no_verified_contact",
                "source": audited.discovery_source,
                "website": audited.website,
                "principals": principal_diagnostic or "no_principals_extracted",
            }],
        }

    print(f"[filter] ACCEPTED ({len(state['accepted_records']) + 1}/{state['target_count']}): {audited.entity_name!r}")
    
    # INCREMENTAL SAVE: Append to file instantly so we never lose data on a crash
    with open(DOCS_DIR / "family_offices_dataset_BACKUP.jsonl", "a", encoding="utf-8") as f:
        f.write(audited.model_dump_json() + "\n")

    return {
        "audited_record": None,
        "seen_entity_names": state["seen_entity_names"] + [entity_key],
        "accepted_records": state["accepted_records"] + [audited],
    }


def export_node(state: PipelineState) -> dict:
    """
    Writes two output formats deliberately: a flattened CSV (the human-readable
    client deliverable) and a JSONL file with full nested structure (clean input
    for the RAG ingestion step later — no need to re-parse the flattened CSV).
    Also writes the rejection log as its own file for methodology transparency.
    """
    import pandas as pd

    rows = [_flatten_record_for_csv(r) for r in state["accepted_records"]]
    df = pd.DataFrame(rows)
    df.to_csv(DOCS_DIR / "family_offices_dataset.csv", index=False)

    with open(DOCS_DIR / "family_offices_dataset.jsonl", "w", encoding="utf-8") as f:
        for r in state["accepted_records"]:
            f.write(r.model_dump_json() + "\n")

    with open(DOCS_DIR / "rejected_log.json", "w", encoding="utf-8") as f:
        json.dump(state["rejected_log"], f, indent=2)

    print(
        f"Export complete: {len(state['accepted_records'])} accepted, "
        f"{len(state['rejected_log'])} rejected, "
        f"{state['total_processed']} total candidates processed."
    )
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_discover(state: PipelineState) -> str:
    if state["current_query_results"]:
        return "extract"
    return "export"  # queries exhausted, nothing left to discover


def route_after_extract(state: PipelineState) -> str:
    if state["raw_record"] is None:
        return "discover"  # extraction failed — loop back for the next candidate
    entity_key = state["raw_record"].entity_name.strip().lower()
    if entity_key in state["seen_entity_names"]:
        # Skip the audit entirely for an entity already accepted/rejected in
        # a prior run -- auditing it again just burns several LLM + Tavily
        # calls to re-derive a result that gets thrown away in filter_node
        # anyway. Cheap to check now, expensive to discover after auditing.
        return "discover"
    return "audit"


def route_after_filter(state: PipelineState) -> str:
    if len(state["accepted_records"]) >= state["target_count"]:
        return "export"
    if state["total_processed"] >= MAX_CANDIDATES_SAFETY_CAP:
        return "export"  # safety cap hit — export whatever we have rather than risk exhausting API budget
    if not state["current_query_results"] and not state["queries"]:
        return "export"  # genuinely out of candidates
    return "discover"


# ---------------------------------------------------------------------------
# Graph assembly + entry point
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("discover", discover_node)
    graph.add_node("extract", extract_node)
    graph.add_node("audit", audit_node)
    graph.add_node("filter", filter_node)
    graph.add_node("export", export_node)

    graph.set_entry_point("discover")
    graph.add_conditional_edges("discover", route_after_discover, {"extract": "extract", "export": "export"})
    graph.add_conditional_edges("extract", route_after_extract, {"audit": "audit", "discover": "discover"})
    graph.add_edge("audit", "filter")
    graph.add_conditional_edges("filter", route_after_filter, {"discover": "discover", "export": "export"})
    graph.add_edge("export", END)

    return graph.compile()


def run_pipeline(target_count: int = DEFAULT_CONFIG.target_record_count):
    app = build_graph()
    
    # RECOVERY & DEDUPLICATION LAYER
    recovered_records = []
    seen_names = []
    backup_file = DOCS_DIR / "family_offices_dataset_BACKUP.jsonl"
    
    if backup_file.exists():
        with open(backup_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record_dict = json.loads(line)
                        record = FamilyOfficeRecord.model_validate(record_dict)
                        recovered_records.append(record)
                        seen_names.append(record.entity_name.strip().lower())
                    except Exception as e:
                        print(f"Warning: Could not parse a line in backup file: {e}")
        
        print(f"🔄 Recovered {len(recovered_records)} previously accepted records from backup.")
        print(f"Remaining to target: {target_count - len(recovered_records)}")

        # Also recover previously REJECTED entity names, so restarting after a Ctrl+C
    # or a rate-limit interruption doesn't re-spend a full audit pass re-confirming
    # something we already know fails the same way every time.
    recovered_rejected_log = []
    rejected_log_file = DOCS_DIR / "rejected_log.json"
    if rejected_log_file.exists():
        try:
            with open(rejected_log_file, "r", encoding="utf-8") as f:
                recovered_rejected_log = json.load(f)
            for entry in recovered_rejected_log:
                name = entry.get("entity_name", "").strip().lower()
                if name and name not in seen_names:
                    seen_names.append(name)
            print(f"🔄 Recovered {len(recovered_rejected_log)} previously rejected entity names.")
        except Exception as e:
            print(f"Warning: Could not parse rejected_log.json: {e}")

    initial_state: PipelineState = {
        "queries": list(DISCOVERY_QUERIES),
        "current_query_results": [],
        "raw_record": None,
        "audited_record": None,
        "accepted_records": recovered_records,  # Feeds previously saved records back into state
        "rejected_log": [],
        "seen_entity_names": seen_names,        # Memory map to automatically skip duplicates
        "target_count": target_count,
        "total_processed": 0,
    }
    
    # Increased recursion limit to allow deep iteration loops over massive search streams
    final_state = app.invoke(initial_state, config={"recursion_limit": 2000})
    return final_state


if __name__ == "__main__":
    run_pipeline(target_count=50)