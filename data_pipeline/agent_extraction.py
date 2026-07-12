"""
agent_extraction.py
Sourcing and raw extraction logic.

Responsibility: find candidate Family Office entities and pull raw text about
them from live web sources. This layer does NOT decide what's "verified" —
it only ever produces UNVERIFIED records. agent_auditor.py is the sole
authority that can promote a field to VERIFIED or COULD_NOT_VERIFY.
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq


from config import (
    FamilyOfficeRecord,
    FamilyOfficePrincipal,
    FamilyOfficeSignal,
    VerifiableField,
    VerificationStatus,
    ExtractionMethod,
    DEFAULT_CONFIG,
    invoke_llm_with_retry,
)

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(
    model=DEFAULT_CONFIG.llm_model,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,  # deterministic extraction, not creative writing
    max_tokens=1500,  # raised from the emergency 800 now that extraction has its own quota
    # pool — still bounded, not unlimited, since daily budget is finite and shared across
    # every candidate attempted (see graph_orchestrator.py's MAX_CANDIDATES_SAFETY_CAP)
)


def discover_candidate_entities(query: str, max_results: int = 10) -> List[dict]:
    """
    Uses Tavily to find real, live web pages naming Family Office entities.
    Returns raw search hits (title, url, content snippet) — no LLM involved yet.
    """
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )
    return response.get("results", [])


def raw_extract_entity_from_source(entity_name: str, source_url: str, raw_text: str) -> FamilyOfficeRecord:
    """
    Given raw scraped/searched text about one entity, ask the LLM to structure it
    into a FamilyOfficeRecord. Every wrapped field is created with status=UNVERIFIED
    and extraction_method=LLM_INFERENCE — auditing happens in a separate pass.
    """
    prompt = f"""You are extracting structured facts about a family office from the text below.
Do not invent facts. If information is not present in the text, leave it out.

Entity (as found by search — may be an inaccurate page title, verify against the text): {entity_name}
Source URL: {source_url}

Raw text:
{raw_text[:4500]}

First, classify: is this entity itself plausibly a private family office — a dedicated
wealth/investment management entity serving one specific ultra-high-net-worth family, or
a small number of families (roughly 2-30 client families is still a normal-sized
multi-family office and should PASS — do not reject an entity just because it says it
serves "multiple families" or "multiple clients"; that phrase alone is not disqualifying).
Answer false for: research reports or articles about family offices, institutional asset
managers/OCIOs/endowment or pension managers that serve a broad mix of client types
(pensions, endowments, corporations, RIAs, banks) with family offices as only one category
among many, regulatory filings or guidance documents, or any unrelated company that just
happens to mention the term. Also answer false for: a private bank, wealth manager, or
trust company's internal "family office advisory," "family office group," or "family
office services" business unit. Also answer false for: software, technology, or
professional-service vendors (including recruitment/staffing agencies) that sell products
or services TO family offices — a job listing or "we placed this candidate" case study on
a recruiter's own site is about the recruiter, not the family office named in it.

Extract (if present in the text): the organization's actual proper name (not the raw
page title, not SEO text), investing thesis, investing mandate, background info, 
AUM, corporate LinkedIn URL, entity type, location, website, and any named principals 
with title/LinkedIn/email/phone if mentioned. Also extract any recent signals (investments, hires, news) with dates if given.

CRITICAL FORMATTING RULES:
1. You MUST return EXACTLY ONE valid JSON object mapping to the schema below. 
2. Do NOT wrap the response in an outer JSON array container ([ ... ]). 
3. Do NOT include any conversational prose before or after the JSON.
4. Do NOT include markdown block wrappers like ```json.

Respond ONLY as compact JSON with keys matching: is_family_office (true/false),
is_family_office_reasoning (one short sentence), entity_name, investing_thesis,
investing_mandate, background_info, aum, corporate_linkedin, entity_type, location,
website, principals (list of {{name, title, linkedin_url, work_email, direct_phone}}),
signals (list of {{signal_type, description, date}}). Use null for anything not found.
"""
    response = invoke_llm_with_retry(llm, prompt)
    import json
    import re
    raw_output = response.content.strip()
    
    # ROBUST PARSING: Extract JSON block even if the LLM hallucinates markdown or prose around it
    match = re.search(r'(\{.*\}|\[.*\])', raw_output, re.DOTALL)
    if match:
        raw_output = match.group(1)
        
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        # Still not valid JSON even after regex stripping — fail loud, don't guess
        raise ValueError(f"Extraction LLM returned non-JSON for {entity_name}: {response.content[:200]}")

    if isinstance(parsed, list):
        # Un-wrap if it still hallucinates an array
        if len(parsed) >= 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            raise ValueError(
                f"Extraction LLM returned a JSON array instead of one object for {entity_name}: "
                f"{response.content[:200]}"
            )

    if not isinstance(parsed, dict):
        raise ValueError(f"Extraction LLM returned unexpected JSON shape for {entity_name}: {response.content[:200]}")

    def _coerce_bool(value) -> bool:
        """Smaller Groq models sometimes emit "true"/"false" as JSON strings rather
        than real booleans — bool("false") is True in Python, so a naive truthy
        check would silently let a rejected entity through. Coerce explicitly."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)

    if not _coerce_bool(parsed.get("is_family_office", False)):
        reasoning = parsed.get("is_family_office_reasoning", "no reasoning given")
        cleaned_name = parsed.get("entity_name") or entity_name
        raise ValueError(f"Not a family office — {cleaned_name}: {reasoning}")

    def wrap(field_value: Optional[str]) -> VerifiableField:
        """Every extracted field starts life as UNVERIFIED — auditor promotes it later."""
        if not field_value:
            return VerifiableField(status=VerificationStatus.UNVERIFIED)
        field_value = str(field_value)  # guards against LLM returning aum as a raw JSON number
        return VerifiableField(
            value=field_value,
            status=VerificationStatus.UNVERIFIED,
            source_url=source_url,
            extraction_method=ExtractionMethod.LLM_INFERENCE,
        )

    principals = [
        FamilyOfficePrincipal(
            name=(p.get("name") or "Unknown"),  # .get(key, default) only fires on a MISSING
            # key — an explicit JSON null still returns None here, which crashes Pydantic's
            # str-typed name field unless coerced with `or`
            title=p.get("title"),
            linkedin_url=wrap(p.get("linkedin_url")),
            work_email=wrap(p.get("work_email")),
            direct_phone=wrap(p.get("direct_phone")),
        )
        for p in parsed.get("principals", []) or []
    ]


    signals = [
        FamilyOfficeSignal(
        signal_type=s.get("signal_type", "unknown"),
        description=s.get("description", ""),
        date=s.get("date"),
        source=VerifiableField(
            value=source_url,
            status=VerificationStatus.UNVERIFIED,
            source_url=source_url,
            extraction_method=ExtractionMethod.LLM_INFERENCE,
        ),
    )
    for s in parsed.get("signals", []) or []
    if s.get("description")  # skip empty/malformed signal entries
    ]

    record = FamilyOfficeRecord(
        signals=signals,
        entity_name=parsed.get("entity_name") or entity_name,  # prefer the LLM's cleaned name
        # over the raw search-result page title, which is often SEO/headline noise
        entity_type=parsed.get("entity_type"),
        location=parsed.get("location"),
        website=parsed.get("website"),
        investing_thesis=wrap(parsed.get("investing_thesis")),
        investing_mandate=wrap(parsed.get("investing_mandate")),
        background_info=wrap(parsed.get("background_info")),
        aum=wrap(parsed.get("aum")),
        corporate_linkedin=wrap(parsed.get("corporate_linkedin")),
        principals=principals,
        discovery_source=source_url,
    )
    return record


def rate_limited_search(query: str, max_results: int = 10, delay: float = 1.0) -> List[dict]:
    """Thin wrapper to respect Tavily's free-tier rate limits during batch discovery."""
    results = discover_candidate_entities(query, max_results=max_results)
    time.sleep(delay)
    return results