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
)

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(
    model=DEFAULT_CONFIG.llm_model,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,  # deterministic extraction, not creative writing
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

Entity: {entity_name}
Source URL: {source_url}

Raw text:
{raw_text[:6000]}

Extract (if present in the text): investing thesis, investing mandate, background
info, AUM, corporate LinkedIn URL, entity type, location, website, and any named
principals with title/LinkedIn/email/phone if mentioned. Also extract any recent
signals (investments, hires, news) with dates if given.

Respond ONLY as compact JSON with keys matching: investing_thesis, investing_mandate,
background_info, aum, corporate_linkedin, entity_type, location, website, principals
(list of {{name, title, linkedin_url, work_email, direct_phone}}), signals (list of
{{signal_type, description, date}}). Use null for anything not found. No prose, no markdown.
"""
    response = llm.invoke(prompt)
    import json
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # LLM didn't return clean JSON — fail loud, don't guess
        raise ValueError(f"Extraction LLM returned non-JSON for {entity_name}: {response.content[:200]}")

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
            name=p.get("name", "Unknown"),
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
        entity_name=entity_name,
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