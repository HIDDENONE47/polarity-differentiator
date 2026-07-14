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

STRICT ATTRIBUTION RULE: only extract a fact if the text explicitly and directly
states it ABOUT this specific entity ({entity_name}) by name. Do NOT extract a fact
that actually comes from:
- a job/hiring listing FOR a role at the entity (what a recruiter says the hired
  person will do is not the entity making a statement about itself)
- a page that profiles or compares MULTIPLE different family offices (e.g. "Top 10
  Family Offices in X", "N Verified Firms" directories) — a number or description
  elsewhere on such a page may belong to a DIFFERENT entity even if this one is
  also named there
- a general statement about family offices as a category, not this specific one
If you are not sure a fact is explicitly and specifically about {entity_name} itself,
output null for that field. A missing field is expected and fine; an unsupported one
just gets thrown away downstream anyway, so guessing only wastes audit budget.

DO NOT UPGRADE VAGUE MENTIONS INTO SPECIFIC CLAIMS: if the text mentions a number,
date, or fact in one general context (e.g. team size, "years of experience", "a
limited number of clients"), do not reuse it to support a DIFFERENT, more specific
claim (a founding year, a client count, a "signal" event) unless the text itself
draws that connection explicitly. A signal must describe an actual EVENT — a specific
investment, hire, partnership announcement, award, or news item with some concrete
detail — not a static fact about team size, tenure, or general capability. If the
only evidence is a vague or general mention, leave the field or signal out entirely
rather than sharpening it into something more specific-sounding than the source.

IGNORE PLACEHOLDER/UNRENDERED VALUES: some pages contain animated stat counters
(e.g. "$0B AUM", "0 Years", "0 Clients") that display "0" or another placeholder
before JavaScript fills in the real number — a static scrape can capture this
pre-animation state. A bare "0" attached to a stat label is NOT a real reported
figure. Leave that field null rather than reporting a zero as if it were the
actual AUM, team size, or track record.

EACH PRINCIPAL'S FIELDS ARE THEIR OWN: a principal's linkedin_url, work_email,
and direct_phone must be found explicitly next to THAT person's name. Never
reuse a company-level fact, another principal's info, or a generic company
link for a different principal or a different field just because it's the
only concrete-looking thing on the page. If you cannot find a value clearly
and specifically tied to this one person, output null for that field.

LINKEDIN COMPANY PAGES NEED SPECIAL HANDLING: if the source is a linkedin.com/company
page, it is mostly structured metadata (Company size, Followers, Founded) with very
little real prose, and it causes two specific mistakes:
- corporate_linkedin is simply this page's own URL, {source_url} — do not try to
  find or infer it from the page content, and never put a number, count, or date
  into this field.
- Do NOT invent a principal's linkedin_url from a company page. Listing someone's
  name and title does not give you their personal profile URL unless the page
  contains an actual distinct link to it. If there isn't one, output null.
Also, do not turn LinkedIn's own UI labels (a company-size range, a follower count,
"Founded: N/A") into background_info or a signal — that is page furniture, not a
fact about the entity's history or activity.

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
    
    def _coerce_str(value) -> Optional[str]:
        """Multi-location or multi-URL entities sometimes make the LLM return a
        list instead of a plain string for a field the schema expects to be
        text-only (location, website) — join it into one string rather than
        crashing the whole record on a Pydantic validation error."""
        if value is None:
            return None
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v) or None
        return str(value)

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
            value=s.get("description", ""),  # the actual claim to verify -- auditing
            # the bare source_url against its own page text is a tautology that can
            # never meaningfully confirm or deny anything
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
        location=_coerce_str(parsed.get("location")),
        website=_coerce_str(parsed.get("website")),
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