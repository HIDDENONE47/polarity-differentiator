"""
agent_auditor.py
Adversarial validation layer.

Responsibility: the ONLY code in this pipeline authorized to promote a field
from UNVERIFIED to VERIFIED or COULD_NOT_VERIFY. It does this by re-fetching
the claimed source live and asking an adversarial LLM pass whether the page
content actually substantiates the claimed value — not by trusting the
extraction agent's say-so.

Failure mode this file exists to prevent: agent_extraction.py hallucinating
a plausible-sounding fact and that fact silently becoming "verified" with
no independent check.
"""

import os
import time
import json
import requests
from urllib.parse import urlparse
from typing import Optional
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
auditor_llm = ChatGroq(
    model=DEFAULT_CONFIG.auditor_llm_model,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

HUNTER_API_KEY = os.getenv("HUNTER_API_KEY")

# Cache fetched page content by URL. Important: within one extraction pass,
# multiple fields (thesis, mandate, aum, background, corporate_linkedin) all
# share the SAME source_url, since they came from one raw_extract call over
# one page. Without this cache we'd re-fetch that identical page 5+ times
# per record — a real cost on Tavily's free-tier request budget.
_page_cache: dict = {}


def _fetch_live_page_text(url: str) -> Optional[str]:
    """
    Re-fetches the source page live via Tavily's extract endpoint.
    Returns None if the page can't be reached — a normal, expected outcome
    (dead link, paywall, LinkedIn blocking scrapers) that must lead to
    COULD_NOT_VERIFY, never to a crash and never to a default "trust it".
    """
    if url in _page_cache:
        return _page_cache[url]
    try:
        result = tavily_client.extract(urls=[url])
        pages = result.get("results", [])
        failed = result.get("failed_results", [])
        if pages:
            content = pages[0].get("raw_content")
        else:
            content = None
            if failed:
                print(f"  [audit] extract failed for {url}: {failed[0].get('error', 'unknown')}")
    except Exception as e:
        print(f"  [audit] extract exception for {url}: {e}")
        content = None
    _page_cache[url] = content
    return content


def _adversarial_confirm(claimed_value: str, field_label: str, page_text: str) -> dict:
    """
    Asks the LLM to adjudicate whether page_text actually substantiates
    claimed_value. Returns {"supported": bool, "reasoning": str, "confidence": float}
    """
    prompt = f"""You are a skeptical fact-checker. Your job is to find reasons a claim
is WRONG, not reasons it might be right. Do not give benefit of the doubt.

Claimed fact ({field_label}): {claimed_value}

Source page content (may be partial or truncated):
{page_text[:8000]}

Question: Does the source page content ACTUALLY, EXPLICITLY state or clearly
support this claimed fact? Vague relatedness is not support. The entity being
merely mentioned near a number is not support for that number. Silence on the
fact means NOT supported.

WHAT COUNTS AS EXPLICIT SUPPORT (read this before answering "not supported"):
A direct declarative sentence stating the fact IS explicit support, even
without additional surrounding context, citations, or corroborating detail.
Example: the sentence "Assets under management stand at $1.62 billion." is,
by itself, full support for an AUM claim of $1.62 billion -- do not withhold
support because the sentence lacks a citation, a date, or further
explanation. Likewise, a labeled structured field (a "General information"
panel, key-value pair, or table row) whose label matches the fact being
checked (e.g. label "Year founded" supporting a founding-year claim) is
explicit support on its own. "Vague relatedness" means an unlabeled number
or fact floating near the entity's name with no direct statement tying it to
the claim -- it does NOT mean a plainly stated sentence or labeled field that
you are withholding credit from for lacking extra context it doesn't need.
If you find yourself writing a reason like "lacks context" or "no supporting
evidence" while the exact fact is stated in a direct sentence you can quote,
that is a mistake -- the direct statement itself is the evidence.

EXCEPTION for structured data: if the page contains a labeled field (e.g. a
"General information" panel, key-value pair, or table row) whose label
matches the fact being checked (e.g. label "Assets under management" or "AUM"
supporting an AUM claim; "Year founded" supporting a founding-year claim),
that IS explicit support even without surrounding prose sentences. Do not
reject a structured field-value pair for lacking narrative context — the
field label itself is the context. Vague relatedness still means an
unlabeled number floating near the entity's name, not a clearly labeled
field.

CRITICAL FORMATTING RULES:
1. Respond ONLY as compact JSON.
2. Do NOT wrap the response in an outer JSON array container ([ ... ]). 
3. Do NOT include any conversational prose before or after the JSON.
4. Do NOT include markdown block wrappers like ```json.

Required JSON shape: {{"supported": true/false, "reasoning": "<one sentence>", "confidence": <0.0 to 1.0>}}
"""
    response = invoke_llm_with_retry(auditor_llm, prompt)
    
    import re
    raw_output = response.content.strip()
    
    # ROBUST PARSING: Extract JSON block even if the LLM hallucinates markdown
    match = re.search(r'(\{.*\})', raw_output, re.DOTALL)
    if match:
        raw_output = match.group(1)

    try:
        parsed = json.loads(raw_output)
        result = {
            "supported": bool(parsed.get("supported", False)),
            "reasoning": str(parsed.get("reasoning", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
        }
        print(f"  [audit verdict] {field_label}: supported={result['supported']} "
              f"confidence={result['confidence']:.2f} -- {result['reasoning']}")
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # If the adjudicator itself fails to respond cleanly, we cannot trust
        # its verdict — default to "not supported" rather than guessing.
        print(f"  [audit error] unparseable JSON from auditor: {raw_output[:100]}")
        return {"supported": False, "reasoning": "Adjudicator response unparseable.", "confidence": 0.0}


def _verify_linkedin_identity(url: str, principal_name: str, entity_name: str) -> VerifiableField:
    page_text = _fetch_live_page_text(url)
    if not page_text:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    verdict = _adversarial_confirm(
        claimed_value=f"This LinkedIn profile belongs to {principal_name}, who works at {entity_name}.",
        field_label="principal_linkedin_identity_match",
        page_text=page_text,
    )
    if verdict["supported"] and verdict["confidence"] >= 0.4:
        return VerifiableField(
            value=url,
            status=VerificationStatus.VERIFIED,
            source_url=url,
            extraction_method=ExtractionMethod.WEB_SEARCH,
            verification_notes=verdict["reasoning"],
            confidence_score=verdict["confidence"],
        )
    return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)


def find_and_audit_principal_linkedin(principal_name: str, entity_name: str) -> VerifiableField:
    try:
        response = tavily_client.search(
            query=f'"{principal_name}" "{entity_name}" site:linkedin.com/in',
            search_depth="basic",
            max_results=5,
        )
    except Exception:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    candidate_url = next(
        (r["url"] for r in response.get("results", []) if "linkedin.com/in/" in r.get("url", "")),
        None,
    )
    if not candidate_url:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    return _verify_linkedin_identity(candidate_url, principal_name, entity_name)


def audit_field(field: VerifiableField, field_label: str) -> VerifiableField:
    """
    Takes a single UNVERIFIED VerifiableField and returns a new field that is
    either VERIFIED (with confirmed source + method + notes) or
    COULD_NOT_VERIFY (value cleared, per schema rules). This is the sole gate
    that may set status=VERIFIED anywhere in the pipeline.
    """
    if field.status != VerificationStatus.UNVERIFIED:
        return field  # already resolved, or never had a value

    if not field.value or not field.source_url:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    page_text = _fetch_live_page_text(field.source_url)
    if not page_text:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    verdict = _adversarial_confirm(field.value, field_label, page_text)

    if verdict["supported"] and verdict["confidence"] >= 0.4:
        return VerifiableField(
            value=field.value,
            status=VerificationStatus.VERIFIED,
            source_url=field.source_url,
            extraction_method=ExtractionMethod.WEB_SEARCH,  # verified via live re-fetch, not the original inference
            verification_notes=verdict["reasoning"],
            confidence_score=verdict["confidence"],
        )
    return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

def _extract_domain(website: Optional[str]) -> Optional[str]:
    """Pulls a bare domain (e.g. 'acmecapital.com') out of a website URL for Hunter.io lookups."""
    if not website:
        return None
    parsed = urlparse(website if "://" in website else f"https://{website}")
    domain = parsed.netloc or parsed.path
    return domain.replace("www.", "") or None


_BLOCKED_WEBSITE_DOMAINS = (
    "linkedin.com", "crunchbase.com", "bloomberg.com", "wikipedia.org",
    "facebook.com", "twitter.com", "x.com", "google.com", "sec.gov",
)


def _discover_entity_website(entity_name: str) -> Optional[str]:
    try:
        response = tavily_client.search(
            query=f'"{entity_name}" official website',
            search_depth="basic",
            max_results=5,
        )
    except Exception:
        return None

    for r in response.get("results", []):
        url = r.get("url", "")
        domain = _extract_domain(url)
        if domain and not any(b in domain for b in _BLOCKED_WEBSITE_DOMAINS):
            return url
    return None


def hunter_find_email(domain: str, full_name: str) -> dict:
    """
    Uses Hunter.io's Email Finder to discover/confirm a likely work email for a
    named person at a given company domain. Free tier: ~25 searches/month.
    Returns confidence 0-100 and any source URLs Hunter used as evidence.
    """
    if not HUNTER_API_KEY or not domain or not full_name:
        return {"email": None, "confidence": 0, "sources": []}

    name_parts = full_name.strip().split()
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else ""

    try:
        resp = requests.get(
            "https://api.hunter.io/v2/email-finder",
            params={
                "domain": domain,
                "first_name": first_name,
                "last_name": last_name,
                "api_key": HUNTER_API_KEY,
            },
            timeout=10,
        )
        data = resp.json().get("data", {}) or {}
        sources = [s.get("uri") for s in (data.get("sources") or []) if s.get("uri")]
        return {
            "email": data.get("email"),
            "confidence": data.get("score", 0) or 0,
            "sources": sources,
        }
    except Exception:
        return {"email": None, "confidence": 0, "sources": []}


def audit_work_email(field: VerifiableField, principal_name: str, entity_website: Optional[str]) -> VerifiableField:
    """
    Email-specific auditing path — used for ALL 50 records, not just the 3
    spotlight ones. Hunter.io is purpose-built for domain-based email discovery
    and outperforms generic web search/re-fetch for this one field specifically.

    If extraction already claimed an email, Hunter's independent lookup must
    AGREE with it to be marked VERIFIED. If extraction found nothing, Hunter
    can discover one itself. Disagreement between the two is flagged, never
    silently resolved by picking one.
    """
    domain = _extract_domain(entity_website)
    if not domain:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    result = hunter_find_email(domain, principal_name)

    if not result["email"] or result["confidence"] < 50:
        return VerifiableField(status=VerificationStatus.COULD_NOT_VERIFY)

    if field.value and field.value.lower().strip() != result["email"].lower().strip():
        return VerifiableField(
            status=VerificationStatus.COULD_NOT_VERIFY,
            verification_notes=(
                f"Extraction claimed '{field.value}' but Hunter.io independently found "
                f"'{result['email']}' (confidence {result['confidence']}). Conflicting "
                f"sources — flagged for manual review, not auto-resolved."
            ),
        )

    source_url = result["sources"][0] if result["sources"] else f"https://hunter.io/verify/{domain}"
    return VerifiableField(
        value=result["email"],
        status=VerificationStatus.VERIFIED,
        source_url=source_url,
        extraction_method=ExtractionMethod.DIRECT_SOURCE,
        verification_notes=f"Confirmed via Hunter.io Email Finder, confidence score {result['confidence']}/100.",
        confidence_score=result["confidence"] / 100,
    )

def audit_record(record: FamilyOfficeRecord, rate_limit_delay: float = 1.0) -> FamilyOfficeRecord:
    """
    Runs the adversarial audit over every high-value field in a record.
    Returns a new FamilyOfficeRecord where every field is resolved to
    VERIFIED or COULD_NOT_VERIFY — never leaves UNVERIFIED in the output.

    rate_limit_delay paces calls between fields/principals/signals to stay
    under Tavily's free-tier rate limits across a 50-record batch run.
    """
    audited_thesis = audit_field(record.investing_thesis, "investing_thesis")
    time.sleep(rate_limit_delay)
    audited_mandate = audit_field(record.investing_mandate, "investing_mandate")
    time.sleep(rate_limit_delay)
    audited_background = audit_field(record.background_info, "background_info")
    time.sleep(rate_limit_delay)
    audited_aum = audit_field(record.aum, "aum")
    time.sleep(rate_limit_delay)
    audited_linkedin = audit_field(record.corporate_linkedin, "corporate_linkedin")
    time.sleep(rate_limit_delay)

    website = record.website or _discover_entity_website(record.entity_name)

    audited_principals = []
    for p in record.principals:
        linkedin_result = audit_field(p.linkedin_url, f"{p.name}_linkedin_url")
        if linkedin_result.status != VerificationStatus.VERIFIED and p.name and p.name != "Unknown":
            linkedin_result = find_and_audit_principal_linkedin(p.name, record.entity_name)
        audited_principals.append(
            FamilyOfficePrincipal(
                name=p.name,
                title=p.title,
                linkedin_url=linkedin_result,
                work_email=audit_work_email(p.work_email, p.name, website),
                direct_phone=audit_field(p.direct_phone, f"{p.name}_direct_phone"),
            )
        )
        time.sleep(rate_limit_delay)

    audited_signals = []
    for s in record.signals:
        audited_signals.append(
            FamilyOfficeSignal(
                signal_type=s.signal_type,
                description=s.description,
                date=s.date,
                source=audit_field(s.source, f"signal_{s.signal_type}"),
            )
        )
        time.sleep(rate_limit_delay)

    return FamilyOfficeRecord(
        entity_name=record.entity_name,
        entity_type=record.entity_type,
        location=record.location,
        website=website,
        investing_thesis=audited_thesis,
        investing_mandate=audited_mandate,
        background_info=audited_background,
        aum=audited_aum,
        corporate_linkedin=audited_linkedin,
        principals=audited_principals,
        signals=audited_signals,
        discovery_source=record.discovery_source,
        record_created_at=record.record_created_at,
    )


def passes_actionability_bar(record: FamilyOfficeRecord) -> bool:
    """
    Per the assessment doc: a record with zero usable principal contact info
    is 'a hole in the product, not a formatting choice.' Use this after
    auditing to flag which candidate entities need re-sourcing rather than
    being accepted as-is into the final 50.
    """
    return any(
        p.work_email.status == VerificationStatus.VERIFIED
        or p.direct_phone.status == VerificationStatus.VERIFIED
        or p.linkedin_url.status == VerificationStatus.VERIFIED
        for p in record.principals
    )

def _find_corroborating_source(entity_name: str, field_label: str, claimed_value: str, exclude_url: str) -> Optional[dict]:
    """
    Searches for a SECOND, independent source to corroborate a claim, explicitly
    excluding the original source URL. Reserved for the 3 spotlight records —
    doubles API cost, which the free-tier budget can't absorb across all 50.
    """
    query = f"{entity_name} {field_label.replace('_', ' ')} {claimed_value}"
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        for r in response.get("results", []):
            if r.get("url") and r["url"] != exclude_url:
                return r
    except Exception:
        pass
    return None


def deep_audit_field(field: VerifiableField, field_label: str, entity_name: str) -> VerifiableField:
    """
    Stronger version of audit_field(): after the normal single-source adversarial
    check passes, additionally searches for a second independent source. If it
    corroborates, the field's notes reflect double confirmation. If none is found
    or it disagrees, that's stated honestly rather than silently upgraded or hidden.
    """
    single_source_result = audit_field(field, field_label)

    if single_source_result.status != VerificationStatus.VERIFIED:
        return single_source_result

    corroborating = _find_corroborating_source(
        entity_name, field_label, single_source_result.value, single_source_result.source_url
    )

    existing_notes = single_source_result.verification_notes or ""

    if not corroborating:
        single_source_result.verification_notes = (
            existing_notes + " (Single source only — no independent corroboration found.)"
        )
        return single_source_result

    second_page_text = corroborating.get("content", "")
    verdict = _adversarial_confirm(single_source_result.value, field_label, second_page_text)

    if verdict["supported"] and verdict["confidence"] >= 0.6:
        single_source_result.verification_notes = (
            existing_notes + f" Cross-confirmed by independent source: {corroborating['url']} "
            f"({verdict['reasoning']})"
        )
        single_source_result.confidence_score = max(
            single_source_result.confidence_score or 0, verdict["confidence"]
        )
    else:
        single_source_result.verification_notes = (
            existing_notes + f" Note: second source ({corroborating['url']}) did not corroborate this claim."
        )
    return single_source_result


def audit_record_deep(record: FamilyOfficeRecord) -> FamilyOfficeRecord:
    """
    Full validation-chain version of audit_record(), for the 3 records requiring
    complete discovery/extraction/enrichment/validation documentation. Applies
    cross-source corroboration to entity-level facts (thesis, AUM, background,
    etc.) since those have real independent secondary sources (press, filings).
    Principal contact fields still use audit_work_email/audit_field — a second
    free source for a direct phone number essentially never exists, so that
    extra cost isn't worth it there.
    """
    audited_thesis = deep_audit_field(record.investing_thesis, "investing_thesis", record.entity_name)
    audited_mandate = deep_audit_field(record.investing_mandate, "investing_mandate", record.entity_name)
    audited_background = deep_audit_field(record.background_info, "background_info", record.entity_name)
    audited_aum = deep_audit_field(record.aum, "aum", record.entity_name)
    audited_linkedin = deep_audit_field(record.corporate_linkedin, "corporate_linkedin", record.entity_name)

    website = record.website or _discover_entity_website(record.entity_name)

    audited_principals = []
    for p in record.principals:
        linkedin_result = audit_field(p.linkedin_url, f"{p.name}_linkedin_url")
        if linkedin_result.status != VerificationStatus.VERIFIED and p.name and p.name != "Unknown":
            linkedin_result = find_and_audit_principal_linkedin(p.name, record.entity_name)
        audited_principals.append(
            FamilyOfficePrincipal(
                name=p.name,
                title=p.title,
                linkedin_url=linkedin_result,
                work_email=audit_work_email(p.work_email, p.name, website),
                direct_phone=audit_field(p.direct_phone, f"{p.name}_direct_phone"),
            )
        )

    audited_signals = [
        FamilyOfficeSignal(
            signal_type=s.signal_type,
            description=s.description,
            date=s.date,
            source=audit_field(s.source, f"signal_{s.signal_type}"),
        )
        for s in record.signals
    ]

    return FamilyOfficeRecord(
        entity_name=record.entity_name,
        entity_type=record.entity_type,
        location=record.location,
        website=website,
        investing_thesis=audited_thesis,
        investing_mandate=audited_mandate,
        background_info=audited_background,
        aum=audited_aum,
        corporate_linkedin=audited_linkedin,
        principals=audited_principals,
        signals=audited_signals,
        discovery_source=record.discovery_source,
        record_created_at=record.record_created_at,
    )