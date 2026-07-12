"""
config.py
Schema definitions and data models for the Family Office dataset pipeline.

Design principle: every high-value cell (per the assessment's scoring criteria)
carries its own verification metadata. A value with no source/method is not
allowed to present itself as "verified" anywhere downstream.
"""

import time
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from groq import RateLimitError


# ---------------------------------------------------------------------------
# Verification status
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    VERIFIED = "verified"                  # Has a live source URL + stated method
    COULD_NOT_VERIFY = "could_not_verify"  # Honest blank — allowed and scored as candor
    UNVERIFIED = "unverified"              # Internal pipeline state only.
                                            # Must NEVER reach the final CSV/export.


class ExtractionMethod(str, Enum):
    WEB_SEARCH = "web_search"              # e.g. Tavily result
    LLM_INFERENCE = "llm_inference"        # Model inferred/summarized from raw text
    DIRECT_SOURCE = "direct_source"        # Pulled directly from a named page (e.g. LinkedIn bio)
    MANUAL_SPOT_CHECK = "manual_spot_check"  # Human (you) verified directly — allowed per the doc


# ---------------------------------------------------------------------------
# Verifiable field wrapper
# ---------------------------------------------------------------------------

class VerifiableField(BaseModel):
    """
    Wraps any high-value data cell with its provenance.
    This is the object that makes 'verified' a checkable claim, not a label.
    """
    value: Optional[str] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    source_url: Optional[str] = None
    extraction_method: Optional[ExtractionMethod] = None
    verification_notes: Optional[str] = None   # e.g. "cross-checked against SEC ADV filing"
    confidence_score: Optional[float] = None   # 0.0–1.0, set by the auditor agent later

    @model_validator(mode="after")
    def enforce_verification_integrity(self):
        if self.status == VerificationStatus.VERIFIED:
            if not self.source_url or not self.extraction_method:
                raise ValueError(
                    f"Field marked VERIFIED without source_url/extraction_method "
                    f"(value={self.value!r}). This is exactly the failure mode the "
                    f"assessment penalizes — fix the caller, don't relax this check."
                )
        if self.status == VerificationStatus.COULD_NOT_VERIFY and self.value:
            raise ValueError(
                f"Field marked COULD_NOT_VERIFY but still has a value ({self.value!r}). "
                f"An honest blank must actually be blank."
            )
        return self

    def is_exportable(self) -> bool:
        """A field is only exportable if it's honestly resolved — never left UNVERIFIED."""
        return self.status in (VerificationStatus.VERIFIED, VerificationStatus.COULD_NOT_VERIFY)


# ---------------------------------------------------------------------------
# Family Office record schema
# ---------------------------------------------------------------------------

class FamilyOfficePrincipal(BaseModel):
    """Decision-maker intelligence — the highest-value section per the assessment doc."""
    name: str
    title: Optional[str] = None
    linkedin_url: VerifiableField = Field(default_factory=VerifiableField)
    work_email: VerifiableField = Field(default_factory=VerifiableField)
    direct_phone: VerifiableField = Field(default_factory=VerifiableField)


class FamilyOfficeSignal(BaseModel):
    """A single recent activity/signal — a record can have multiple."""
    signal_type: str  # e.g. "recent_investment", "fund_commitment", "key_hire", "news"
    description: str
    date: Optional[str] = None
    source: VerifiableField = Field(default_factory=VerifiableField)


class FamilyOfficeRecord(BaseModel):
    # --- Structural / low-stakes fields (plain values, no verification wrapper) ---
    entity_name: str
    entity_type: Optional[str] = None      # single-family vs multi-family office
    location: Optional[str] = None
    website: Optional[str] = None

    # --- High-value entity attributes (wrapped) ---
    investing_thesis: VerifiableField = Field(default_factory=VerifiableField)
    investing_mandate: VerifiableField = Field(default_factory=VerifiableField)
    background_info: VerifiableField = Field(default_factory=VerifiableField)
    aum: VerifiableField = Field(default_factory=VerifiableField)
    corporate_linkedin: VerifiableField = Field(default_factory=VerifiableField)

    # --- Principal intelligence ---
    principals: List[FamilyOfficePrincipal] = Field(default_factory=list)

    # --- Signals / recent activity ---
    signals: List[FamilyOfficeSignal] = Field(default_factory=list)

    # --- Pipeline metadata ---
    discovery_source: Optional[str] = None   # where the agent first found this entity
    record_created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def high_value_fields(self) -> dict:
        """Returns all wrapped fields for auditing — used by agent_auditor.py."""
        return {
            "investing_thesis": self.investing_thesis,
            "investing_mandate": self.investing_mandate,
            "background_info": self.background_info,
            "aum": self.aum,
            "corporate_linkedin": self.corporate_linkedin,
        }


# ---------------------------------------------------------------------------
# Pipeline-wide config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    target_record_count: int = 50
    min_signals_per_record: int = 1
    require_at_least_one_principal_contact: bool = True  # else record fails actionability
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"# via Groq — extraction: harder task, keep the bigger model
    auditor_llm_model: str = "llama-3.1-8b-instant"  # via Groq — audit is a bounded yes/no judgment call,
    # and Groq quotas are per-model/per-org, so this gives audit its own 500K-tokens/day pool instead of
    # fighting extraction for the 70B model's much smaller 100K/day


DEFAULT_CONFIG = PipelineConfig()


# ---------------------------------------------------------------------------
# Shared LLM call helper
# ---------------------------------------------------------------------------

def invoke_llm_with_retry(llm, prompt: str, max_retries: int = 6, initial_wait: float = 10.0):
    """
    Calls llm.invoke(prompt) with exponential backoff on Groq's free-tier
    rate limit (429 RateLimitError). Hitting the TPM cap is an expected,
    routine condition under normal pipeline load on the free tier — not a
    bug — so the pipeline must tolerate it rather than crash on first hit.
    Shared by agent_extraction.py and agent_auditor.py, which each hold
    their own ChatGroq client instance but should retry identically.
    """
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # exhausted retries — fail loud, don't guess
            print(f"  [groq rate limit] waiting {wait:.0f}s before retry {attempt + 1}/{max_retries - 1}...")
            time.sleep(wait)
            wait *= 2