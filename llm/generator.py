import json
import logging
import os
from dotenv import load_dotenv

from llm.groq_client import call_groq
from llm.prompt_templates import (
    CHECKLIST_SYSTEM_PROMPT,
    CHECKLIST_USER_PROMPT
)

load_dotenv()

logger = logging.getLogger(__name__)

# â”€â”€ Allowed domains â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€      
VALID_DOMAINS = [
    "board_governance",
    "data_privacy",
    "risk_management",
    "audit_compliance",
    "shareholder_rights",
    "csr",
    "hr_policy",
    "financial_compliance",
]


def _parse_json_response(raw: str) -> list[dict]:
    """
    Parse JSON array from LLM response.
    Handles cases where LLM adds extra text around JSON.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from response
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    logger.warning("Could not parse JSON from LLM response")
    return []


def _validate_items(items: list[dict]) -> list[dict]:
    """
    Validate and clean checklist items.
    Ensures all required fields exist.
    """
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue

        text = item.get("item", "").strip()
        if not text:
            continue

        domain = item.get("domain", "").strip().lower()
        if domain not in VALID_DOMAINS:
            domain = "audit_compliance"

        valid.append({
            "item"               : text,
            "domain"             : domain,
            "source_section"     : item.get("source_section", "—"),
            "priority"           : item.get("priority", "Medium"),
            "action_type"        : item.get("action_type", "Process"),
            "evidence_required"  : item.get("evidence_required", "Documentation or log review."),
            "source_url"         : "",
            "chunk_id"           : str(item.get("chunk_id", "")),
            "compliance_framework": ""
        })

    return valid


def _enrich_with_metadata(
    items  : list[dict],
    results: list[dict]
) -> list[dict]:
    # Build lookup: chunk_id -> metadata
    chunk_map = {}
    for r in results:
        cid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id", "")
        meta = r.get("metadata", {})
        if cid:
            chunk_map[cid] = meta

    for item in items:
        cid = item.get("chunk_id", "")
        meta = chunk_map.get(cid)

        if meta:
            item["source_url"]           = meta.get("source_url","")
            item["compliance_framework"] = meta.get("compliance_framework","")
        else:
            source_section = item.get("source_section", "")
            for rcid, rmeta in chunk_map.items():
                sec = rmeta.get("section_title", "")
                if sec and (source_section.lower() in sec.lower() or sec.lower() in source_section.lower()):
                    item["source_url"]           = rmeta.get("source_url","")
                    item["chunk_id"]             = rcid
                    item["compliance_framework"] = rmeta.get("compliance_framework","")
                    break

    return items


def generate_checklist(
    context_string: str,
    results       : list[dict]
) -> list[dict]:
    """
    Generate a structured governance checklist
    from retrieved chunks using Groq API.
    """
    if not context_string or not context_string.strip():
        logger.warning(
            "generate_checklist called with empty context"
        )
        return []

    logger.info("Starting checklist generation via Groq API")

    user_prompt = CHECKLIST_USER_PROMPT.format(
        context=context_string
    )

    # ACCURACY FIX: Lowered temperature to 0.0 (from 0.3) to force purely deterministic, factual outputs.
    # High temperatures allow the model to get "creative", which leads to hallucinations in legal/compliance checks.
    logger.info("Attempt 1: Generating checklist with 0.0 temperature for maximum accuracy...")
    try:
        raw = call_groq(
            system_prompt = CHECKLIST_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            temperature   = 0.0,  # Zero temperature for deterministic facts
            max_tokens    = 3000  # Increased token limit so it doesn't arbitrarily cut off long checklists
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return []

    items = _parse_json_response(raw)

    if not items:
        logger.warning("Attempt 1 failed. Retrying with stricter prompt...")
        strict_prompt = (
            user_prompt
            + "\n\nCRITICAL: Return ONLY a raw JSON array. "
            "Start with [ and end with ]. "
            "No markdown. No explanation. "
            "No text before or after the array."
        )
        try:
            raw = call_groq(
                system_prompt = CHECKLIST_SYSTEM_PROMPT,
                user_prompt   = strict_prompt,
                temperature   = 0.0,
                max_tokens    = 3000
            )
            items = _parse_json_response(raw)
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return []

    if not items:
        return []

    items = _validate_items(items)
    items = _enrich_with_metadata(items, results)
    
    return items


def generate_answer(
    query  : str,
    context: str
) -> str:
    from llm.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        ANSWER_USER_PROMPT
    )

    if not query or not context:
        return "Insufficient context to answer this question."

    user_prompt = ANSWER_USER_PROMPT.format(
        query   = query,
        context = context
    )

    try:
        # ACCURACY FIX for Q&A Generation
        answer = call_groq(
            system_prompt = ANSWER_SYSTEM_PROMPT,
            user_prompt   = user_prompt,
            temperature   = 0.0, # Lock to zero to avoid Q&A hallucination
            max_tokens    = 1500
        )
        return answer
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Could not generate answer: {e}"

def stream_answer(
    query  : str,
    context: str
):
    from llm.prompt_templates import (
        ANSWER_SYSTEM_PROMPT,
        ANSWER_USER_PROMPT
    )
    from llm.groq_client import get_groq_client

    if not query or not context:
        yield "Insufficient context to answer this question."
        return

    user_prompt = ANSWER_USER_PROMPT.format(
        query   = query,
        context = context
    )
    
    model = os.getenv("GROQ_GENERATOR_MODEL", "llama-3.3-70b-versatile")

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model       = model,
            messages    = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature = 0.0,
            max_tokens  = 1500,
            stream      = True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Answer stream failed: {e}")
        yield f"Could not generate answer stream: {e}"
