"""
Layer 2 - Free-Text Analyser 

Only runs when Layer 1 passes. Analyses source_of_funds_description and accreditation_details in 2 stages: 
1. Keyword matching (no LLM)
2. LLM assessment via Anthropic tool use (skipped in mock mode or if Stage 1 already escalates)

Data minimisation: only the two free-text fields are sent to the LLM, never the full record 
"""

import difflib 
import json
import os 
import re
from pathlib import Path 

import anthropic 

from agent.learning.feedback import get_few_shot_examples

KEYWORD_RULES_PATH = Path("data/keyword_rules.json")
MODEL = "claude-haiku-4-5-20251001"

#Tool schema passed to the Anthropic API to force structured output 
ASSESS_TOOL = { 
    "name": "assess_free_text", 
    "description": (
        "Assess the source of funds description and accrediation details for a PE fund subscription questionnaire"
    ), 
    "input_schema": {
        "type": "object",
        "properties": {
            "source_of_funds_assessment": {
                "type": "string",
                "enum": ["clear", "ambiguous", "concerning"],
                "description": "Assessment of the source of funds description.",
            },
            "accreditation_assessment": {
                "type": "string",
                "enum": ["clear", "ambiguous", "concerning"],
                "description": "Assessment of the accreditation details.",
            },
            "escalation_reason": {
                "type": "string",
                "description": "Required if either assessment is not 'clear'. Null otherwise. Maximum 8 words.",
            },
            "confidence": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Confidence score 0-100 in this assessment.",
            },
        },
        "required": [
            "source_of_funds_assessment",
            "accreditation_assessment",
            "escalation_reason",
            "confidence",
        ],
    },
}

def analyse(record: dict, mock: bool=False) -> dict: 
    """
    Run keyword matching then (if needed) LLM assessment. 

    Args: 
        record: Full question dict (only relevant fields are forwarded to LLM)
        mock: If true, skip the API call and return a hardcoded clear response
    
    Returns: 
        A dict with the keys: 
            passed (bool) : True if both stages are clear 
            escalate (bool) : True if either stage flags the record 
            escalation_reason (str | None) : Reason for escalation if needed 
    """
    source = record.get("source_of_funds_description")
    accreditation = record.get("accreditation_details")

    keyword_result = _check_keywords(source, accreditation)
    if keyword_result["matched"]: 
        return { 
            "passed": False, 
            "escalate": True, 
            "escalation_reason": f"Keyword match: '{keyword_result['pattern']}'",
        }
    
    if mock: 
        return _mock_llm_response()

    return _llm_assess(source, accreditation)


def _check_keywords(source: str, accreditation: str) -> dict: 
    """
    Check both fields against the keyword kust in keyword_rules.json

    Matching is case-insensitive and uses whole-phrase matching

    Args: 
        source: source_of_funds_description value 
        accreditation: accreditation_details value

    Returns: 
        Dict with keys: 
            matched (bool) : True if any keyword was found
            pattern (str | None) : The first matched keyword/phrase
    """
    try: 
        with open(KEYWORD_RULES_PATH, "r") as f: 
            rules = json.load(f)
            keywords = rules.get("keywords", [])
    except FileNotFoundError: 
        print(f"Warning: keyword_rules.json not found at {KEYWORD_RULES_PATH}. Skipping keyword check.")
        return {"matched": False, "pattern": None}      
    except json.JSONDecodeError as e:
        print(f"Warning: malformed keyword_rules.json: {e}. Skipping keyword check.")
        return {"matched": False, "pattern": None}

    combined = f"{source} {accreditation}".lower()

    #Sort by length descending so longer phrases match before their substrings
    for keyword in sorted(keywords, key=len, reverse=True): 
        if keyword.lower() in combined: 
            return {"matched": True, "pattern" : keyword}

    return {"matched": False, "pattern": None}       

def _llm_assess(source: str, accreditation: str) -> dict:
    """
    Call the Anthropic API to assess the fields

    Retrieve up to 3 similar past corrections from the feedback log and inject them as few-shot
    examples into the system prompt. Forces structured output via tool use 

    Args: 
        source: source_of_funds_description value
        accreditation: accreditation_Details value
    
    Returns: 
        Dict with keys: passed, escalate, escalation_reason (same as _check_keywords())

    Raises: 
        SystemExit: If ANTHROPIC_API_KEY is missing or the API call fails
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: 
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set")

    #Retrive few-shot examples from human corrections, injected into system prompt so model learns from past overrides. 
    few_shot = get_few_shot_examples(source, accreditation, n=3)
    system_prompt = _build_system_prompt(few_shot)

    user_message = (
        f"Please assess the following investor submission fields: \n\n"
        f"Source of funds: {source}\n\n"
        f"Accreditation details: {accreditation}"
    )

    try: 
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model = MODEL, 
            max_tokens = 512, 
            temperature = 0, 
            system = system_prompt, 
            tools = [ASSESS_TOOL],
            tool_choice={"type": "tool", "name":"assess_free_text"}, 
            messages = [
                {"role": "user", "content": user_message}
            ],
        )
    except anthropic.AuthenticationError:
        raise SystemExit("Invalid Anthropic API Key")
    except anthropic.APIConnectionError as e: 
        raise SystemExit(f"Could not connect to Anthropic API: {e}")
    except anthropic.APIStatusError as e: 
        raise SystemExit(f"Anthropic API returned status {e.status_code}: {e.message}")
    
    return _parse_llm_response(response)


def _build_system_prompt(few_shot_examples: list) -> str:
    """
    Build the system prompt, injecting few-shot examples when available.

    Args:
        few_shot_examples: List of formatted example strings from feedback log.

    Returns:
        Full system prompt string.
    """
    base = (
        "You are a compliance analyst at a private equity firm reviewing investor "
        "subscription questionnaires. Your job is to assess whether the source of "
        "funds description and accreditation details are sufficiently clear and "
        "credible for regulatory purposes. Vague, ambiguous, or potentially "
        "problematic descriptions must be flagged. Only the fields relevant to "
        "analysis are provided — not the full record. You must call the "
        "assess_free_text tool with your assessment. "
        "You are only assessing the quality and credibility of the content — "
        "not whether fields are present. Presence checks have already been completed. "
        "If a field contains a specific, verifiable claim, assess it as 'clear' "
        "unless there is a genuine substantive concern."
    )

    if not few_shot_examples:
        return base

    examples_text = "\n\n".join(few_shot_examples)
    return (
        f"{base}\n\n"
        f"The following are recent cases where a human compliance officer "
        f"overrode the agent's decision. Use these to calibrate your assessment:\n\n"
        f"{examples_text}"
    )

def _parse_llm_response(response) -> dict: 
    """
    Extract the tool call result from the Anthropic API response and map it to the output format 

    Confidence < 75 or any non-clear assessment triggers escalation 

    Args: 
        response: anthropic.types.Message

    Returns: 
        Dict with keys: passed, escalate, escalation_reason
    """
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"), 
        None, 
    )

    if tool_use_block is None: 
        #Model didnt call tool - low confidence 
        return {
            "passed" : False, 
            "escalate" : True, 
            "escalation_reason" : "LLM did not return a structured assessment - human review required"
        }

    result = tool_use_block.input

    confidence = result.get("confidence", 0)
    source_assessment = result.get("source_of_funds_assessment", "concerning")
    accreditation_assessment = result.get("accreditation_assessment", "concerning")
    escalation_reason = result.get("escalation_reason")

    if confidence < 75: 
        return {
            "passed": False,
            "escalate": True,
            "escalation_reason": "Low confidence — human review required",
        }

    if source_assessment != "clear" or accreditation_assessment != "clear": 
         return {
            "passed": False,
            "escalate": True,
            "escalation_reason": escalation_reason or "Ambiguous or concerning free-text fields",
        }

    return { 
        "passed": True, 
        "escalate": False, 
        "escalation_reason": None
    }               

def _mock_llm_response() -> dict:
    """
    Return a hardcoded passing response for use in mock mode.

    Layer 1 still runs fully in mock mode — this only replaces the API call.
    Records that should escalate via keyword matching will still escalate
    because Stage 1 runs before this is ever reached.

    Returns:
        Dict indicating a clear assessment with high confidence.
    """
    return {"passed": True, "escalate": False, "escalation_reason": None}