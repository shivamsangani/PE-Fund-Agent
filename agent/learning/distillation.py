"""
Rules distillation pipeline

When a human correction is logged, this module attempts to extract a reusable keyword 
rule from the correction via 2 LLM calls: 

1. Proposer: given the correction, propose a new keyword
2. Evaluator: given the proposed keyword and existing list, approve or reject 

Approved proposals are written to data/pending_rules.json for human review. 
Human reviews via 'python3 main.py --approve-rules' before anything is added to the
live keyword_rules.json

This two LLM pattern reduces the chance of low-quality or redundant rules reaching the
live keyword list 
"""

import json 
import os 
from pathlib import Path 

import anthropic 

KEYWORD_RULES_PATH = Path("data/keyword_rules.json")
PENDING_RULES_PATH = Path("data/pending_rules.json")
MODEL = "claude-haiku-4-5-20251001"

def run_distillation(correction: dict) -> None: 
    """
    run the full distillation pipeline for a single correction 

    proposes a new keyword, evaluates it, and if approved writes it to pending_rules.json
    for human review. prints staus at each step so the operator can follow along

    args: 
        correction: a feedback log entry dict with keys: 
                    source_of_funds_description, accreditation_details,
                    agent_decision, human_decision, human_reason
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: 
        print("ANTHROPIC_API_KEY not set. Skipping distillation")
        return

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Distillation: proposing new keyword rule...")
    proposed_keyword = _propose_keyword(client, correction)

    if not proposed_keyword: 
        print("Distillation: proposer returned no keyword. Skipping.")
        return

    approved, reason = _evaluate_keyword(client, proposed_keyword)

    if approved: 
        _write_pending_rule(proposed_keyword, reason, correction)
        print(f"Distillation: rule approved and written to pending_rules.json.")
        print(f"Run `python3 main.py approve-rules` to add it to the live keyword list.")
    else:
        print(f"Distillation: rule rejected. Reason: {reason}")       


def _propose_keyword(client: anthropic.Anthropic, correction: dict) -> str | None: 
    """
    ask the llm to propose a keyword or phrase that would catch similar cases

    args: 
        client: Anthropic client instance 
        correction: the feedback log entry dict
    
    returns: 
        a keyword/phrase string, or none if the call fails
    """
    prompt = (
        f"A human compliance officer corrected an AI agent's decision on a PE fund "
        f"subscription questionnaire.\n\n"
        f"Source of funds: \"{correction.get('source_of_funds_description', '')}\"\n"
        f"Accreditation details: \"{correction.get('accreditation_details', '')}\"\n"
        f"Agent decided: {correction.get('agent_decision')}\n"
        f"Human overrode to: {correction.get('human_decision')}\n"
        f"Human reason: {correction.get('human_reason')}\n\n"
        f"Based on this correction, propose a specific keyword or short phrase that "
        f"should be added to an escalation keyword list to catch similar cases in future. "
        f"Return only the keyword or phrase as a plain string, nothing else."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=64,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        keyword = response.content[0].text.strip().strip('"').lower()
        return keyword if keyword else None
    except Exception as e:
        print(f"distillation proposer call failed: {e}")
        return None

def _evaluate_keyword(client: anthropic.Anthropic, proposed_keyword: str) -> tuple[bool, str]: 
    """
    ask a second llm call to evaluate whether the proposed keyword should be approved

    checks specificity, consistency with existing list, and whether it would have caught the original case

    args: 
        client: anthropic client instance 
        proposed_keyword: the keyword string proposed in step 1

    returns: 
        tuple of (approved: bool, reason: str)
    """
    existing_keywords = _load_existing_keywords()

    prompt = (
        f"You are reviewing a proposed new escalation keyword for a PE fund compliance system.\n\n"
        f"Proposed keyword: \"{proposed_keyword}\"\n\n"
        f"Existing keyword list: {json.dumps(existing_keywords)}\n\n"
        f"Check the following:\n"
        f"1. Is it specific enough to be useful without causing false positives?\n"
        f"2. Is it consistent with the existing keyword list (not redundant, not contradictory)?\n"
        f"3. Would it plausibly catch ambiguous or concerning fund descriptions?\n\n"
        f"Return only valid JSON with no markdown, no code fences, no explanation. Exact format: "
        f'{{\"approved\": true or false, \"reason\": \"one sentence\"}}'
    )   


    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=128,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        result = json.loads(raw)
        return bool(result.get("approved", False)), result.get("reason", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: distillation evaluator response malformed: {e}")
        return False, "Evaluator response could not be parsed."
    except Exception as e:
        print(f"Warning: distillation evaluator call failed: {e}")
        return False, str(e)

def _load_existing_keywords() -> list:
    """
    load the current live keyword list from keyword_rules.json.

    returns:
        list of keyword strings. returns empty list if file is missing.
    """
    try:
        with open(KEYWORD_RULES_PATH, "r") as f:
            return json.load(f).get("keywords", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _write_pending_rule(keyword: str, evaluator_reason: str, correction: dict) -> None: 
    """
    append an approved-by-evaluator rule to the pending_rules.json

    args: 
        keyword: the proposed keyword string
        evaluator reason: the evaluator LLM's reason for approving 
        correction: the original correction dict for traceability
    """
    pending = _load_pending()

    if any(r.get("keyword") == keyword for r in pending): 
        print(f"Distillation: '{keyword}' already in pending_rules.json. Skipping.")
        return 
    
    pending.append({
        "keyword": keyword,
        "evaluator_reason": evaluator_reason,
        "source_questionnaire_id": correction.get("questionnaire_id"),
        "human_reason": correction.get("human_reason"),
    })

    PENDING_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PENDING_RULES_PATH, "w") as f:
        json.dump(pending, f, indent=2, ensure_ascii=False)    


def _load_pending() -> list:
    """
    Load pending_rules.json from disk.

    Returns:
        List of pending rule dicts. Returns empty list if file is missing.
    """
    try:
        with open(PENDING_RULES_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []