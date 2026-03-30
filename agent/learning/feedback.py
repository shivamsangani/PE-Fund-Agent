"""
feedback log management and few-shot example retrieval.

the feedback log stores human corrections to agent decisions. These corrections
serve two purposes:
  1. few-shot examples injected into the Layer 2 system prompt so the LLM
     learns from past overrides without retraining.
  2. Input to the distillation pipeline which proposes new keyword rules.

PII note: only the two free-text fields are stored — never investor_name,
investor_address, or tax_id.
"""

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_LOG_PATH = Path("data/feedback_log.json")


def log_correction(
    questionnaire_id: str,
    source_of_funds: str,
    accreditation_details: str,
    agent_decision: str,
    human_decision: str,
    human_reason: str,
) -> None:
    """
    append a human correction to the feedback log.

    args:
        questionnaire_id:    ID of the corrected record.
        source_of_funds:     source_of_funds_description from the original record.
        accreditation_details: accreditation_details from the original record.
        agent_decision:      The decision the agent originally made.
        human_decision:      The decision the human compliance officer made.
        human_reason:        Free-text reason for the override.
    """
    log = _load_log()

    entry = {
        "questionnaire_id": questionnaire_id,
        "source_of_funds_description": source_of_funds,
        "accreditation_details": accreditation_details,
        "agent_decision": agent_decision,
        "human_decision": human_decision,
        "human_reason": human_reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log.append(entry)
    _save_log(log)
    print(f"Correction logged for {questionnaire_id}.")


def get_few_shot_examples(source: str, accreditation: str, n: int = 3) -> list:
    """
    Retrieve the n most similar past human corrections from the feedback log.

    Similarity is computed by comparing the concatenated source+accreditation
    text against each logged correction using difflib.SequenceMatcher.
    No embeddings — keeps dependencies minimal.

    Args:
        source:        source_of_funds_description of the current record.
        accreditation: accreditation_details of the current record.
        n:             Maximum number of examples to return.

    Returns:
        List of formatted example strings for injection into the system prompt.
        Returns empty list if the feedback log is empty or missing.
    """
    log = _load_log()
    if not log:
        return []

    query = f"{source} {accreditation}".lower()

    scored = []
    for entry in log:
        candidate = (
            f"{entry.get('source_of_funds_description', '')} "
            f"{entry.get('accreditation_details', '')}".lower()
        )
        score = difflib.SequenceMatcher(None, query, candidate).ratio()
        scored.append((score, entry))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]
    return [_format_example(entry) for _, entry in top]


def get_entry_by_id(questionnaire_id: str) -> dict | None:
    """
    Look up a single feedback log entry by questionnaire ID.

    Used by the distillation pipeline to retrieve the full correction context.

    Args:
        questionnaire_id: The ID to look up.

    Returns:
        The matching entry dict, or None if not found.
    """
    log = _load_log()
    # Return the most recent correction for this ID, not the first.
    matches = [e for e in log if e.get("questionnaire_id") == questionnaire_id]
    return matches[-1] if matches else None


def _format_example(entry: dict) -> str:
    """
    Format a feedback log entry as a few-shot example string for prompt injection.

    Args:
        entry: A single feedback log dict.

    Returns:
        A human-readable string describing the case and the human override.
    """
    return (
        f"Case: Source of funds: \"{entry.get('source_of_funds_description', '')}\". "
        f"Accreditation: \"{entry.get('accreditation_details', '')}\".\n"
        f"Agent decided: {entry.get('agent_decision')}. "
        f"Human overrode to: {entry.get('human_decision')}. "
        f"Reason: {entry.get('human_reason')}."
    )


def _load_log() -> list:
    """
    Load the feedback log from disk.

    Returns:
        List of correction dicts. Returns empty list if file is missing or malformed.
    """
    try:
        with open(FEEDBACK_LOG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Warning: malformed feedback_log.json: {e}. Treating as empty.")
        return []


def _save_log(log: list) -> None:
    """
    Write the full feedback log back to disk.

    Args:
        log: The complete list of correction dicts to persist.
    """
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
