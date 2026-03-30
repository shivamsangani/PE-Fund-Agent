"""
Feedback log and few-shot example retrieval.

Full implementation in Step 7. This stub provides get_few_shot_examples()
so analyser.py can import it immediately.
"""

import difflib
import json
from pathlib import Path

FEEDBACK_LOG_PATH = Path("data/feedback_log.json")


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
    try:
        with open(FEEDBACK_LOG_PATH, "r") as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if not log:
        return []

    query = f"{source} {accreditation}".lower()

    scored = []
    for entry in log:
        candidate = f"{entry.get('source_of_funds_description', '')} {entry.get('accreditation_details', '')}".lower()
        score = difflib.SequenceMatcher(None, query, candidate).ratio()
        scored.append((score, entry))

    # Take the top n by similarity score.
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:n]

    return [_format_example(entry) for _, entry in top]


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
