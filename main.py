"""
CLI entry point for the PE fund subscription questionnaire processing agent. 

Orchestrates the 3 layer pipeline (validate -> analyse -> decide) across 
all records in the input file. Business logic lives entirely in agent/
This file only handles Input/Output, argument parsing, and top-level errors
"""

import argparse 
import json 
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.validator import validate

def load_records(input_path:str) -> list: 
    """
    Load and parse a JSON array of questionnaire records from disk. 

    Args: input_path: Path to the input JSON file. 

    Returns: List of questionnaire dicts. 

    Raises: SystemExit: On file not found or malformed JSON 
    """
    try: 
        with open(input_path, "r") as input: 
            records = json.load(input)
    except FileNotFoundError: 
        sys.exit(f"Input file not found: {input_path}")
    except json.JSONDecodeError as e: 
        sys.exit(f"Malformed Input JSON: {e}")

    if not isinstance(records, list): 
        sys.exit("Input file must contain a JSON array of records")
    
    return records

def write_output(decisions: list, output_path: str) -> None: 
    """
    Write the decisions array to disk as formatted JSON

    Args: 
        decisions: List of decision dicts in the exact output form 
        output_path: Destination file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as output: 
        json.dump(decisions, output, indent=2)
    print(f"Output written to {output_path}")

def run_pipeline(records: list, mock: bool=False) -> list: 
    """
    Run the full pipeline over a full list of questionnaire records

    Layer 1 always runs. Layer 2 is skipped if Layer 1 does not pass. 
    mock=True bypasses the LLM call in Layer 2 with a hardcoded response

    Args: 
        records: List of questionnaire dicts
        mock: if True, Layer 3 returns a hardcoded response instead of calling a LLM API

    Returns: 
        List of decision dicts matching the output format
    """
    decisions = []

    for record in records: 
        qid = record.get("questionnaire_id", "UNKNOWN")

        #Layer 1 - Validation
        validation = validate(record)

        decision = _placeholder_decision(qid, validation)
        decisions.append(decision)
    
    return decisions

def _placeholder_decision(questionnaire_id: str, validation: dict) -> dict:
    """
    Temporary decision builder used until Layer 3 is implemented.

    Converts Layer 1 output into the output schema format so the pipeline
    can be tested end-to-end before Layers 2 and 3 exist.

    Args:
        questionnaire_id: The record's ID string.
        validation: Output dict from agent.validator.validate().

    Returns:
        A decision dict matching the output schema.
    """
    if validation["missing_fields"]:
        return {
            "questionnaire_id": questionnaire_id,
            "decision": "Return",
            "missing_fields": validation["missing_fields"],
            "escalation_reason": None,
        }
    if validation["escalate"]:
        return {
            "questionnaire_id": questionnaire_id,
            "decision": "Escalate",
            "missing_fields": None,
            "escalation_reason": validation["escalation_reason"],
        }
    return {
        "questionnaire_id": questionnaire_id,
        "decision": "Approve",  # placeholder — Layer 2 may override this
        "missing_fields": None,
        "escalation_reason": None,
    }

def main(): 
    load_dotenv()

    parser = argparse.ArgumentParser( 
        description="PE fund subscription questionnairse processing agent"
    )

    parser.add_argument("--input", help="Path to input JSON file")
    parser.add_argument("--output", help="Path to write output JSON")
    parser.add_argument(
        "--mock", 
        action="store_true", 
        help="Run in mock mode - skips live LLM API call in Layer 2"
    )

    #Feedback mode
    parser.add_argument("--feedback", action="store_true", help="Log a human correction")
    parser.add_argument("--id", dest="questionnaire_id", help="Questionnaire ID to correct")
    parser.add_argument("--human-decision", choices=["Approve", "Return", "Escalate"])
    parser.add_argument("--reason", help="Human-provided reason for the correction")

    parser.add_argument(
        "--approve-rules", 
        action="store_true", 
        help="Approve or reject pending rules"
    )

    args = parser.parse_args()

    if args.feedback: 
        sys.exit("not implemented yet")
    
    if args.approve_rules: 
        sys.exit("not implemented")
    
    if not args.input or not args.output: 
        parser.print_help()
        sys.exit(1)
    
    records = load_records(args.input)
    decisions = run_pipeline(records, mock=args.mock)
    write_output(decisions, args.output)

if __name__ == "__main__": 
    main()