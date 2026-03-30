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
from agent.analyser import analyse
from agent.decision import decide
from agent.learning.feedback import log_correction
from agent.learning.distillation import run_distillation, _load_pending, KEYWORD_RULES_PATH, PENDING_RULES_PATH

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
        json.dump(decisions, output, indent=2, ensure_ascii=False)
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

        #Layer 2 - Analysis, skipped if validation has returned/escalated
        analysis = None
        if validation["passed"]: 
            analysis = analyse(record, mock=mock)
        
        #Layer 3 - assemble final decision 
        decision = decide(qid, validation, analysis)

        decisions.append(decision)
    
    return decisions


def handle_feedback(args) -> None: 
    """
    log a human correction and trigger distillation pipeline 

    requires --id, --human-decision, and --reason all to be provided 
    fetches the original record from the input file to extract the text fields - 
    avoids storing anything not needed in the feedback log

    args: 
        args: parsed argparse namespace
    """
    if not args.questionnaire_id or not args.human_decision or not args.reason: 
        sys.exit("--feedback requires --id, --human-decision, and --reason")

    records = load_records(args.input)
    record = next((r for r in records if r.get("questionnaire_id") == args.questionnaire_id), None)


    if record is None: 
        sys.exit(f"questionnaire_id '{args.questionnaire_id}' not found in {args.input}")

    try:
        with open(args.output, "r") as f:
            decisions = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Output file not found: {args.output}. Run the pipeline first.")
    except json.JSONDecodeError as e:
        sys.exit(f"Malformed output file: {e}")

    decision_record = next((d for d in decisions if d.get("questionnaire_id") == args.questionnaire_id), None)
    if decision_record is None:
        sys.exit(f"questionnaire_id '{args.questionnaire_id}' not found in {args.output}.")

    agent_decision = decision_record["decision"]

    log_correction(
        questionnaire_id=args.questionnaire_id,
        source_of_funds=record.get("source_of_funds_description", ""),
        accreditation_details=record.get("accreditation_details", ""),
        agent_decision=agent_decision,
        human_decision=args.human_decision,
        human_reason=args.reason,
    )

    if agent_decision != args.human_decision:
        from agent.learning.feedback import get_entry_by_id
        correction = get_entry_by_id(args.questionnaire_id)
        run_distillation(correction)
    else:
        print("Agent and human decisions match — skipping distillation.")


def handle_approve_rules() -> None:
    """
    Interactively present pending distilled rules for human approval.

    For each pending rule, shows the keyword, the evaluator's reasoning,
    and the source correction. The operator approves or rejects each one.
    Approved rules are appended to keyword_rules.json and removed from
    pending_rules.json.
    """
    try:
        with open(PENDING_RULES_PATH, "r") as f:
            pending = json.load(f)
    except FileNotFoundError:
        print("No pending rules to review.")
        return
    except json.JSONDecodeError as e:
        sys.exit(f"Error: malformed pending_rules.json: {e}")

    if not pending:
        print("No pending rules to review.")
        return

    approved_keywords = []
    remaining = []

    for rule in pending:
        print("\n" + "─" * 50)
        print(f"Keyword:          {rule['keyword']}")
        print(f"Evaluator reason: {rule['evaluator_reason']}")
        print(f"Source case ID:   {rule.get('source_questionnaire_id', 'unknown')}")
        print(f"Human reason:     {rule.get('human_reason', '')}")

        while True:
            choice = input("Approve this rule? [y/n]: ").strip().lower()
            if choice in ("y", "n"):
                break
            print("Please enter y or n.")

        if choice == "y":
            approved_keywords.append(rule["keyword"])
            print(f"Approved: '{rule['keyword']}'")
        else:
            remaining.append(rule)
            print(f"Rejected: '{rule['keyword']}'")

    if approved_keywords:
        # Append to live keyword list.
        try:
            with open(KEYWORD_RULES_PATH, "r") as f:
                rules = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            rules = {"keywords": []}

        existing = set(rules.get("keywords", []))
        new_keywords = [k for k in approved_keywords if k not in existing]
        rules["keywords"].extend(new_keywords)

        with open(KEYWORD_RULES_PATH, "w") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

        print(f"\nAdded {len(new_keywords)} new keyword(s) to keyword_rules.json.")

    # Write back only the rejected rules.
    with open(PENDING_RULES_PATH, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="PE fund subscription questionnaire processing agent."
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Process a questionnaire file.")
    run_parser.add_argument("--input", default="data/questionnaires.json", help="Path to input JSON (default: data/questionnaires.json)")
    run_parser.add_argument("--output", default="output/decisions.json", help="Path to write decisions (default: output/decisions.json)")
    run_parser.add_argument("--mock", action="store_true", help="Skip live LLM call in Layer 2.")

    # --- feedback ---
    fb_parser = subparsers.add_parser("feedback", help="Log a human correction for a processed record.")
    fb_parser.add_argument("--id", dest="questionnaire_id", required=True, help="Questionnaire ID to correct.")
    fb_parser.add_argument("--human-decision", required=True, choices=["Approve", "Return", "Escalate"])
    fb_parser.add_argument("--reason", required=True, help="Reason for the correction.")
    fb_parser.add_argument("--input", default="data/questionnaires.json", help="Input file (default: data/questionnaires.json)")
    fb_parser.add_argument("--output", default="output/decisions.json", help="Decisions file to read agent decision from (default: output/decisions.json)")

    # --- approve-rules ---
    subparsers.add_parser("approve-rules", help="Interactively approve or reject distilled pending rules.")

    args = parser.parse_args()

    if args.command == "run":
        records = load_records(args.input)
        decisions = run_pipeline(records, mock=args.mock)
        write_output(decisions, args.output)

    elif args.command == "feedback":
        handle_feedback(args)

    elif args.command == "approve-rules":
        handle_approve_rules()

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__": 
    main()