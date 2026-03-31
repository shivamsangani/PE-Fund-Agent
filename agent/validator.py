"""
Layer 1 - Rule-Based Validator

Deterministic checks only - No LLM calls. 
Responsible for catching structurally incomplete or immediately disqualifying submissions
before any AI processing occurs. 
"""

REQUIRED_FIELDS = [
    "investor_name",
    "investor_address",
    "investment_amount",
    "is_accredited_investor",
    "signature_present",
    "tax_id_provided",
]

# Expected types for fields that have a specific required type.
# Fields present but of the wrong type are treated the same as missing.
_BOOLEAN_FIELDS = ["is_accredited_investor", "signature_present", "tax_id_provided"]
_STRING_FIELDS = ["investor_name", "investor_address"]

def validate(record: dict) -> dict:
    """
    Run all deterministic validation checks on a validation record.

    Checks for missing required fields, invalid values, and immediate disqualifiers
    (non-accredited investor). Missing fields trigger Return; non-accredited triggers Escalate.
    Both stop the pipeline - Layer 2 (analyser.py) is never called if this layer does not pass

    Args:
        record: A single questionnaire dictionary as parsed from the input JSON


    Returns:
        A dict with keys:
            passed (bool): True only if all checks clear.
            missing_fields (list[str]): Field names that are null or missing.
            escalate (bool): True if the investor is not accredited.
            escalation_reason (str | None): Human-readable reason if escalating.
    """

    missing_fields = _check_required_fields(record)
    _check_field_types(record, missing_fields)

    #Non-accredited is an immediate escalate regardless of other fields 
    #Check this even when fields are missing so the reason is recorded, 
    #but Return takes priority in the Decision Engine (Layer 3)

    escalate = False
    escalation_reason = None

    if record.get("is_accredited_investor") is False: 
        escalate = True
        escalation_reason = "Investor is not accredited"

    #Additional value checks - only meaningful if the value is present
    if record.get("investment_amount") is not None:
        amount = record["investment_amount"]
        # bool is a subclass of int in Python, so True/False must be explicitly excluded
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            if "investment_amount" not in missing_fields:
                missing_fields.append("investment_amount")
    
    if record.get("signature_present") is False: 
        if "signature_present" not in missing_fields: 
            missing_fields.append("signature_present")
    
    if record.get("tax_id_provided") is False:
        if "tax_id_provided" not in missing_fields: 
            missing_fields.append("tax_id_provided")

    passed = len(missing_fields) == 0 and not escalate

    return { 
        "passed": passed, 
        "missing_fields": missing_fields, 
        "escalate": escalate, 
        "escalation_reason": escalation_reason, 
    }


def _check_required_fields(record: dict) -> list:
    """
    Identify which required fields are null, missing, or empty

    Args:
        record: A single questionnaire dict

    Returns:
        List of field name strings that failed the presence check
    """
    missing = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        # Treat None, missing key, and empty string as absent.
        # Boolean False is valid but handled separately in validate().
        if value is None or value == "":
            missing.append(field)

    return missing


def _check_field_types(record: dict, missing_fields: list) -> None:
    """
    Validate that present fields match their expected types.

    Fields already in missing_fields are skipped. Fields that are present
    but of the wrong type are appended to missing_fields so they trigger
    a Return rather than silently being misinterpreted (e.g. the string
    "false" passing the boolean accreditation check).

    Args:
        record: A single questionnaire dict
        missing_fields: The list built by _check_required_fields; mutated in place.
    """
    for field in _BOOLEAN_FIELDS:
        if field in missing_fields:
            continue
        value = record.get(field)
        if value is not None and not isinstance(value, bool):
            missing_fields.append(field)

    for field in _STRING_FIELDS:
        if field in missing_fields:
            continue
        value = record.get(field)
        if value is not None and value != "" and not isinstance(value, str):
            missing_fields.append(field)
