"""
Layer 3 - Decision Engine

Assembles the final decision object from the outputs of Layer 1 (validator) and Layer 2 (analyser)
Enforces strict priority so that the most important decision always wins 
"""

def decide(questionnaire_id: str, validation: dict, analysis: dict | None) -> dict: 
    """
    Priority order: 
    1. Missing required fields -> Return
    2. Non-accredited investor -> Escalate 
    3. Keyword match in free text -> Escalate
    4. LLM flags ambiguous or concerning -> Escalate
    5. LLM confidence < 75 -> Escalate
    6. All clear -> Approve

    Priorities 3-5 are all reported via analysis["escalation_reason"]

    Args: 
        questionnaire_id : The record ID's string
        validation : Output dict from agent.validator.validate()
        analysis: Output dict from agent.analyser.analyse(), or None if Layer 2 was not reached

   Returns:
        A decision dict matching the exact output schema:
        {
            "questionnaire_id": str,
            "decision": "Approve" | "Return" | "Escalate",
            "missing_fields": list[str] | None,
            "escalation_reason": str | None,
        }
    """
    #missing fields always produce return if it needed to be escalated for other reasons
    #return is more actionable - the investor can fix and resubmit
    if validation["missing_fields"]:
        return {
            "questionnaire_id": questionnaire_id,
            "decision": "Return",
            "missing_fields": validation["missing_fields"],
            "escalation_reason": None,
        }
    
    #non-accredited investor - cannot be resolved by resubmission, so escalate rather than return
    if validation["escalate"]:
        return {
            "questionnaire_id": questionnaire_id,
            "decision": "Escalate",
            "missing_fields": None,
            "escalation_reason": validation["escalation_reason"],
        }
    
    #layer 2 flags (priorities 3-5)
    if analysis and analysis["escalate"]: 
        return {
            "questionnaire_id": questionnaire_id,
            "decision": "Escalate",
            "missing_fields": None,
            "escalation_reason": analysis["escalation_reason"],
        }
    
    #all checks passed
    return {
        "questionnaire_id": questionnaire_id,
        "decision": "Approve",
        "missing_fields": None,
        "escalation_reason": None,
    }    