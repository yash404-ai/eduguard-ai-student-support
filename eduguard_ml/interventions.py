"""Supportive, rule-based intervention SUGGESTIONS. Nothing is created automatically:
staff must review each suggestion and record a decision. No rule recommends punishment, exclusion or denial of opportunity."""
from datetime import datetime, timezone

RULES = {
    "uci_dropout": [
        ("Debtor", lambda r: str(r.get("Debtor")) == "1", "Student has outstanding debt", "Offer a meeting with financial-aid / scholarship advisor"),
        ("Tuition fees up to date", lambda r: str(r.get("Tuition fees up to date")) == "0", "Tuition fees not up to date", "Discuss payment plans or hardship support"),
        ("Curricular units 2nd sem (approved)", lambda r: float(r.get("Curricular units 2nd sem (approved)", 99)) <= 1, "Few 2nd-semester units approved", "Academic advising and tutoring referral"),
        ("Curricular units 2nd sem (without evaluations)", lambda r: float(r.get("Curricular units 2nd sem (without evaluations)", 0)) >= 2, "Several units without evaluations", "Check-in on attendance/assessment barriers"),
    ],
    "oulad_withdrawal": [
        ("days_since_last_click", lambda r: float(r.get("days_since_last_click", 0)) >= 14, "No VLE activity for 2+ weeks", "Friendly outreach message from tutor"),
        ("assess_submit_rate", lambda r: float(r.get("assess_submit_rate") or 1) < 0.5, "Less than half of due assessments submitted", "Offer study-skills session or deadline planning help"),
        ("clicks_last14", lambda r: float(r.get("clicks_last14", 99)) < float(r.get("clicks_prev14", 0)) * 0.5, "Engagement dropped sharply", "Tutor check-in call"),
    ],
    "student_performance": [
        ("failures", lambda r: float(r.get("failures", 0)) >= 1, "Previous class failures", "Remedial support / peer tutoring"),
        ("absences", lambda r: float(r.get("absences", 0)) >= 10, "High absences", "Attendance conversation with family and counsellor (supportive)"),
        ("studytime", lambda r: float(r.get("studytime", 4)) <= 1, "Very low weekly study time", "Study-habits workshop"),
    ],
    "xapi_performance": [
        ("raisedhands", lambda r: float(r.get("raisedhands", 99)) < 20, "Low class participation", "Teacher encourages participation; check for barriers"),
        ("VisITedResources", lambda r: float(r.get("VisITedResources", 99)) < 20, "Few learning resources visited", "Show student the available resources"),
        ("StudentAbsenceDays", lambda r: r.get("StudentAbsenceDays") == "Above-7", "7+ absence days", "Supportive attendance check-in"),
    ],
}
HIGH_CONCERN = {"uci_dropout": "Dropout", "oulad_withdrawal": "Withdrawn", "student_performance": "fail", "xapi_performance": "L"}


def suggest(dataset_id, record, prediction):
    concern = HIGH_CONCERN.get(dataset_id)
    p = prediction["probabilities"].get(concern, 0.0)
    actions = [{"indicator": col, "observation": obs, "suggested_support": act}
               for col, test, obs, act in RULES.get(dataset_id, []) if _safe(test, record)]
    return {"dataset_id": dataset_id, "concern_class": concern, "concern_probability": p,
            "status": "pending_staff_review", "suggested_actions": actions,
            "note": "Suggestions are supportive and need staff approval before any ticket is created. "
                    "Model output alone must never be used to punish, exclude or deny opportunities. "
                    "No causal claim is made about intervention effects."}


def _safe(test, r):
    try: return bool(test(r))
    except (TypeError, ValueError): return False


def decision_record(suggestion, staff_id, decision, notes=""):
    """Audit record to persist in your tickets DB: decision in {approved, modified, rejected}; follow-up outcome added later."""
    assert decision in {"approved", "modified", "rejected"}
    return {"suggestion": suggestion, "staff_id": staff_id, "decision": decision, "notes": notes,
            "decided_at": datetime.now(timezone.utc).isoformat(), "follow_up": [], "outcome": None}
