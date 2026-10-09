"""Telecaller-disposition catalogue and the "final disposition" rule.

A lead can carry several cfTelecallerDisposition ids. `select_final_disposition`
reduces them to the single disposition that represents the lead.
"""
from typing import Iterable, Optional

ALL_DISPOSITIONS = [
    {"internal_name": "EXPLORING_OPTIONS", "id": 2989604},
    {"internal_name": "NEED_MORE_INFORMATION", "id": 2989605},
    {"internal_name": "NEED_MORE_TIME", "id": 2989606},
    {"internal_name": "PARENT_DISCUSSION_PENDING", "id": 2989607},
    {"internal_name": "FEE_SENSITIVE", "id": 2989608},
    {"internal_name": "SCHOLARSHIP_INFO_REQUESTED", "id": 2989609},
    {"internal_name": "DISCOUNT_NEGOTIATION", "id": 2989610},
    {"internal_name": "FEE_REQUESTED", "id": 2989611},
    {"internal_name": "FEE_STRUCTURE_SHARED", "id": 2989612},
    {"internal_name": "PAYMENT_DISCUSSION", "id": 2989613},
    {"internal_name": "PAYMENT_LINK_SHARED", "id": 2989614},
    {"internal_name": "DEMO_REQUESTED", "id": 2989615},
    {"internal_name": "DEMO_ATTENDED", "id": 2989616},
    {"internal_name": "CENTRE_VISIT_REQUESTED", "id": 2989617},
    {"internal_name": "CENTRE_VISIT_SCHEDULED", "id": 2989618},
    {"internal_name": "CENTRE_VISIT_COMPLETED", "id": 2989619},
    {"internal_name": "EVALUATING_COMPETITORS", "id": 2989620},
    {"internal_name": "ALREADY_ENROLLED_ELSEWHERE", "id": 2989621},
    {"internal_name": "BATCH_TIMING_OBJECTION", "id": 2989622},
    {"internal_name": "DISTANCE_OBJECTION", "id": 2989623},
    {"internal_name": "TRUST_OBJECTION", "id": 2989624},
    {"internal_name": "CALL_BACK_LATER", "id": 2989625},
    {"internal_name": "NOT_REACHABLE", "id": 2989626},
    {"internal_name": "FOLLOW_UP_REQUIRED", "id": 2989627},
    {"internal_name": "BATCH_START_DATE_ASKED", "id": 2989628},
    {"internal_name": "FACULTY_INQUIRY", "id": 2989629},
    {"internal_name": "RESULTS_INQUIRY", "id": 2989630},
    {"internal_name": "COMPETITOR_RISK", "id": 2989631},
    {"internal_name": "BUDGET_RISK", "id": 2989632},
    {"internal_name": "DECISION_DELAY_RISK", "id": 2989633},
    {"internal_name": "ADMISSION_THIS_WEEK", "id": 2989634},
    {"internal_name": "ADMISSION_THIS_MONTH", "id": 2989635},
    {"internal_name": "FUTURE_BATCH_INTEREST", "id": 2989636},
    {"internal_name": "WRONG_NUMBER", "id": 2989637},
    {"internal_name": "DID_NOT_PICK_UP", "id": 2989638},
    {"internal_name": "REGISTRATION_DONE", "id": 2989639},
    {"internal_name": "INTERNAL_STUDENT", "id": 2989640},
    {"internal_name": "DISCUSS_AND_DECIDE", "id": 2989641},
    {"internal_name": "DISCONNECTED_WHILE_TALKING", "id": 2989642},
    {"internal_name": "C4S", "id": 2989643},
    {"internal_name": "SEMINAR_ATTENDED", "id": 2989644},
    {"internal_name": "C4T", "id": 2989645},
    {"internal_name": "TEST_ATTENDED", "id": 2989646},
    {"internal_name": "ADMISSION_DONE", "id": 2989647},
    {"internal_name": "CALL_BUSY", "id": 2990225},
    {"internal_name": "NOT_INTRESTED", "id": 2990573},
    {"internal_name": "SWITCH_OFF", "id": 2990574},
    {"internal_name": "MISTAKENLY_VISITED", "id": 2990575},
    {"internal_name": "COMING_FOR_ADMISSION", "id": 3010415},
    {"internal_name": "INTERESTED_FOR_ONLINE", "id": 3010416},
    {"internal_name": "TEST_DONE", "id": 3010417},
    {"internal_name": "STREAM_CHANGE", "id": 3010418},
]

# "Empty" dispositions: the call produced no real conversation.
EMPTY_DISPOSITIONS = [
    {"internal_name": "CALL_BACK_LATER", "id": 2989625},
    {"internal_name": "NOT_REACHABLE", "id": 2989626},
    {"internal_name": "DID_NOT_PICK_UP", "id": 2989638},
    {"internal_name": "DISCONNECTED_WHILE_TALKING", "id": 2989642},
    {"internal_name": "CALL_BUSY", "id": 2990225},
    {"internal_name": "SWITCH_OFF", "id": 2990574},
    {"internal_name": "WRONG_NUMBER", "id": 2989637},
]

_EMPTY_IDS = frozenset(d["id"] for d in EMPTY_DISPOSITIONS)

# Everything in the master list that is not empty.
UNEMPTY_DISPOSITIONS = [d for d in ALL_DISPOSITIONS if d["id"] not in _EMPTY_IDS]

_BY_ID = {d["id"]: d for d in ALL_DISPOSITIONS}


def select_final_disposition(disposition_ids: Optional[Iterable[int]]) -> Optional[dict]:
    """Pick the one disposition that represents a lead.

    - Any unempty disposition present: the unempty one with the highest id
      (unempty always beats empty, whatever the ids).
    - Only empty dispositions present: the empty one with the highest id.
    - Nothing tagged (None / empty): None.

    Ids that aren't in ALL_DISPOSITIONS can't be classified and are ignored.
    Returns {"internal_name": ..., "id": ...}.
    """
    known = [_BY_ID[i] for i in (disposition_ids or []) if i in _BY_ID]
    unempty = [d for d in known if d["id"] not in _EMPTY_IDS]
    pool = unempty or known
    return dict(max(pool, key=lambda d: d["id"])) if pool else None
