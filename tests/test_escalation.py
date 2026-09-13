"""
Unit tests for Escalation Router.
"""

import pytest
from src.escalation import EscalationRouter


def test_escalation_sensitive_keyword():
    router = EscalationRouter()
    # Customer explicitly asks for lawyer/legal action
    res = router.evaluate(
        customer_message="I am taking legal action and filing a complaint with my lawyer.",
        predicted_intent="general_inquiry_support",
        intent_confidence=0.95,
        retrieved_examples=[{"similarity": 0.85}]
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "lawyer" in res["reason"].lower() or "legal" in res["reason"].lower()


def test_escalation_security_intent():
    router = EscalationRouter()
    # Account access & security intent triggers automatic human verification routing
    res = router.evaluate(
        customer_message="I cannot log in to my account, password was changed.",
        predicted_intent="account_access_security",
        intent_confidence=0.90,
        retrieved_examples=[{"similarity": 0.85}]
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "security" in res["reason"].lower() or "account" in res["reason"].lower()


def test_escalation_low_confidence():
    router = EscalationRouter(intent_confidence_threshold=0.65)
    # Low classifier confidence
    res = router.evaluate(
        customer_message="Random garbled text qwerty 123",
        predicted_intent="delivery_tracking_delay",
        intent_confidence=0.45,
        retrieved_examples=[{"similarity": 0.75}]
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "low intent" in res["reason"].lower() or "confidence" in res["reason"].lower()


def test_escalation_weak_retrieval():
    router = EscalationRouter(retrieval_similarity_threshold=0.60)
    # High confidence but novel scenario with weak historical evidence
    res = router.evaluate(
        customer_message="Can you deliver to a remote Antarctic research station via drone?",
        predicted_intent="delivery_tracking_delay",
        intent_confidence=0.90,
        retrieved_examples=[{"similarity": 0.42}]
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert "weak historical retrieval" in res["reason"].lower() or "evidence" in res["reason"].lower()


def test_escalation_auto_handle_success():
    router = EscalationRouter()
    # Standard query with high confidence and strong historical resolution
    res = router.evaluate(
        customer_message="Where is my package? It was supposed to arrive today.",
        predicted_intent="delivery_tracking_delay",
        intent_confidence=0.88,
        retrieved_examples=[{"similarity": 0.82}]
    )
    assert res["decision"] == "AUTO_HANDLE"
