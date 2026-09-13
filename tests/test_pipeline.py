"""
Unit and integration tests for SupportAgentPipeline.
"""

import pytest
from src.pipeline import SupportAgentPipeline


@pytest.fixture(scope="module")
def pipeline():
    return SupportAgentPipeline()


def test_pipeline_process_structure(pipeline):
    query = "My package has not arrived yet. Tracking says Tracy CA."
    result = pipeline.process(customer_message=query)
    
    assert "customer_message" in result
    assert "intent" in result
    assert "intent_confidence" in result
    assert "retrieved_examples" in result
    assert "reply" in result
    assert "decision" in result
    assert "reason" in result
    
    assert result["decision"] in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    assert len(result["reply"]) > 5
    assert len(result["retrieved_examples"]) <= 3


def test_pipeline_escalation_routing(pipeline):
    # Sensitive legal query
    query = "I will contact my attorney and sue your company for this fraud!"
    result = pipeline.process(customer_message=query)
    
    assert result["decision"] == "ESCALATE_TO_HUMAN"
    assert "sensitive" in result["reason"].lower() or "attorney" in result["reason"].lower() or "fraud" in result["reason"].lower()
