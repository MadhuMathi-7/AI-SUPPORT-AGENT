"""
Unit tests for Evaluation package entrypoints and function exports.
"""

import pytest


def test_evaluate_intents_import():
    # Verify exact import expected by run_pipeline.py
    from evaluation.evaluate_intents import evaluate_all_intent_classifiers, run_intent_evaluation
    assert callable(evaluate_all_intent_classifiers)
    assert callable(run_intent_evaluation)
    assert evaluate_all_intent_classifiers == run_intent_evaluation


def test_evaluate_replies_import():
    from evaluation.evaluate_replies import evaluate_replies
    assert callable(evaluate_replies)


def test_evaluate_escalation_import():
    from evaluation.evaluate_escalation import evaluate_escalation
    assert callable(evaluate_escalation)


def test_llm_judge_import():
    from evaluation.llm_judge import LLMJudge, run_llm_judge_evaluation
    assert callable(run_llm_judge_evaluation)
    judge = LLMJudge()
    assert hasattr(judge, "evaluate_single")


def test_human_judge_agreement_import():
    from evaluation.human_judge_agreement import evaluate_human_judge_agreement
    assert callable(evaluate_human_judge_agreement)
