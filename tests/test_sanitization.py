"""
Regression and unit tests for evidence sanitization, customer PII/name stripping,
and grounded response generation safety.
"""

import pytest
from src.reply_generator import (
    sanitize_brand_response,
    sanitize_customer_message,
    GroundedReplyGenerator
)
from src.pipeline import SupportAgentPipeline


def test_sanitize_brand_response_removes_customer_names():
    # Various customer greeting and name patterns
    test_cases = [
        ("@530262 Helo Gill! Thanks for reaching out. Upon inspection here: https://t.co/test do you see multiple orders? ^TM",
         "Thanks for reaching out. Upon inspection here: https://t.co/test do you see multiple orders?"),
        ("@12345 Hi, Amy! We're here to help! Have you tried the steps here: https://t.co/test? ^TS",
         "We're here to help! Have you tried the steps here: https://t.co/test?"),
        ("Hello Sarah, please DM us your order number so we can investigate. ^AB",
         "Please DM us your order number so we can investigate."),
        ("Hey John! You can return items directly via Your Orders. ^MC",
         "You can return items directly via Your Orders."),
        ("Thanks David, please send us a private message. ^MO",
         "Please send us a private message.")
    ]
    
    for raw, expected in test_cases:
        cleaned = sanitize_brand_response(raw)
        assert "Gill" not in cleaned
        assert "Amy" not in cleaned
        assert "Sarah" not in cleaned
        assert "John" not in cleaned
        assert "David" not in cleaned
        assert "^" not in cleaned
        assert "@" not in cleaned
        assert cleaned == expected


def test_sanitize_brand_response_removes_handles_and_signatures():
    raw = "@530262 @AmazonHelp 1/2 We'd like to check this out with you in real-time! Link: https://t.co/abc ^GL"
    cleaned = sanitize_brand_response(raw)
    
    assert "@530262" not in cleaned
    assert "@AmazonHelp" not in cleaned
    assert "^GL" not in cleaned
    assert "1/2" not in cleaned
    assert "We'd like to check this out" in cleaned


def test_sanitize_customer_message():
    raw = "@AmazonHelp @366906 Hi I ordered one item but have been charged twice. Order: 123-4567890-1234567"
    cleaned = sanitize_customer_message(raw)
    
    assert "@AmazonHelp" not in cleaned
    assert "@366906" not in cleaned
    assert "123-4567890-1234567" not in cleaned
    assert "charged twice" in cleaned


def test_reply_generator_does_not_leak_customer_names():
    generator = GroundedReplyGenerator()
    evidence = [
        {
            "historical_customer_message": "@AmazonHelp charged twice!",
            "historical_brand_response": "@530262 Helo Gill! Thanks for reaching out. Upon inspection here: https://t.co/aaDyEz1VgE do you see multiple orders? ^TM",
            "similarity": 0.86
        }
    ]
    
    reply = generator.generate_reply(
        customer_message="I was charged twice for an item. Can you help?",
        predicted_intent="payment_billing_issue",
        retrieved_examples=evidence
    )
    
    assert "Gill" not in reply
    assert "@530262" not in reply
    assert "^TM" not in reply
    assert "https://t.co/aaDyEz1VgE" in reply


def test_pipeline_output_sanitizes_evidence_and_reply():
    pipeline = SupportAgentPipeline()
    query = "I was charged twice for an item I ordered once. Can you help me check the duplicate charge?"
    result = pipeline.process(customer_message=query)
    
    # Check reply
    assert "Gill" not in result["reply"]
    assert "@" not in result["reply"]
    assert "^" not in result["reply"]
    
    # Check evidence cards returned to UI
    for ex in result["retrieved_examples"]:
        assert "@" not in ex["historical_customer_message"]
        assert "@" not in ex["historical_brand_response"]
        assert "Gill" not in ex["historical_brand_response"]
        assert "^" not in ex["historical_brand_response"]


def test_ui_redaction_function():
    from app.app import redact_ui_text
    
    sample_text = "@530262 Hi Amy! We can check order 404-9391432-5602714 for you. Call 1234567890. ^TM"
    redacted = redact_ui_text(sample_text)
    
    assert "Amy" not in redacted
    assert "[Customer Name]" in redacted
    assert "404-9391432-5602714" not in redacted
    assert "[Order Number]" in redacted
    assert "1234567890" not in redacted
    assert "[Phone Number]" in redacted
    assert "^TM" not in redacted
    assert "@530262" not in redacted

