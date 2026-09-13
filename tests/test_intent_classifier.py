"""
Unit tests for Intent Classifiers.
"""

import pytest
import numpy as np
from src.intent_classifier import (
    MajorityBaselineClassifier,
    TfidfLogisticRegressionClassifier,
    SemanticIntentClassifier
)


def test_majority_baseline():
    clf = MajorityBaselineClassifier()
    texts = ["where is package", "refund my order", "damaged item", "where is package"]
    labels = ["delivery_tracking_delay", "refund_return_cancellation", "damaged_defective_item", "delivery_tracking_delay"]
    clf.fit(texts, labels)
    
    assert clf.majority_class == "delivery_tracking_delay"
    preds = clf.predict(["new query 1", "new query 2"])
    assert preds == ["delivery_tracking_delay", "delivery_tracking_delay"]


def test_tfidf_logistic_regression():
    clf = TfidfLogisticRegressionClassifier(max_features=100)
    texts = [
        "where is my package tracking delay",
        "track my delivery status shipment",
        "refund my order return item",
        "cancel subscription refund money"
    ]
    labels = ["delivery", "delivery", "refund", "refund"]
    clf.fit(texts, labels)
    
    pred = clf.predict(["where is my delivery package tracking"])
    assert pred[0] == "delivery"


def test_semantic_intent_classifier():
    clf = SemanticIntentClassifier()
    texts = [
        "my package has not arrived yet",
        "where is my delivery order",
        "i want to return this product for a refund",
        "please cancel this charge and refund me"
    ]
    labels = ["delivery", "delivery", "refund", "refund"]
    clf.fit(texts, labels, batch_size=4)
    
    pred, conf = clf.predict_single("where is my shipment parcel")
    assert pred in ["delivery", "refund"]
    assert 0.0 <= conf <= 1.0
