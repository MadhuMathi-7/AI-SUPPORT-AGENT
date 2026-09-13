"""
Unit tests for Historical Retrieval Engine, Leakage Prevention, and Quality Guardrails.
"""

import pytest
import pandas as pd
import numpy as np
from src.retrieval import HistoricalRetrievalEngine


def test_retrieval_engine_leakage_filter(tmp_path):
    cache_path = str(tmp_path / "test_index.joblib")
    retriever = HistoricalRetrievalEngine(index_cache_path=cache_path)
    
    # Create mock historical pairs
    mock_pairs = pd.DataFrame([
        {
            "conversation_id": 1001,
            "customer_tweet_id": 1,
            "brand_tweet_id": 2,
            "customer_text_clean": "where is my book delivery",
            "brand_text_clean": "Please check your tracking link in orders.",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 5.0
        },
        {
            "conversation_id": 1002,
            "customer_tweet_id": 3,
            "brand_tweet_id": 4,
            "customer_text_clean": "where is my book delivery exact query",
            "brand_text_clean": "We are looking into your book order.",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 10.0
        },
        {
            "conversation_id": 1003,
            "customer_tweet_id": 5,
            "brand_tweet_id": 6,
            "customer_text_clean": "i need a refund for my headphones",
            "brand_text_clean": "You can return your item via the returns center.",
            "intent": "refund_return_cancellation",
            "response_time_minutes": 2.0
        }
    ])
    
    # Encode mock embeddings
    texts = mock_pairs['customer_text_clean'].tolist()
    embs = retriever.encoder.encode(texts, normalize_embeddings=True)
    retriever.pairs_df = mock_pairs
    retriever.embeddings = np.array(embs, dtype=np.float32)
    
    # Retrieve without filter: conv 1001 or 1002 should be top 1
    res1 = retriever.retrieve("where is my book delivery", top_k=2, min_similarity_threshold=0.0)
    assert len(res1) == 2
    assert res1[0]["conversation_id"] in [1001, 1002]
    
    # Retrieve WITH leakage exclusion for conversation 1002:
    res2 = retriever.retrieve("where is my book delivery", top_k=2, exclude_conversation_id=1002, min_similarity_threshold=0.0)
    retrieved_conv_ids = [r["conversation_id"] for r in res2]
    assert 1002 not in retrieved_conv_ids


def test_retrieval_engine_fragment_and_pii_filtering(tmp_path):
    cache_path = str(tmp_path / "test_index_filter.joblib")
    retriever = HistoricalRetrievalEngine(index_cache_path=cache_path)
    
    mock_pairs = pd.DataFrame([
        {
            "conversation_id": 2001,
            "customer_tweet_id": 10,
            "brand_tweet_id": 11,
            "customer_text_clean": "My package has not arrived yet delivery status",
            "brand_text_clean": "your details as we consider them to be personal information. Our Twitter page is public. 2/2",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 3.0
        },
        {
            "conversation_id": 2002,
            "customer_tweet_id": 12,
            "brand_tweet_id": 13,
            "customer_text_clean": "My package has not arrived yet delivery status",
            "brand_text_clean": "Please track your shipment in 'Your Orders' with your tracking ID.",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 5.0
        }
    ])
    
    texts = mock_pairs['customer_text_clean'].tolist()
    embs = retriever.encoder.encode(texts, normalize_embeddings=True)
    retriever.pairs_df = mock_pairs
    retriever.embeddings = np.array(embs, dtype=np.float32)
    
    res = retriever.retrieve("My package has not arrived yet delivery status", top_k=1, min_similarity_threshold=0.0)
    assert len(res) == 1
    # Must retrieve the valid resolution, NOT the 2/2 PII warning
    assert res[0]["conversation_id"] == 2002
    assert "Your Orders" in res[0]["historical_brand_response"]


def test_retrieval_engine_intent_constraining(tmp_path):
    cache_path = str(tmp_path / "test_index_intent.joblib")
    retriever = HistoricalRetrievalEngine(index_cache_path=cache_path)
    
    mock_pairs = pd.DataFrame([
        {
            "conversation_id": 3001,
            "customer_tweet_id": 20,
            "brand_tweet_id": 21,
            "customer_text_clean": "Where is my package tracking shipment?",
            "brand_text_clean": "Check the delivery status in your orders tab.",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 4.0
        },
        {
            "conversation_id": 3002,
            "customer_tweet_id": 22,
            "brand_tweet_id": 23,
            "customer_text_clean": "I need a refund for my order item.",
            "brand_text_clean": "You can request a return label via the online returns center.",
            "intent": "refund_return_cancellation",
            "response_time_minutes": 6.0
        }
    ])
    
    texts = mock_pairs['customer_text_clean'].tolist()
    embs = retriever.encoder.encode(texts, normalize_embeddings=True)
    retriever.pairs_df = mock_pairs
    retriever.embeddings = np.array(embs, dtype=np.float32)
    
    # Query with matching intent delivery_tracking_delay
    res_delivery = retriever.retrieve("Where is my package", intent="delivery_tracking_delay", min_similarity_threshold=0.0)
    assert len(res_delivery) == 1
    assert res_delivery[0]["intent"] == "delivery_tracking_delay"
    assert res_delivery[0]["conversation_id"] == 3001
    
    # Query with mismatched intent filter: refund intent specified for delivery query
    # The refund pool candidate should be evaluated, but since delivery candidate is excluded by intent filter,
    # if threshold is 0.70 (since refund candidate similarity to delivery query is lower), it returns 0 candidates
    res_mismatch = retriever.retrieve("Where is my package", intent="account_access_security", min_similarity_threshold=0.0)
    assert len(res_mismatch) == 0


def test_retrieval_engine_minimum_similarity_threshold(tmp_path):
    cache_path = str(tmp_path / "test_index_threshold.joblib")
    retriever = HistoricalRetrievalEngine(index_cache_path=cache_path)
    
    mock_pairs = pd.DataFrame([
        {
            "conversation_id": 4001,
            "customer_tweet_id": 30,
            "brand_tweet_id": 31,
            "customer_text_clean": "Where is my package tracking shipment?",
            "brand_text_clean": "Check the delivery status in your orders tab.",
            "intent": "delivery_tracking_delay",
            "response_time_minutes": 4.0
        }
    ])
    
    texts = mock_pairs['customer_text_clean'].tolist()
    embs = retriever.encoder.encode(texts, normalize_embeddings=True)
    retriever.pairs_df = mock_pairs
    retriever.embeddings = np.array(embs, dtype=np.float32)
    
    # Irrelevant query with high similarity threshold -> returns empty list
    res_irrelevant = retriever.retrieve(
        "quantum physics supercomputing dark matter",
        min_similarity_threshold=0.60
    )
    assert res_irrelevant == []
