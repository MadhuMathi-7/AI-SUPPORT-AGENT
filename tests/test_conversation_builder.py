"""
Unit tests for data cleaning and conversation reconstruction modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_tweet_text, parse_twitter_timestamp, preprocess_tweets
from src.conversation_builder import (
    resolve_conversation_ids,
    build_conversation_threads,
    extract_customer_brand_pairs,
    compute_brand_candidate_metrics
)


def test_clean_tweet_text():
    """Test text cleaning preserves emojis, punctuation, URLs, mentions, while cleaning whitespace."""
    # Preserves emojis, punctuation, capitalization, @mentions, URLs
    raw_text = "  @AmazonHelp   Where is my order #12345?? 😡  Need it ASAP!! https://amzn.to/track \n\n Thanks.  "
    cleaned = clean_tweet_text(raw_text)
    
    assert "@AmazonHelp" in cleaned
    assert "#12345??" in cleaned
    assert "😡" in cleaned
    assert "ASAP!!" in cleaned
    assert "https://amzn.to/track" in cleaned
    assert "\n" not in cleaned
    assert "   " not in cleaned
    assert cleaned.startswith("@AmazonHelp")
    assert cleaned.endswith("Thanks.")
    
    # HTML unescaping
    html_text = "I &amp; my friend need help &lt;urgent&gt;"
    assert clean_tweet_text(html_text) == "I & my friend need help <urgent>"
    
    # Empty / None handling
    assert clean_tweet_text("") == ""
    assert clean_tweet_text(None) == ""


def test_parse_twitter_timestamp():
    """Test parsing of Twitter RFC 2822 timestamps into datetime."""
    sample_series = pd.Series([
        "Tue Oct 31 22:10:47 +0000 2017",
        "Wed Nov 01 08:15:00 +0000 2017",
        "INVALID_TIMESTAMP_STRING"
    ])
    parsed = parse_twitter_timestamp(sample_series)
    
    assert pd.notna(parsed.iloc[0])
    assert pd.notna(parsed.iloc[1])
    assert pd.isna(parsed.iloc[2]) # Coerced to NaT
    assert parsed.iloc[0].year == 2017
    assert parsed.iloc[0].month == 10
    assert parsed.iloc[0].day == 31


def test_preprocess_tweets():
    """Test preprocess_tweets handles duplicates, empty text, and ID types."""
    raw_df = pd.DataFrame([
        {
            "tweet_id": 1,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "My internet is down!",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None
        },
        # Duplicate row of 1
        {
            "tweet_id": 1,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "My internet is down!",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None
        },
        # Empty text row
        {
            "tweet_id": 99,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "   ",
            "response_tweet_id": None,
            "in_response_to_tweet_id": None
        }
    ])
    
    cleaned_df, metrics = preprocess_tweets(raw_df)
    
    assert metrics["initial_records"] == 3
    assert metrics["exact_duplicates_removed"] == 1
    assert metrics["empty_text_records_removed"] == 1
    assert metrics["cleaned_records"] == 1
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["tweet_id"] == 1
    assert "text_clean" in cleaned_df.columns
    assert "text_raw" in cleaned_df.columns


def test_conversation_graph_and_thread_reconstruction():
    """Test multi-turn conversation graph traversal, chronological ordering, and role assignment."""
    # Synthetic conversation matching twcs schema:
    # Tweet 10 (Cust) -> Tweet 11 (Brand reply to 10) -> Tweet 12 (Cust reply to 11) -> Tweet 13 (Brand reply to 12)
    records = [
        # Tweet 13 (Brand response 2)
        {
            "tweet_id": 13,
            "author_id": "sprintcare",
            "inbound": False,
            "created_at": "Tue Oct 31 22:15:00 +0000 2017",
            "text": "@115712 Your ticket has been created.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 12
        },
        # Tweet 10 (Root customer message)
        {
            "tweet_id": 10,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "@sprintcare I need help with my account.",
            "response_tweet_id": "11",
            "in_response_to_tweet_id": None
        },
        # Tweet 12 (Customer follow-up)
        {
            "tweet_id": 12,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:00 +0000 2017",
            "text": "@sprintcare Sent the details.",
            "response_tweet_id": "13",
            "in_response_to_tweet_id": 11
        },
        # Tweet 11 (Brand response 1)
        {
            "tweet_id": 11,
            "author_id": "sprintcare",
            "inbound": False,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "@115712 Please DM your phone number.",
            "response_tweet_id": "12",
            "in_response_to_tweet_id": 10
        },
    ]
    df = pd.DataFrame(records)
    df_cleaned, _ = preprocess_tweets(df)
    threads_df = build_conversation_threads(df_cleaned)
    
    # Check conversation ID resolution
    assert (threads_df['conversation_id'] == 10).all()
    
    # Check chronological ordering
    assert list(threads_df['tweet_id']) == [10, 11, 12, 13]
    assert list(threads_df['turn_number']) == [1, 2, 3, 4]
    assert list(threads_df['speaker_role']) == ['customer', 'brand', 'customer', 'brand']
    assert (threads_df['brand'] == 'sprintcare').all()


def test_customer_brand_pair_extraction():
    """Test extracting customer-brand interaction pairs and thread context."""
    records = [
        {
            "tweet_id": 10,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "@sprintcare I need help with my account.",
            "response_tweet_id": "11",
            "in_response_to_tweet_id": None
        },
        {
            "tweet_id": 11,
            "author_id": "sprintcare",
            "inbound": False,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "@115712 Please DM your phone number.",
            "response_tweet_id": "12",
            "in_response_to_tweet_id": 10
        },
        {
            "tweet_id": 12,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:00 +0000 2017",
            "text": "@sprintcare Sent the details.",
            "response_tweet_id": "13",
            "in_response_to_tweet_id": 11
        },
        {
            "tweet_id": 13,
            "author_id": "sprintcare",
            "inbound": False,
            "created_at": "Tue Oct 31 22:15:00 +0000 2017",
            "text": "@115712 Your ticket has been created.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 12
        },
    ]
    df = pd.DataFrame(records)
    df_cleaned, _ = preprocess_tweets(df)
    threads_df = build_conversation_threads(df_cleaned)
    pairs_df, metrics = extract_customer_brand_pairs(threads_df)
    
    assert len(pairs_df) == 2
    assert metrics['valid_customer_brand_pairs'] == 2
    
    # Pair 1: Tweet 10 -> Tweet 11
    p1 = pairs_df.iloc[0]
    assert p1['customer_tweet_id'] == 10
    assert p1['brand_tweet_id'] == 11
    assert p1['brand'] == 'sprintcare'
    assert p1['response_time_seconds'] == 300.0  # 5 minutes
    assert p1['response_time_minutes'] == 5.0
    assert p1['has_prior_context'] == False
    
    # Pair 2: Tweet 12 -> Tweet 13
    p2 = pairs_df.iloc[1]
    assert p2['customer_tweet_id'] == 12
    assert p2['brand_tweet_id'] == 13
    assert p2['response_time_seconds'] == 300.0
    assert p2['has_prior_context'] == True
    assert "[Customer (115712)]" in p2['thread_context']
    assert "[Brand (sprintcare)]" in p2['thread_context']


def test_dangling_parent_and_negative_time_filtering():
    """Test that dangling references and negative response times are safely handled."""
    records = [
        # Normal customer tweet
        {
            "tweet_id": 10,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "@AppleSupport My phone is frozen.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": None
        },
        # Brand response with timestamp BEFORE customer message (anomaly)
        {
            "tweet_id": 11,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Tue Oct 31 21:50:00 +0000 2017", # 10 mins before!
            "text": "@115712 Restart your device.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 10
        },
        # Brand response to a missing parent (dangling)
        {
            "tweet_id": 999,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Tue Oct 31 22:30:00 +0000 2017",
            "text": "@115712 Follow up.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 888 # 888 is not in dataset
        }
    ]
    df = pd.DataFrame(records)
    df_cleaned, _ = preprocess_tweets(df)
    threads_df = build_conversation_threads(df_cleaned)
    pairs_df, metrics = extract_customer_brand_pairs(threads_df)
    
    assert metrics['skipped_negative_response_time'] == 1
    assert metrics['dangling_parent_references_excluded'] == 1  # 1 dangling parent (888 not in dataset)
    assert metrics['brand_to_brand_replies_excluded'] == 0
    assert len(pairs_df) == 0
