"""
Unit tests for data preprocessing and text normalization.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_tweet_text, parse_twitter_timestamp


def test_clean_tweet_text():
    raw = "Hey &amp; @AmazonHelp!   where is my package??? \u200b"
    clean = clean_tweet_text(raw)
    assert "@AmazonHelp" in clean  # Mentions preserved
    assert "&" in clean  # HTML unescaped
    assert "where is my package???" in clean  # Punctuation preserved
    assert "   " not in clean  # Multiple whitespaces collapsed


def test_clean_tweet_text_empty():
    assert clean_tweet_text("") == ""
    assert clean_tweet_text(None) == ""


def test_parse_twitter_timestamp():
    s = pd.Series(["Tue Nov 07 15:43:00 +0000 2017", "invalid_date"])
    parsed = parse_twitter_timestamp(s)
    assert not pd.isna(parsed.iloc[0])
    assert parsed.iloc[0].year == 2017
    assert parsed.iloc[0].month == 11
    assert parsed.iloc[0].day == 7
    assert pd.isna(parsed.iloc[1])

