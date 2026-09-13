"""
Preprocessing Module for Customer Support Tweets Dataset.

This module provides simple, modular functions to clean raw Twitter support data:
1. clean_tweet_text: Normalizes excessive whitespace and unescapes HTML entities 
   while strictly preserving emojis, punctuation, URLs, @mentions, and casing.
2. parse_twitter_timestamp: Fast, vectorized parsing of Twitter's RFC 2822 timestamps 
   into timezone-aware UTC datetime objects.
3. preprocess_tweets: Standardizes types, removes exact duplicates, cleans text into 
   'text_clean' while preserving 'text_raw', and returns cleaned data alongside 
   empirical quality metrics.
"""

import html
import re
from typing import Dict, Any, Tuple
import pandas as pd


def clean_tweet_text(text: str) -> str:
    """
    Clean and normalize tweet text without removing linguistic signals.
    
    Operations performed:
    - Converts input to string and strips leading/trailing whitespace.
    - Unescapes standard HTML entities (e.g. '&amp;' -> '&', '&lt;' -> '<').
    - Removes zero-width characters / non-printable control characters.
    - Collapses multiple consecutive whitespaces and newlines into single spaces.
    
    Signals strictly PRESERVED:
    - Emojis (e.g., 😡, 🙏, 😊)
    - Punctuation and question marks (e.g., '???', '!', '.')
    - Brand mentions (e.g., '@AmazonHelp', '@AppleSupport')
    - URLs and order numbers / alphanumeric codes
    - Case / capitalization (e.g., 'HELP ME PLEASE')
    
    Args:
        text: Raw text string of the tweet.
        
    Returns:
        Cleaned text string. Returns empty string if input is None or NaN.
    """
    if text is None or pd.isna(text):
        return ""
        
    text_str = str(text)
    
    # Unescape HTML entities (common in scraped/API Twitter text)
    text_str = html.unescape(text_str)
    
    # Remove zero-width spaces and control characters (except standard whitespace)
    text_str = re.sub(r'[\u200b-\u200f\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]', '', text_str)
    
    # Normalize multiple whitespace characters (spaces, tabs, newlines) into a single space
    text_str = re.sub(r'\s+', ' ', text_str).strip()
    
    return text_str


def parse_twitter_timestamp(ts_series: pd.Series) -> pd.Series:
    """
    Parse Twitter format timestamps into pandas datetime series.
    
    Expected format: '%a %b %d %H:%M:%S %z %Y' (e.g., 'Tue Oct 31 22:10:47 +0000 2017').
    
    Args:
        ts_series: A pandas Series containing Twitter timestamp strings.
        
    Returns:
        A pandas Series of UTC datetime objects (errors coerced to NaT).
    """
    return pd.to_datetime(ts_series, format='%a %b %d %H:%M:%S %z %Y', errors='coerce', utc=True)


def preprocess_tweets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess raw tweets DataFrame and compute data-cleaning quality metrics.
    
    Steps:
    1. Records initial count.
    2. Identifies and removes exact duplicate rows (if any).
    3. Handles missing or empty text records.
    4. Parses timestamps into 'created_at_dt' datetime column.
    5. Creates 'text_clean' column while keeping original 'text' as 'text_raw'.
    6. Ensures standard types for tweet_id and in_response_to_tweet_id.
    
    Args:
        df: Raw pandas DataFrame loaded from twcs.csv.
        
    Returns:
        Tuple of:
        - Cleaned pandas DataFrame.
        - Dictionary of empirical cleaning metrics.
    """
    initial_rows = len(df)
    
    # 1. Exact duplicate rows
    duplicate_rows_count = int(df.duplicated().sum())
    cleaned_df = df.drop_duplicates().copy()
    
    # 2. Standardize column names if needed
    if 'text' in cleaned_df.columns and 'text_raw' not in cleaned_df.columns:
        cleaned_df['text_raw'] = cleaned_df['text'].fillna('').astype(str)
    
    # 3. Clean text column
    cleaned_df['text_clean'] = cleaned_df['text_raw'].apply(clean_tweet_text)
    
    # 4. Filter empty text records
    empty_text_mask = cleaned_df['text_clean'] == ''
    empty_text_count = int(empty_text_mask.sum())
    cleaned_df = cleaned_df[~empty_text_mask].copy()
    
    # 5. Parse timestamps
    cleaned_df['created_at_dt'] = parse_twitter_timestamp(cleaned_df['created_at'])
    invalid_ts_count = int(cleaned_df['created_at_dt'].isna().sum())
    cleaned_df = cleaned_df[cleaned_df['created_at_dt'].notna()].copy()
    
    # 6. ID standardization
    cleaned_df['tweet_id'] = cleaned_df['tweet_id'].astype('int64')
    cleaned_df['author_id'] = cleaned_df['author_id'].astype(str)
    cleaned_df['inbound'] = cleaned_df['inbound'].astype(bool)
    
    # 7. Check self-referencing links (where tweet is in response to itself)
    self_referencing_mask = cleaned_df['in_response_to_tweet_id'] == cleaned_df['tweet_id']
    self_referencing_count = int(self_referencing_mask.sum())
    if self_referencing_count > 0:
        cleaned_df.loc[self_referencing_mask, 'in_response_to_tweet_id'] = None
        
    final_rows = len(cleaned_df)
    
    metrics = {
        "initial_records": initial_rows,
        "exact_duplicates_removed": duplicate_rows_count,
        "empty_text_records_removed": empty_text_count,
        "invalid_timestamps_removed": invalid_ts_count,
        "self_referencing_links_sanitized": self_referencing_count,
        "cleaned_records": final_rows,
        "retention_percentage": round((final_rows / initial_rows) * 100, 2) if initial_rows > 0 else 0.0,
    }
    
    return cleaned_df, metrics
