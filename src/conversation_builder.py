"""
Conversation Reconstruction Module for Customer Support Tweets Dataset.

This module provides modular, explainable functions to:
1. resolve_conversation_ids: Trace parent-child graph (in_response_to_tweet_id -> tweet_id)
   using memoized path compression to group tweets into unique conversation trees.
2. build_conversation_threads: Sequence conversation turns chronologically, assign turn numbers,
   speaker roles (customer vs brand), and identify the brand entity.
3. extract_customer_brand_pairs: Match brand responses to their preceding customer messages,
   compute response time in seconds, build historical thread context, and enforce temporal validity.
4. compute_brand_candidate_metrics: Calculate empirical support volume and response statistics
   per brand to support evidence-based brand selection in Phase 3.
5. export_processed_datasets: Save structured Parquet and CSV files under data/processed/.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np


def resolve_conversation_ids(df: pd.DataFrame) -> pd.Series:
    """
    Resolve the root tweet ID for each tweet in the dataset using graph path traversal.
    
    In Twitter conversations:
    - Root tweets have in_response_to_tweet_id == NaN (or reference a tweet not in dataset).
    - Child tweets have in_response_to_tweet_id pointing to their parent tweet.
    - Path compression with memoization ensures fast O(V+E) graph resolution.
    
    Args:
        df: DataFrame containing 'tweet_id' and 'in_response_to_tweet_id'.
        
    Returns:
        A pandas Series of integer conversation IDs aligned with df's index.
    """
    # Build fast parent lookup dictionary: child_id -> parent_id
    valid_parents = df.dropna(subset=['in_response_to_tweet_id'])
    
    # Cast to integer safely
    parent_map: Dict[int, int] = {}
    for tid, pid in zip(valid_parents['tweet_id'], valid_parents['in_response_to_tweet_id']):
        try:
            parent_map[int(tid)] = int(pid)
        except (ValueError, TypeError):
            continue
            
    # Set of known tweet IDs in current DataFrame
    known_tweet_ids = set(df['tweet_id'].astype(int))
    
    root_memo: Dict[int, int] = {}
    
    def find_root(tid: int) -> int:
        if tid in root_memo:
            return root_memo[tid]
            
        path: List[int] = []
        curr = tid
        visited = set()
        
        # Traverse upwards until reaching a root or dangling reference
        while curr in parent_map:
            parent = parent_map[curr]
            # Stop if parent is missing from dataset (dangling reference) or cycle detected
            if parent not in known_tweet_ids or curr in visited:
                break
            visited.add(curr)
            path.append(curr)
            curr = parent
            
        root = curr
        # Path compression: point all visited nodes directly to root
        for node in path:
            root_memo[node] = root
        root_memo[tid] = root
        return root

    # Vectorized / list comprehension mapping
    conversation_ids = [find_root(int(tid)) for tid in df['tweet_id']]
    return pd.Series(conversation_ids, index=df.index, name='conversation_id', dtype='int64')


def build_conversation_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct multi-turn conversation threads ordered chronologically.
    
    Adds:
    - conversation_id: Root ancestor tweet ID.
    - turn_number: 1-indexed turn position in the dialogue.
    - speaker_role: 'customer' (inbound=True) or 'brand' (inbound=False).
    - brand: Brand handle associated with the conversation thread.
    
    Args:
        df: Cleaned tweets DataFrame (output of preprocess_tweets).
        
    Returns:
        DataFrame sorted by [conversation_id, created_at_dt, tweet_id] with turn metadata.
    """
    threads_df = df.copy()
    
    # 1. Resolve conversation_id if not already present
    if 'conversation_id' not in threads_df.columns:
        threads_df['conversation_id'] = resolve_conversation_ids(threads_df)
        
    # 2. Assign speaker role
    threads_df['speaker_role'] = np.where(threads_df['inbound'], 'customer', 'brand')
    
    # 3. Sort chronologically within each conversation
    threads_df = threads_df.sort_values(
        by=['conversation_id', 'created_at_dt', 'tweet_id'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    # 4. Assign 1-indexed turn numbers per conversation
    threads_df['turn_number'] = threads_df.groupby('conversation_id').cumcount() + 1
    
    # 5. Resolve brand handle for each conversation
    # For outbound tweets, author_id is the brand. Group by conversation to find the brand.
    brand_per_conv = (
        threads_df[threads_df['speaker_role'] == 'brand']
        .groupby('conversation_id')['author_id']
        .first()
    )
    threads_df['brand'] = threads_df['conversation_id'].map(brand_per_conv).fillna('Unknown')
    
    return threads_df


def extract_customer_brand_pairs(
    threads_df: pd.DataFrame,
    max_context_turns: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract structured Customer Query -> Brand Response interaction pairs.
    
    Vectorized matching pairs each brand response (inbound=False) with its 
    immediate parent customer tweet (inbound=True).
    Computes response time and compiles prior dialogue turns as thread context.
    
    Args:
        threads_df: DataFrame generated by build_conversation_threads.
        max_context_turns: Maximum number of previous dialogue turns to include in context.
        
    Returns:
        Tuple of:
        - DataFrame of customer-brand interaction pairs.
        - Data quality metrics dictionary for pair extraction.
    """
    # 1. Filter inbound customer messages and outbound brand responses
    cust_df = threads_df[threads_df['inbound']].copy()
    brand_df = threads_df[(~threads_df['inbound']) & (threads_df['in_response_to_tweet_id'].notna())].copy()
    
    total_brand_responses = len(brand_df)
    
    # Cast linking ID safely to integer
    brand_df['parent_tweet_id'] = brand_df['in_response_to_tweet_id'].astype('int64')
    
    # 2. Fast vectorized inner merge on parent_tweet_id == tweet_id
    merged = pd.merge(
        cust_df,
        brand_df,
        left_on='tweet_id',
        right_on='parent_tweet_id',
        suffixes=('_customer', '_brand')
    )
    
    # Count brand-to-brand replies vs true dangling references
    brand_parents = set(brand_df['tweet_id'])
    cust_parents = set(cust_df['tweet_id'])
    
    brand_to_brand_count = 0
    dangling_count = 0
    for pid in brand_df['parent_tweet_id']:
        if pid in cust_parents:
            continue
        elif pid in brand_parents:
            brand_to_brand_count += 1
        else:
            dangling_count += 1
            
    # 3. Calculate response time in seconds and minutes
    merged['response_time_seconds'] = (merged['created_at_dt_brand'] - merged['created_at_dt_customer']).dt.total_seconds()
    
    # 4. Filter negative response times (temporal anomalies)
    valid_time_mask = merged['response_time_seconds'] >= 0
    negative_time_count = int((~valid_time_mask).sum())
    valid_merged = merged[valid_time_mask].copy()
    
    valid_merged['response_time_minutes'] = (valid_merged['response_time_seconds'] / 60.0).round(2)
    
    # 5. Build prior thread context
    # Multi-turn pairs (customer turn > 1) have prior context
    turn_str_map = dict(zip(
        zip(threads_df['conversation_id'], threads_df['turn_number']),
        "[" + threads_df['speaker_role'].str.capitalize() + " (" + threads_df['author_id'] + ")]: " + threads_df['text_clean']
    ))
    
    context_list = []
    for cid, c_turn in zip(valid_merged['conversation_id_customer'], valid_merged['turn_number_customer']):
        if c_turn <= 1:
            context_list.append("")
        else:
            start_t = max(1, c_turn - max_context_turns)
            prior_strs = [turn_str_map.get((cid, t)) for t in range(start_t, c_turn) if (cid, t) in turn_str_map]
            context_list.append(" \n ".join(prior_strs))
            
    valid_merged['thread_context'] = context_list
    valid_merged['has_prior_context'] = [bool(len(c) > 0) for c in context_list]
    
    # 6. Format final output columns
    result_df = pd.DataFrame({
        'conversation_id': valid_merged['conversation_id_customer'],
        'customer_turn_number': valid_merged['turn_number_customer'],
        'brand_turn_number': valid_merged['turn_number_brand'],
        'customer_tweet_id': valid_merged['tweet_id_customer'],
        'customer_author_id': valid_merged['author_id_customer'],
        'customer_created_at': valid_merged['created_at_customer'],
        'customer_text_raw': valid_merged['text_raw_customer'],
        'customer_text_clean': valid_merged['text_clean_customer'],
        'brand_tweet_id': valid_merged['tweet_id_brand'],
        'brand_author_id': valid_merged['author_id_brand'],
        'brand': valid_merged['author_id_brand'],
        'brand_created_at': valid_merged['created_at_brand'],
        'brand_text_raw': valid_merged['text_raw_brand'],
        'brand_text_clean': valid_merged['text_clean_brand'],
        'response_time_seconds': valid_merged['response_time_seconds'],
        'response_time_minutes': valid_merged['response_time_minutes'],
        'thread_context': valid_merged['thread_context'],
        'has_prior_context': valid_merged['has_prior_context']
    }).reset_index(drop=True)
    
    metrics = {
        "total_brand_responses_evaluated": total_brand_responses,
        "valid_customer_brand_pairs": len(result_df),
        "brand_to_brand_replies_excluded": brand_to_brand_count,
        "dangling_parent_references_excluded": dangling_count,
        "skipped_negative_response_time": negative_time_count,
        "unique_brands_in_pairs": int(result_df['brand'].nunique()) if not result_df.empty else 0,
        "unique_customers_in_pairs": int(result_df['customer_author_id'].nunique()) if not result_df.empty else 0,
        "unique_conversations_in_pairs": int(result_df['conversation_id'].nunique()) if not result_df.empty else 0,
    }
    
    return result_df, metrics


def compute_brand_candidate_metrics(
    pairs_df: pd.DataFrame,
    threads_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute empirical volume, response time, and conversation metrics per candidate brand.
    
    This provides measured evidence for Phase 3 brand selection.
    
    Args:
        pairs_df: Extracted customer-brand pairs DataFrame.
        threads_df: Reconstructed conversation threads DataFrame.
        
    Returns:
        DataFrame summarizing candidate brands sorted by pair count.
    """
    conv_lengths = threads_df.groupby('conversation_id').size()
    
    records = []
    for brand, b_pairs in pairs_df.groupby('brand'):
        b_cids = set(b_pairs['conversation_id'])
        b_conv_lens = conv_lengths.loc[conv_lengths.index.intersection(b_cids)]
        
        # Count brand responses and inbound mentions
        brand_responses_count = int((threads_df['author_id'] == brand).sum())
        
        records.append({
            'brand': brand,
            'brand_responses': brand_responses_count,
            'pair_count': len(b_pairs),
            'conversation_count': len(b_cids),
            'avg_conversation_length': round(float(b_conv_lens.mean()), 2) if len(b_conv_lens) > 0 else 0.0,
            'median_response_time_min': round(float(b_pairs['response_time_minutes'].median()), 2),
            'mean_response_time_min': round(float(b_pairs['response_time_minutes'].mean()), 2),
            'pairs_with_context_pct': round((b_pairs['has_prior_context'].sum() / len(b_pairs)) * 100, 2)
        })
        
    summary_df = pd.DataFrame(records).sort_values(by='pair_count', ascending=False).reset_index(drop=True)
    return summary_df


def export_processed_datasets(
    pairs_df: pd.DataFrame,
    threads_df: pd.DataFrame,
    output_dir: str = "data/processed",
    save_parquet: bool = True,
    save_csv: bool = True
) -> Dict[str, str]:
    """
    Export reconstructed datasets to data/processed in Parquet and/or CSV formats.
    
    Args:
        pairs_df: DataFrame of Customer Query -> Brand Response pairs.
        threads_df: DataFrame of chronological multi-turn conversation threads.
        output_dir: Output directory path.
        save_parquet: If True, saves .parquet files (fast, compact).
        save_csv: If True, saves .csv files.
        
    Returns:
        Dictionary of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = {}
    
    # 1. Pairs dataset
    if save_parquet:
        pairs_parquet = os.path.join(output_dir, "customer_brand_pairs.parquet")
        pairs_df.to_parquet(pairs_parquet, index=False)
        created_files["pairs_parquet"] = pairs_parquet
        
    if save_csv:
        pairs_csv = os.path.join(output_dir, "customer_brand_pairs.csv")
        pairs_df.to_csv(pairs_csv, index=False)
        created_files["pairs_csv"] = pairs_csv
        
    # 2. Threads dataset
    if save_parquet:
        threads_parquet = os.path.join(output_dir, "conversation_threads.parquet")
        threads_df.to_parquet(threads_parquet, index=False)
        created_files["threads_parquet"] = threads_parquet
        
    if save_csv:
        threads_csv = os.path.join(output_dir, "conversation_threads.csv")
        threads_df.to_csv(threads_csv, index=False)
        created_files["threads_csv"] = threads_csv
        
    return created_files
