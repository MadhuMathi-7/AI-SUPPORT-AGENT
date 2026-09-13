"""
Historical Response Retrieval Module for Customer Support System.

Provides semantic similarity search over historical customer-brand interactions:
1. Indexes historical training pairs (Customer Query -> Brand Response).
2. Filters out truncated fragments (e.g. '2/2', '1/2' tails) and generic PII warnings to ensure high-quality resolution evidence.
3. Encodes customer queries into 384-dimensional dense semantic vectors using Sentence Transformers.
4. Performs strict candidate filtering by predicted intent before ranking.
5. Enforces configurable minimum similarity threshold (returns [] if evidence is weak).
6. Enforces strict conversation-level leakage prevention: excludes turns from the query's own conversation_id.
"""

import os
import re
import joblib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Patterns for filtering out non-resolution fragments and PII notices
FRAGMENT_PATTERN = re.compile(
    r'(?:\b\d+/\d+\b|\[\d+/\d+\]|\(\d+/\d+\))',
    re.IGNORECASE
)
PII_WARNING_PATTERN = re.compile(
    r'personal information|page is public|public page|delete (?:your )?tweet|remove your (?:order|phone|personal|details|number)',
    re.IGNORECASE
)


class HistoricalRetrievalEngine:
    """
    Semantic retrieval engine for retrieving grounded historical support interactions.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        index_cache_path: str = "models/retrieval_index.joblib",
        default_min_similarity_threshold: float = 0.60
    ):
        self.model_name = model_name
        self.device = device
        self.index_cache_path = index_cache_path
        self.default_min_similarity_threshold = default_min_similarity_threshold
        self.encoder = SentenceTransformer(model_name, device=device)
        self.pairs_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        
    def build_index(
        self,
        train_pairs_path: str = "data/splits/train_pairs.parquet",
        max_index_size: int = 25000,
        random_seed: int = 42,
        force_rebuild: bool = False
    ):
        """
        Build and cache the semantic vector index from the training partition.
        Applies quality filters to remove fragments and PII moderation tweets.
        """
        if not force_rebuild and os.path.exists(self.index_cache_path):
            print(f"[RETRIEVAL] Loading cached retrieval index from: {self.index_cache_path}")
            cached_data = joblib.load(self.index_cache_path)
            self.pairs_df = cached_data["pairs_df"]
            self.embeddings = cached_data["embeddings"]
            print(f"[RETRIEVAL] Loaded {len(self.pairs_df):,} indexed historical interactions.")
            return self
            
        print(f"[RETRIEVAL] Building high-quality semantic index from: {train_pairs_path}")
        df = pd.read_parquet(train_pairs_path)
        
        # 1. Quality Filter: Remove fragment replies and generic PII warnings
        brand_texts = df['brand_text_clean'].fillna('')
        is_fragment = brand_texts.str.contains(FRAGMENT_PATTERN, regex=True)
        is_pii = brand_texts.str.contains(PII_WARNING_PATTERN, regex=True)
        is_too_short = brand_texts.str.len() < 25
        
        clean_mask = (~is_fragment) & (~is_pii) & (~is_too_short)
        clean_df = df[clean_mask].copy().reset_index(drop=True)
        print(f"[RETRIEVAL] Filtered raw training pool from {len(df):,} to {len(clean_df):,} quality resolution pairs.")
        
        # 2. Stratified Sampling across intents for balanced resolution coverage
        if len(clean_df) > max_index_size:
            sampled_dfs = []
            per_intent_target = max_index_size // max(1, clean_df['intent'].nunique())
            for intent, group in clean_df.groupby('intent'):
                n_sample = min(len(group), per_intent_target)
                sampled_dfs.append(group.sample(n=n_sample, random_state=random_seed))
                
            indexed_df = pd.concat(sampled_dfs).reset_index(drop=True)
            # If still below max_index_size, top up randomly from remaining pool
            if len(indexed_df) < max_index_size:
                remaining_pool = clean_df[~clean_df.index.isin(indexed_df.index)]
                needed = max_index_size - len(indexed_df)
                if len(remaining_pool) > 0:
                    top_up = remaining_pool.sample(n=min(needed, len(remaining_pool)), random_state=random_seed)
                    indexed_df = pd.concat([indexed_df, top_up]).reset_index(drop=True)
        else:
            indexed_df = clean_df.reset_index(drop=True)
            
        print(f"[RETRIEVAL] Encoding {len(indexed_df):,} customer query vectors with {self.model_name}...")
        texts = indexed_df['customer_text_clean'].tolist()
        embeddings = self.encoder.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        self.pairs_df = indexed_df
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        # Cache index
        os.makedirs(os.path.dirname(self.index_cache_path), exist_ok=True)
        joblib.dump({"pairs_df": self.pairs_df, "embeddings": self.embeddings}, self.index_cache_path)
        print(f"[RETRIEVAL] Cached index ({len(self.pairs_df):,} items) to: {self.index_cache_path}")
        return self
        
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_conversation_id: Optional[int] = None,
        min_similarity_threshold: Optional[float] = None,
        intent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k semantically similar historical customer interactions.
        
        Guarantees:
        1. Intent Filtering: When intent is specified, only candidates matching that intent are considered.
        2. Leakage Protection: Excludes historical records matching exclude_conversation_id.
        3. Quality Filtering: Excludes multi-part fragments and PII moderation disclaimers.
        4. Relevance Threshold: Candidates with true cosine similarity < min_similarity_threshold are excluded.
           If no candidate qualifies, returns an empty list [].
        5. Reports exact true cosine similarity score without artificial modification.
        """
        if self.embeddings is None or self.pairs_df is None:
            raise ValueError("Retrieval index is not built or loaded.")
            
        if min_similarity_threshold is None:
            min_similarity_threshold = self.default_min_similarity_threshold
            
        # Encode query into normalized vector
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        
        # Compute exact cosine similarity: dot product of normalized vectors
        raw_scores = np.dot(self.embeddings, query_emb)
        
        # Initialize candidate validity mask
        valid_mask = np.ones(len(self.pairs_df), dtype=bool)
        
        # 1. Leakage Prevention: mask out any records from the same conversation
        if exclude_conversation_id is not None:
            valid_mask &= (self.pairs_df['conversation_id'] != exclude_conversation_id).values
            
        # 2. Quality Guard: mask any lingering fragments or PII notices
        if 'brand_text_clean' in self.pairs_df.columns:
            brand_texts = self.pairs_df['brand_text_clean'].fillna('')
            frag_mask = brand_texts.str.contains(FRAGMENT_PATTERN, regex=True).values
            pii_mask = brand_texts.str.contains(PII_WARNING_PATTERN, regex=True).values
            valid_mask &= (~frag_mask) & (~pii_mask)
            
        # 3. Strict Intent Filtering: filter candidates to matching intent before ranking
        if intent is not None and 'intent' in self.pairs_df.columns:
            intent_mask = (self.pairs_df['intent'] == intent).values
            valid_mask &= intent_mask
            
        # 4. Minimum Relevance Threshold Filtering
        valid_mask &= (raw_scores >= min_similarity_threshold)
        
        # Candidate selection
        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return []
            
        # Rank candidate indices by true cosine similarity descending
        candidate_scores = raw_scores[candidate_indices]
        sorted_order = np.argsort(candidate_scores)[::-1]
        top_indices = candidate_indices[sorted_order[:top_k]]
        
        results = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            row = self.pairs_df.iloc[idx]
            results.append({
                "historical_customer_message": str(row['customer_text_clean']),
                "historical_brand_response": str(row['brand_text_clean']),
                "similarity": round(score, 4),
                "intent": str(row.get('intent', 'unknown')),
                "conversation_id": int(row['conversation_id']),
                "customer_tweet_id": int(row['customer_tweet_id']),
                "brand_tweet_id": int(row['brand_tweet_id']),
                "response_time_minutes": float(row.get('response_time_minutes', 0.0))
            })
            
        return results
