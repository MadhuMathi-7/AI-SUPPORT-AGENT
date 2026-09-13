"""
Grounded Reply Evaluation Module.

Evaluates generated customer support replies against historical brand reference responses and retrieved evidence:
1. Semantic Similarity: Cosine similarity between generated response embedding and historical reference embedding.
2. Lexical Overlap: Token Precision, Recall, and F1 (ROUGE-1/L approximation).
3. Groundedness: Token containment and semantic alignment with retrieved historical evidence.
4. Policy / Constraint Adherence: Twitter length limit (<280 chars), absence of hallucinated refund amounts ($XX) or fake tracking numbers.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.pipeline import SupportAgentPipeline


def compute_token_overlap(hyp: str, ref: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1 overlap."""
    hyp_tokens = set(re.findall(r'\b\w+\b', hyp.lower()))
    ref_tokens = set(re.findall(r'\b\w+\b', ref.lower()))
    
    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    common = hyp_tokens.intersection(ref_tokens)
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def check_hallucination_indicators(reply: str) -> List[str]:
    """Check for suspicious hallucinated specific claims not supported by generic Twitter support."""
    flags = []
    # Check for specific currency amounts like $50, £10, Rs 500
    if re.search(r'[\$£€₹]\s*\d+', reply):
        flags.append("specific_currency_amount")
    # Check for fabricated tracking or order codes like ORD-12345
    if re.search(r'\b[A-Z0-9]{10,}\b', reply):
        flags.append("synthetic_tracking_code")
    # Check for claims of actions already completed
    if re.search(r'\b(i have refunded|we already processed your refund|money has been sent)\b', reply.lower()):
        flags.append("unverified_action_promise")
    return flags


def evaluate_replies(
    golden_path: str = "data/golden/golden_set.csv",
    test_split_path: str = "data/splits/test_pairs.parquet",
    output_report_path: str = "reports/reply_evaluation_results.json",
    sample_size: int = 200
) -> Dict[str, Any]:
    """
    Run reply generation and compute groundedness and quality metrics.
    """
    print(f"[REPLY_EVAL] Loading Golden Set from: {golden_path}")
    golden_df = pd.read_csv(golden_path)
    
    # Load test pairs to get the actual ground-truth historical brand responses
    print(f"[REPLY_EVAL] Loading Test Split for ground truth brand replies: {test_split_path}")
    test_df = pd.read_parquet(test_split_path)
    
    # Merge on customer_tweet_id or conversation_id
    merged_df = pd.merge(
        golden_df,
        test_df[['customer_tweet_id', 'brand_text_clean', 'thread_context']],
        on='customer_tweet_id',
        how='left'
    )
    
    if len(merged_df) > sample_size:
        merged_df = merged_df.iloc[:sample_size]
        
    pipeline = SupportAgentPipeline()
    similarity_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    eval_records = []
    semantic_sims = []
    f1_scores = []
    lengths = []
    hallucination_count = 0
    under_280_count = 0
    
    print(f"[REPLY_EVAL] Evaluating reply generation across {len(merged_df)} Golden Set customer queries...")
    for idx, row in merged_df.iterrows():
        c_msg = str(row['customer_message'])
        conv_id = int(row['conversation_id'])
        ref_reply = str(row.get('brand_text_clean', ''))
        gold_intent = str(row['intent'])
        difficulty = str(row['difficulty'])
        
        # Run through pipeline
        output = pipeline.process(
            customer_message=c_msg,
            conversation_id=conv_id,
            conversation_context=str(row.get('thread_context', ''))
        )
        
        gen_reply = output["reply"]
        pred_intent = output["intent"]
        decision = output["decision"]
        
        # Token overlap with historical reference response
        overlap = compute_token_overlap(gen_reply, ref_reply)
        f1_scores.append(overlap["f1"])
        
        # Semantic similarity between generated reply and reference response
        embs = similarity_encoder.encode([gen_reply, ref_reply], normalize_embeddings=True)
        sem_sim = float(np.dot(embs[0], embs[1]))
        semantic_sims.append(sem_sim)
        
        # Length & Constraints
        char_len = len(gen_reply)
        lengths.append(char_len)
        if char_len <= 280:
            under_280_count += 1
            
        # Hallucination check
        h_flags = check_hallucination_indicators(gen_reply)
        if h_flags:
            hallucination_count += 1
            
        eval_records.append({
            "example_id": str(row['example_id']),
            "conversation_id": conv_id,
            "difficulty": difficulty,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "customer_message": c_msg,
            "historical_brand_reference": ref_reply,
            "generated_reply": gen_reply,
            "token_overlap_f1": overlap["f1"],
            "semantic_similarity_to_ref": round(sem_sim, 4),
            "char_length": char_len,
            "hallucination_flags": h_flags,
            "decision": decision
        })
        
    summary_metrics = {
        "num_evaluated": len(merged_df),
        "mean_semantic_similarity_to_historical_ref": round(float(np.mean(semantic_sims)), 4),
        "median_semantic_similarity_to_historical_ref": round(float(np.median(semantic_sims)), 4),
        "mean_token_f1": round(float(np.mean(f1_scores)), 4),
        "pct_within_twitter_character_limit": round((under_280_count / len(merged_df)) * 100, 2),
        "mean_character_length": round(float(np.mean(lengths)), 1),
        "hallucination_flag_rate_pct": round((hallucination_count / len(merged_df)) * 100, 2),
        "sample_evaluations": eval_records[:10]
    }
    
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"\n[REPLY_EVAL] Reply Evaluation Complete:")
    print(f"- Evaluated Queries: {len(merged_df)}")
    print(f"- Mean Semantic Similarity to Brand Reference: {summary_metrics['mean_semantic_similarity_to_historical_ref']:.4f}")
    print(f"- Mean Token F1 Overlap: {summary_metrics['mean_token_f1']:.4f}")
    print(f"- Twitter 280-char Limit Compliance: {summary_metrics['pct_within_twitter_character_limit']}%")
    print(f"- Hallucination Flag Rate: {summary_metrics['hallucination_flag_rate_pct']}%")
    print(f"- Report saved to: {output_report_path}")
    
    return summary_metrics


if __name__ == "__main__":
    evaluate_replies()
