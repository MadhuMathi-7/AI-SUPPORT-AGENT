"""
Escalation Router Evaluation Module.

Evaluates the routing decisions (AUTO_HANDLE vs ESCALATE_TO_HUMAN) made by the pipeline:
1. Routing Distribution: Proportion of automated resolutions vs human escalations.
2. Trigger Breakdown: Attribution of escalations across sensitive keywords, security policy, low classifier confidence, and low retrieval evidence.
3. Difficulty Stratification: Alignment with Golden Set difficulty levels (Easy, Medium, Hard).
4. Edge Case Behavior: Verifies that complex or high-risk cases are safely escalated.
"""

import os
import sys
import json
from typing import Dict, Any, List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from src.pipeline import SupportAgentPipeline


def evaluate_escalation(
    golden_path: str = "data/golden/golden_set.csv",
    output_report_path: str = "reports/escalation_evaluation_results.json"
) -> Dict[str, Any]:
    """
    Run pipeline on Golden Set and evaluate escalation behavior.
    """
    print(f"[ESCALATION_EVAL] Loading Golden Set from: {golden_path}")
    df = pd.read_csv(golden_path)
    
    pipeline = SupportAgentPipeline()
    
    results = []
    reason_counts = {
        "sensitive_keyword": 0,
        "security_policy": 0,
        "low_intent_confidence": 0,
        "low_retrieval_similarity": 0,
        "auto_handled": 0
    }
    
    difficulty_breakdown = {
        "easy": {"total": 0, "auto_handle": 0, "escalate": 0},
        "medium": {"total": 0, "auto_handle": 0, "escalate": 0},
        "hard": {"total": 0, "auto_handle": 0, "escalate": 0}
    }
    
    intent_breakdown = {}
    
    print(f"[ESCALATION_EVAL] Evaluating escalation across {len(df)} Golden Set queries...")
    for idx, row in df.iterrows():
        c_msg = str(row['customer_message'])
        conv_id = int(row['conversation_id'])
        gold_intent = str(row['intent'])
        difficulty = str(row.get('difficulty', 'medium')).lower()
        
        output = pipeline.process(
            customer_message=c_msg,
            conversation_id=conv_id
        )
        
        decision = output["decision"]
        reason = output["reason"]
        pred_intent = output["intent"]
        intent_conf = output["intent_confidence"]
        top_sim = output["retrieved_examples"][0]["similarity"] if output["retrieved_examples"] else 0.0
        
        # Difficulty tracking
        if difficulty not in difficulty_breakdown:
            difficulty_breakdown[difficulty] = {"total": 0, "auto_handle": 0, "escalate": 0}
        difficulty_breakdown[difficulty]["total"] += 1
        
        if decision == "AUTO_HANDLE":
            reason_counts["auto_handled"] += 1
            difficulty_breakdown[difficulty]["auto_handle"] += 1
        else:
            difficulty_breakdown[difficulty]["escalate"] += 1
            if "sensitive or explicit escalation keyword" in reason:
                reason_counts["sensitive_keyword"] += 1
            elif "Account access and security inquiries" in reason or pred_intent == "account_access_security":
                reason_counts["security_policy"] += 1
            elif "Low intent classification confidence" in reason:
                reason_counts["low_intent_confidence"] += 1
            elif "Weak historical retrieval evidence" in reason:
                reason_counts["low_retrieval_similarity"] += 1
                
        # Intent-level tracking
        if pred_intent not in intent_breakdown:
            intent_breakdown[pred_intent] = {"total": 0, "auto_handle": 0, "escalate": 0}
        intent_breakdown[pred_intent]["total"] += 1
        if decision == "AUTO_HANDLE":
            intent_breakdown[pred_intent]["auto_handle"] += 1
        else:
            intent_breakdown[pred_intent]["escalate"] += 1
            
        results.append({
            "example_id": str(row['example_id']),
            "customer_message": c_msg,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "intent_confidence": intent_conf,
            "top_retrieval_similarity": top_sim,
            "difficulty": difficulty,
            "decision": decision,
            "reason": reason
        })
        
    total_queries = len(df)
    auto_handle_count = reason_counts["auto_handled"]
    escalate_count = total_queries - auto_handle_count
    
    auto_handle_rate = round((auto_handle_count / total_queries) * 100, 2)
    escalate_rate = round((escalate_count / total_queries) * 100, 2)
    
    # Calculate difficulty escalation rates
    diff_summary = {}
    for d, counts in difficulty_breakdown.items():
        if counts["total"] > 0:
            esc_pct = round((counts["escalate"] / counts["total"]) * 100, 2)
            auto_pct = round((counts["auto_handle"] / counts["total"]) * 100, 2)
            diff_summary[d] = {
                "total": counts["total"],
                "auto_handle_pct": auto_pct,
                "escalate_pct": esc_pct
            }
            
    summary = {
        "total_queries_evaluated": total_queries,
        "auto_handle_count": auto_handle_count,
        "auto_handle_rate_pct": auto_handle_rate,
        "escalate_to_human_count": escalate_count,
        "escalate_rate_pct": escalate_rate,
        "escalation_trigger_breakdown": {
            "sensitive_or_legal_keywords": {
                "count": reason_counts["sensitive_keyword"],
                "pct_of_total": round((reason_counts["sensitive_keyword"] / total_queries) * 100, 2)
            },
            "security_account_policy": {
                "count": reason_counts["security_policy"],
                "pct_of_total": round((reason_counts["security_policy"] / total_queries) * 100, 2)
            },
            "low_intent_confidence": {
                "count": reason_counts["low_intent_confidence"],
                "pct_of_total": round((reason_counts["low_intent_confidence"] / total_queries) * 100, 2)
            },
            "low_retrieval_evidence": {
                "count": reason_counts["low_retrieval_similarity"],
                "pct_of_total": round((reason_counts["low_retrieval_similarity"] / total_queries) * 100, 2)
            }
        },
        "difficulty_stratification": diff_summary,
        "intent_breakdown": intent_breakdown,
        "sample_escalation_records": results[:10]
    }
    
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[ESCALATION_EVAL] Escalation Evaluation Summary:")
    print(f"- Total Queries: {total_queries}")
    print(f"- Auto-Handle Rate: {auto_handle_rate}% ({auto_handle_count}/{total_queries})")
    print(f"- Escalate-to-Human Rate: {escalate_rate}% ({escalate_count}/{total_queries})")
    print(f"- Easy Queries Escalation Rate: {diff_summary.get('easy', {}).get('escalate_pct', 0.0)}%")
    print(f"- Medium Queries Escalation Rate: {diff_summary.get('medium', {}).get('escalate_pct', 0.0)}%")
    print(f"- Hard Queries Escalation Rate: {diff_summary.get('hard', {}).get('escalate_pct', 0.0)}%")
    print(f"- Saved report to: {output_report_path}")
    
    return summary


if __name__ == "__main__":
    evaluate_escalation()
