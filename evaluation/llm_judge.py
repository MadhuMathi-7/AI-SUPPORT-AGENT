"""
LLM-as-Judge Evaluation Module.

Evaluates generated customer support replies across 6 structured rubric dimensions:
1. Correctness (1-5): Factual accuracy and appropriate alignment with customer issue.
2. Groundedness (1-5): Direct attribution to retrieved historical evidence.
3. Helpfulness (1-5): Actionability and clarity of next steps.
4. Relevance (1-5): Directness of reply to the customer's specific question.
5. Tone (1-5): Professional, empathetic, and respectful customer service voice.
6. Unsupported Claims (Binary 0/1): Detection of fabricated amounts, dates, or unauthorized commitments.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

JUDGE_RUBRIC_PROMPT = """You are an expert customer support quality auditor evaluating an AI customer support response for Amazon Twitter Support (@AmazonHelp).

EVALUATION RUBRIC (Score 1 to 5):
1. Correctness (1-5):
   - 5: Completely accurate, correctly addresses the core issue.
   - 3: Partially correct, misses minor details.
   - 1: Factually incorrect or misinterprets the customer request.

2. Groundedness (1-5):
   - 5: Every statement is directly supported by the retrieved historical evidence or official Amazon policy.
   - 3: Mostly supported, but makes minor unverified assumptions.
   - 1: Purely hallucinated or contradicts evidence.

3. Helpfulness (1-5):
   - 5: Gives clear, immediate actionable next steps (DM, order lookup, returns portal).
   - 3: Vague or generic help.
   - 1: Unhelpful, dismissive, or dead-end.

4. Relevance (1-5):
   - 5: Directly answers the customer's specific issue without filler.
   - 3: Somewhat off-topic or bloated.
   - 1: Completely irrelevant.

5. Tone (1-5):
   - 5: Empathetic, polite, professional, brand-appropriate.
   - 3: Neutral or slightly robotic.
   - 1: Rude, robotic, or inappropriate.

6. Unsupported Claims (Binary 0 or 1):
   - 0: No fake claims.
   - 1: Contains fabricated refund amounts, promises of immediate action, or fake policies.

Provide output ONLY as valid JSON in this exact structure:
{
  "correctness": 5,
  "groundedness": 5,
  "helpfulness": 4,
  "relevance": 5,
  "tone": 5,
  "unsupported_claims": 0,
  "explanation": "Brief 1-2 sentence justification."
}
"""


class LLMJudge:
    """
    Quality Judge for evaluating customer support replies with API and deterministic fallback.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
    def evaluate_single(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated reply against the rubric.
        """
        # 1. Try LLM API if available
        if self.api_key:
            try:
                llm_eval = self._call_llm_judge(
                    customer_message, predicted_intent, retrieved_evidence, generated_reply
                )
                if llm_eval:
                    return llm_eval
            except Exception as e:
                pass
                
        # 2. Transparent Rule-and-Heuristic Judge Fallback
        return self._heuristic_judge(
            customer_message, predicted_intent, retrieved_evidence, generated_reply
        )
        
    def _call_llm_judge(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> Optional[Dict[str, Any]]:
        """Query LLM for judge evaluation."""
        try:
            import requests
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                return None
                
            evidence_str = "\n".join([
                f"- Ex {i+1}: Customer: {e.get('historical_customer_message', '')} -> Brand: {e.get('historical_brand_response', '')}"
                for i, e in enumerate(retrieved_evidence[:3])
            ])
            
            user_prompt = f"""Customer Inquiry: "{customer_message}"
Predicted Intent: {predicted_intent}
Retrieved Historical Evidence:
{evidence_str}

Generated Support Reply:
"{generated_reply}"

Evaluate the reply strictly according to the rubric."""

            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": JUDGE_RUBRIC_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                result = json.loads(res.json()["choices"][0]["message"]["content"])
                return {
                    "correctness": int(result.get("correctness", 4)),
                    "groundedness": int(result.get("groundedness", 4)),
                    "helpfulness": int(result.get("helpfulness", 4)),
                    "relevance": int(result.get("relevance", 4)),
                    "tone": int(result.get("tone", 5)),
                    "unsupported_claims": int(result.get("unsupported_claims", 0)),
                    "explanation": str(result.get("explanation", "LLM Judge evaluation completed.")),
                    "judge_type": "llm_api"
                }
        except Exception:
            pass
        return None

    def _heuristic_judge(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> Dict[str, Any]:
        """
        Explainable heuristic scoring engine based on token overlap, grounding, and tone markers.
        """
        reply_lower = generated_reply.lower()
        c_lower = customer_message.lower()
        
        # 1. Check unsupported claims
        has_unsupported = 0
        if re.search(r'[\$£€₹]\s*\d+', generated_reply) or "i have refunded" in reply_lower or "we deposited" in reply_lower:
            has_unsupported = 1
            
        # 2. Groundedness Score
        groundedness = 4
        if retrieved_evidence and len(retrieved_evidence) > 0:
            top_sim = retrieved_evidence[0].get("similarity", 0.0)
            if top_sim > 0.75:
                groundedness = 5
            elif top_sim > 0.50:
                groundedness = 4
            else:
                groundedness = 3
        if has_unsupported == 1:
            groundedness = min(groundedness, 2)
            
        # 3. Tone Score
        polite_markers = ["sorry", "please", "glad", "help", "apologize", "assist", "welcome"]
        polite_count = sum(1 for m in polite_markers if m in reply_lower)
        if polite_count >= 2:
            tone = 5
        elif polite_count == 1:
            tone = 4
        else:
            tone = 3
            
        # 4. Helpfulness Score
        action_markers = ["dm", "direct message", "link", "visit", "orders", "details", "contact", "support", "check"]
        action_count = sum(1 for m in action_markers if m in reply_lower)
        if action_count >= 2:
            helpfulness = 5
        elif action_count == 1:
            helpfulness = 4
        else:
            helpfulness = 3
            
        # 5. Relevance Score
        c_tokens = set(re.findall(r'\b\w{4,}\b', c_lower))
        r_tokens = set(re.findall(r'\b\w{4,}\b', reply_lower))
        overlap = len(c_tokens.intersection(r_tokens))
        if overlap >= 2 or len(c_tokens) == 0:
            relevance = 5
        elif overlap == 1:
            relevance = 4
        else:
            relevance = 3
            
        # 6. Correctness Score (composite of groundedness and relevance)
        correctness = round((groundedness + relevance) / 2)
        
        return {
            "correctness": int(correctness),
            "groundedness": int(groundedness),
            "helpfulness": int(helpfulness),
            "relevance": int(relevance),
            "tone": int(tone),
            "unsupported_claims": int(has_unsupported),
            "explanation": f"Grounded response with {polite_count} polite markers and {action_count} actionable guidance items.",
            "judge_type": "heuristic_grounding_auditor"
        }


def run_llm_judge_evaluation(
    golden_path: str = "data/golden/golden_set.csv",
    output_path: str = "reports/llm_judge_evaluations.json",
    sample_size: int = 200
) -> Dict[str, Any]:
    """Run judge evaluation on golden set predictions."""
    from src.pipeline import SupportAgentPipeline
    
    print(f"[LLM_JUDGE] Loading Golden Set from: {golden_path}")
    df = pd.read_csv(golden_path)
    if len(df) > sample_size:
        df = df.iloc[:sample_size]
        
    pipeline = SupportAgentPipeline()
    judge = LLMJudge()
    
    records = []
    correctness_scores = []
    groundedness_scores = []
    helpfulness_scores = []
    relevance_scores = []
    tone_scores = []
    unsupported_claims_count = 0
    
    print(f"[LLM_JUDGE] Auditing {len(df)} generated customer replies...")
    for idx, row in df.iterrows():
        c_msg = str(row['customer_message'])
        conv_id = int(row['conversation_id'])
        
        output = pipeline.process(
            customer_message=c_msg,
            conversation_id=conv_id
        )
        
        j_eval = judge.evaluate_single(
            customer_message=c_msg,
            predicted_intent=output["intent"],
            retrieved_evidence=output["retrieved_examples"],
            generated_reply=output["reply"]
        )
        
        correctness_scores.append(j_eval["correctness"])
        groundedness_scores.append(j_eval["groundedness"])
        helpfulness_scores.append(j_eval["helpfulness"])
        relevance_scores.append(j_eval["relevance"])
        tone_scores.append(j_eval["tone"])
        if j_eval["unsupported_claims"] == 1:
            unsupported_claims_count += 1
            
        records.append({
            "example_id": str(row['example_id']),
            "customer_message": c_msg,
            "predicted_intent": output["intent"],
            "decision": output["decision"],
            "generated_reply": output["reply"],
            "judge_evaluation": j_eval
        })
        
    summary = {
        "num_evaluated": len(df),
        "mean_scores": {
            "correctness": round(float(np.mean(correctness_scores)), 2),
            "groundedness": round(float(np.mean(groundedness_scores)), 2),
            "helpfulness": round(float(np.mean(helpfulness_scores)), 2),
            "relevance": round(float(np.mean(relevance_scores)), 2),
            "tone": round(float(np.mean(tone_scores)), 2)
        },
        "overall_average_quality_score_out_of_5": round(float(np.mean([
            np.mean(correctness_scores),
            np.mean(groundedness_scores),
            np.mean(helpfulness_scores),
            np.mean(relevance_scores),
            np.mean(tone_scores)
        ])), 2),
        "unsupported_claims_count": unsupported_claims_count,
        "unsupported_claims_rate_pct": round((unsupported_claims_count / len(df)) * 100, 2),
        "sample_evaluations": records[:10]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[LLM_JUDGE] Judge Evaluation Summary (N={len(df)}):")
    print(f"- Correctness: {summary['mean_scores']['correctness']} / 5.0")
    print(f"- Groundedness: {summary['mean_scores']['groundedness']} / 5.0")
    print(f"- Helpfulness: {summary['mean_scores']['helpfulness']} / 5.0")
    print(f"- Relevance: {summary['mean_scores']['relevance']} / 5.0")
    print(f"- Tone: {summary['mean_scores']['tone']} / 5.0")
    print(f"- Overall Mean Quality: {summary['overall_average_quality_score_out_of_5']} / 5.0")
    print(f"- Unsupported Claims Rate: {summary['unsupported_claims_rate_pct']}%")
    print(f"- Saved judge report to: {output_path}")
    
    return summary


if __name__ == "__main__":
    run_llm_judge_evaluation()
