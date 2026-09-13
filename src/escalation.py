"""
Escalation Router Module for Customer Support System.

Determines whether an incoming customer issue can be safely handled automatically (AUTO_HANDLE)
or must be routed to a human customer support specialist (ESCALATE_TO_HUMAN).

Decision Factors:
1. Intent Confidence: Low confidence indicates ambiguous or out-of-domain requests.
2. Retrieval Similarity: Weak historical evidence indicates novel, complex, or unsupported edge cases.
3. Sensitive/Legal/Escalation Keywords: Explicit demands for human agents, legal threats, or fraud reports.
4. High-Risk Intent Policies: Security and account takeover incidents require human verification.
"""

from typing import Dict, Any, List, Optional
import yaml
import os
import re


class EscalationRouter:
    """
    Rule- and threshold-based escalation engine for customer support queries.
    """
    
    DEFAULT_SENSITIVE_KEYWORDS = [
        "lawyer", "attorney", "sue", "legal action", "court",
        "police", "fraud", "stolen", "scam", "unauthorized",
        "human agent", "supervisor", "speak to a person", "talk to a person",
        "talk to human", "real person", "manager", "representative"
    ]
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        intent_confidence_threshold: float = 0.65,
        retrieval_similarity_threshold: float = 0.60,
        sensitive_keywords: Optional[List[str]] = None
    ):
        self.intent_confidence_threshold = intent_confidence_threshold
        self.retrieval_similarity_threshold = retrieval_similarity_threshold
        self.sensitive_keywords = sensitive_keywords or self.DEFAULT_SENSITIVE_KEYWORDS
        
        # Load from config.yaml if present
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if "escalation" in cfg:
                        esc_cfg = cfg["escalation"]
                        self.intent_confidence_threshold = esc_cfg.get(
                            "intent_confidence_threshold", self.intent_confidence_threshold
                        )
                        self.retrieval_similarity_threshold = esc_cfg.get(
                            "retrieval_similarity_threshold", self.retrieval_similarity_threshold
                        )
                        if "sensitive_keywords" in esc_cfg:
                            self.sensitive_keywords = esc_cfg["sensitive_keywords"]
            except Exception as e:
                print(f"[ESCALATION] Note: Could not parse config from {config_path} ({e}), using default parameters.")

    def check_sensitive_keywords(self, text: str) -> Optional[str]:
        """
        Check if message contains explicit escalation or high-risk keywords.
        """
        text_lower = text.lower()
        for kw in self.sensitive_keywords:
            # Word boundary check where appropriate or substring
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_lower):
                return kw
        return None

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieved_examples: List[Dict[str, Any]],
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate customer message and pipeline state to route to AUTO_HANDLE or ESCALATE_TO_HUMAN.
        
        Args:
            customer_message: Raw or clean customer message text.
            predicted_intent: Classified intent category.
            intent_confidence: Classifier confidence score (0.0 - 1.0).
            retrieved_examples: List of top retrieved historical pairs.
            conversation_context: Optional previous turns in conversation.
            
        Returns:
            Dict containing:
                - decision: "AUTO_HANDLE" | "ESCALATE_TO_HUMAN"
                - reason: Detailed explanation for decision
                - intent_confidence: Float
                - retrieval_confidence: Float (top similarity)
                - sensitive_keyword_matched: Optional[str]
        """
        # Determine top retrieval similarity score
        top_retrieval_score = 0.0
        if retrieved_examples and len(retrieved_examples) > 0:
            top_retrieval_score = float(retrieved_examples[0].get("similarity", 0.0))
            
        # 1. Rule 1: Sensitive Keyword Detection (Explicit Human Request or Legal/Fraud)
        matched_kw = self.check_sensitive_keywords(customer_message)
        if matched_kw:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": f"Customer message contains sensitive or explicit escalation keyword: '{matched_kw}'.",
                "intent_confidence": round(intent_confidence, 4),
                "retrieval_confidence": round(top_retrieval_score, 4),
                "sensitive_keyword_matched": matched_kw
            }

        # 2. Rule 2: High-Risk Security / Account Compromise Intent
        if predicted_intent == "account_access_security":
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": "Account access and security inquiries require human identity verification and security protocol adherence.",
                "intent_confidence": round(intent_confidence, 4),
                "retrieval_confidence": round(top_retrieval_score, 4),
                "sensitive_keyword_matched": None
            }
            
        # 3. Rule 3: Low Intent Confidence (Ambiguous Query)
        if intent_confidence < self.intent_confidence_threshold:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f} < threshold {self.intent_confidence_threshold:.2f}). Issue is ambiguous or out-of-domain.",
                "intent_confidence": round(intent_confidence, 4),
                "retrieval_confidence": round(top_retrieval_score, 4),
                "sensitive_keyword_matched": None
            }

        # 4. Rule 4: Weak or Missing Historical Evidence (No verified prior resolutions)
        if not retrieved_examples or top_retrieval_score < self.retrieval_similarity_threshold:
            reason = (
                "No relevant historical resolution evidence retrieved. Safe escalation required."
                if not retrieved_examples
                else f"Weak historical retrieval evidence (similarity {top_retrieval_score:.2f} < threshold {self.retrieval_similarity_threshold:.2f}). No verified prior resolution matches this issue."
            )
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "reason": reason,
                "intent_confidence": round(intent_confidence, 4),
                "retrieval_confidence": round(top_retrieval_score, 4),
                "sensitive_keyword_matched": None
            }

        # 5. Rule 5: Safe to Automatically Handle
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"High intent confidence ({intent_confidence:.2f}) and relevant historical evidence (similarity {top_retrieval_score:.2f}) support automated handling.",
            "intent_confidence": round(intent_confidence, 4),
            "retrieval_confidence": round(top_retrieval_score, 4),
            "sensitive_keyword_matched": None
        }
