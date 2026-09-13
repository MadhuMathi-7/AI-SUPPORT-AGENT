"""
End-to-End AI Customer Support Agent Pipeline.

Integrates all system components into a modular, production-ready pipeline:
1. Intent Classifier (Semantic Transformer with calibrated confidence)
2. Historical Retrieval Engine (Cosine similarity over historical resolutions with leakage prevention)
3. Grounded Reply Generator (Evidence-constrained support response generation)
4. Escalation Router (Rule- and threshold-based routing between AUTO_HANDLE and ESCALATE_TO_HUMAN)
"""

import os
from typing import Dict, Any, List, Optional
import yaml

from src.intent_classifier import (
    SemanticIntentClassifier,
    TfidfLogisticRegressionClassifier
)
from src.retrieval import HistoricalRetrievalEngine
from src.reply_generator import (
    GroundedReplyGenerator,
    sanitize_brand_response,
    sanitize_customer_message
)
from src.escalation import EscalationRouter


class SupportAgentPipeline:
    """
    Unified AI Customer Support Agent Pipeline.
    """
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        classifier_path: Optional[str] = None,
        retrieval_index_path: str = "models/retrieval_index.joblib"
    ):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 1. Initialize & load intent classifier based on config (TF-IDF default per benchmark)
        clf_cfg = self.config.get("classification", {})
        default_model = clf_cfg.get("default_model", "tfidf_logistic_regression")
        emb_model = clf_cfg.get("semantic_classifier", {}).get("embedding_model", "all-MiniLM-L6-v2")
        
        if default_model == "semantic_classifier":
            print(f"[PIPELINE] Initializing Semantic Intent Classifier ({emb_model})...")
            self.classifier = SemanticIntentClassifier(model_name=emb_model)
            model_file = classifier_path or "models/semantic_intent_classifier.joblib"
        else:
            print("[PIPELINE] Initializing Production Intent Classifier (TF-IDF + Logistic Regression)...")
            tfidf_cfg = clf_cfg.get("tfidf_logistic_regression", {})
            self.classifier = TfidfLogisticRegressionClassifier(
                max_features=tfidf_cfg.get("max_features", 5000),
                ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
                C=tfidf_cfg.get("c_regularization", 1.0)
            )
            model_file = classifier_path or "models/tfidf_logistic_regression.joblib"

        if os.path.exists(model_file):
            self.classifier.load(model_file)
        else:
            print(f"[PIPELINE] Warning: Classifier model file not found at {model_file}.")

        # 2. Initialize & load retrieval engine
        print("[PIPELINE] Initializing Historical Retrieval Engine...")
        self.retriever = HistoricalRetrievalEngine(model_name=emb_model, index_cache_path=retrieval_index_path)
        if os.path.exists(retrieval_index_path):
            self.retriever.build_index()
        else:
            print(f"[PIPELINE] Warning: Retrieval index not found at {retrieval_index_path}. Building index...")
            train_path = self.config.get("paths", {}).get("train_split", "data/splits/train_pairs.parquet")
            self.retriever.build_index(train_pairs_path=train_path)

        # 3. Initialize reply generator
        gen_cfg = self.config.get("generation", {})
        self.reply_generator = GroundedReplyGenerator(
            model=gen_cfg.get("model", "gpt-3.5-turbo"),
            temperature=gen_cfg.get("temperature", 0.2)
        )

        # 4. Initialize escalation router
        self.escalation_router = EscalationRouter(config_path=config_path)
        print("[PIPELINE] Support Agent Pipeline successfully initialized.")

    def _load_config(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def process(
        self,
        customer_message: str,
        conversation_id: Optional[int] = None,
        conversation_context: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Process an incoming customer support message through the complete pipeline.
        
        Args:
            customer_message: Customer inquiry text.
            conversation_id: ID of the current conversation (for retrieval leakage masking).
            conversation_context: Optional string containing previous conversation turns.
            top_k: Number of historical interactions to retrieve.
            
        Returns:
            Structured output containing:
                - customer_message
                - intent
                - intent_confidence
                - retrieved_examples
                - reply
                - decision
                - reason
                - metadata
        """
        # Step 1: Predict Intent and Calibrated Confidence
        intent, intent_confidence = self.classifier.predict_with_confidence(customer_message)
        
        # Step 2: Retrieve Semantically Similar Historical Brand Resolutions (with leakage protection & intent alignment)
        retrieved_examples = self.retriever.retrieve(
            query=customer_message,
            top_k=top_k,
            exclude_conversation_id=conversation_id,
            intent=intent
        )
        
        # Step 3: Evaluate Escalation Decision (AUTO_HANDLE vs ESCALATE_TO_HUMAN)
        escalation_result = self.escalation_router.evaluate(
            customer_message=customer_message,
            predicted_intent=intent,
            intent_confidence=intent_confidence,
            retrieved_examples=retrieved_examples,
            conversation_context=conversation_context
        )
        
        # Step 4: Generate Grounded Reply
        reply = self.reply_generator.generate_reply(
            customer_message=customer_message,
            predicted_intent=intent,
            retrieved_examples=retrieved_examples,
            conversation_context=conversation_context
        )
        
        # Format final structured result
        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": round(float(intent_confidence), 4),
            "retrieved_examples": [
                {
                    "historical_customer_message": sanitize_customer_message(ex["historical_customer_message"]),
                    "historical_brand_response": sanitize_brand_response(ex["historical_brand_response"]),
                    "similarity": ex["similarity"]
                }
                for ex in retrieved_examples
            ],
            "reply": reply,
            "decision": escalation_result["decision"],
            "reason": escalation_result["reason"],
            "metadata": {
                "retrieval_confidence": escalation_result["retrieval_confidence"],
                "sensitive_keyword": escalation_result.get("sensitive_keyword_matched"),
                "conversation_id": conversation_id
            }
        }
