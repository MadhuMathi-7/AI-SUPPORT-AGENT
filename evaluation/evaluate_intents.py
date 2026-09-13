"""
Intent Classification Evaluation Harness.

Trains and evaluates all 3 intent classifiers on the exact same Golden Evaluation Set:
1. Majority Baseline
2. TF-IDF + Logistic Regression Baseline
3. Proposed Semantic Classifier (Sentence Transformers)

Computes exact empirical Accuracy, Macro F1, Macro Precision, Macro Recall, and Confusion Matrices.
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from src.intent_classifier import (
    MajorityBaselineClassifier,
    TfidfLogisticRegressionClassifier,
    SemanticIntentClassifier
)


from typing import Dict, Any, List, Optional


def evaluate_all_intent_classifiers(
    train_path: str = "data/splits/train_pairs.parquet",
    golden_path: str = "data/golden/golden_set.csv",
    output_report_path: str = "reports/intent_evaluation_results.json",
    sample_size: int = 10000
) -> Dict[str, Any]:
    print("=" * 60)
    print("INTENT CLASSIFICATION EVALUATION HARNESS")
    print("=" * 60)
    
    # 1. Load Training Data and Golden Evaluation Set
    print(f"\n[1] Loading training split from: {train_path}")
    train_df = pd.read_parquet(train_path)
    print(f"    Total Training Pool: {len(train_df):,} pairs")
    
    # Sample training examples for fast, reproducible training
    train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    X_train = train_sample['customer_text_clean'].tolist()
    y_train = train_sample['intent'].tolist()
    print(f"    Sampled {len(X_train):,} training examples.")
    
    print(f"\n[2] Loading Golden Evaluation Set from: {golden_path}")
    golden_df = pd.read_csv(golden_path)
    X_test = golden_df['customer_message'].tolist()
    y_test = golden_df['intent'].tolist()
    print(f"    Golden Evaluation Examples: {len(X_test)}")
    
    unique_labels = sorted(list(set(y_train)))
    print(f"    Target Intent Classes ({len(unique_labels)}): {unique_labels}")
    
    results = {}
    
    # --- MODEL 1: MAJORITY BASELINE ---
    print("\n" + "-" * 50)
    print("1. Training Majority Class Baseline...")
    t0 = time.time()
    majority_model = MajorityBaselineClassifier()
    majority_model.fit(X_train, y_train)
    majority_preds = majority_model.predict(X_test)
    t_maj = time.time() - t0
    
    maj_acc = accuracy_score(y_test, majority_preds)
    maj_p, maj_r, maj_f1, _ = precision_recall_fscore_support(y_test, majority_preds, average='macro', zero_division=0)
    
    print(f"   Majority Class: '{majority_model.majority_class}'")
    print(f"   Accuracy: {maj_acc * 100:.2f}% | Macro F1: {maj_f1 * 100:.2f}% | Prec: {maj_p * 100:.2f}% | Rec: {maj_r * 100:.2f}%")
    
    results["majority_baseline"] = {
        "accuracy": round(float(maj_acc), 4),
        "macro_f1": round(float(maj_f1), 4),
        "macro_precision": round(float(maj_p), 4),
        "macro_recall": round(float(maj_r), 4),
        "training_time_sec": round(t_maj, 3)
    }
    
    # --- MODEL 2: TF-IDF + LOGISTIC REGRESSION ---
    print("\n" + "-" * 50)
    print("2. Training TF-IDF + Logistic Regression Baseline...")
    t0 = time.time()
    tfidf_model = TfidfLogisticRegressionClassifier(max_features=5000, ngram_range=(1, 2), C=1.0)
    tfidf_model.fit(X_train, y_train)
    tfidf_preds = tfidf_model.predict(X_test)
    t_tfidf = time.time() - t0
    
    tfidf_acc = accuracy_score(y_test, tfidf_preds)
    tfidf_p, tfidf_r, tfidf_f1, _ = precision_recall_fscore_support(y_test, tfidf_preds, average='macro', zero_division=0)
    
    print(f"   Accuracy: {tfidf_acc * 100:.2f}% | Macro F1: {tfidf_f1 * 100:.2f}% | Prec: {tfidf_p * 100:.2f}% | Rec: {tfidf_r * 100:.2f}% (Trained in {t_tfidf:.2f}s)")
    
    os.makedirs("models", exist_ok=True)
    tfidf_model.save("models/tfidf_logistic_regression.joblib")
    
    results["tfidf_logistic_regression"] = {
        "accuracy": round(float(tfidf_acc), 4),
        "macro_f1": round(float(tfidf_f1), 4),
        "macro_precision": round(float(tfidf_p), 4),
        "macro_recall": round(float(tfidf_r), 4),
        "training_time_sec": round(t_tfidf, 3)
    }
    
    # --- MODEL 3: PROPOSED SEMANTIC CLASSIFIER ---
    print("\n" + "-" * 50)
    print("3. Training Proposed Semantic Classifier (Sentence Transformers: all-MiniLM-L6-v2)...")
    t0 = time.time()
    semantic_model = SemanticIntentClassifier(model_name="all-MiniLM-L6-v2")
    semantic_model.fit(X_train, y_train, batch_size=128)
    semantic_preds_with_conf = semantic_model.predict_with_confidence(X_test, batch_size=128)
    semantic_preds = [p[0] for p in semantic_preds_with_conf]
    semantic_confs = [p[1] for p in semantic_preds_with_conf]
    t_sem = time.time() - t0
    
    sem_acc = accuracy_score(y_test, semantic_preds)
    sem_p, sem_r, sem_f1, _ = precision_recall_fscore_support(y_test, semantic_preds, average='macro', zero_division=0)
    
    print(f"   Accuracy: {sem_acc * 100:.2f}% | Macro F1: {sem_f1 * 100:.2f}% | Prec: {sem_p * 100:.2f}% | Rec: {sem_r * 100:.2f}% (Trained in {t_sem:.2f}s)")
    print(f"   Average Prediction Confidence: {np.mean(semantic_confs) * 100:.2f}%")
    
    semantic_model.save("models/semantic_intent_classifier.joblib")
    
    results["semantic_classifier"] = {
        "accuracy": round(float(sem_acc), 4),
        "macro_f1": round(float(sem_f1), 4),
        "macro_precision": round(float(sem_p), 4),
        "macro_recall": round(float(sem_r), 4),
        "mean_confidence": round(float(np.mean(semantic_confs)), 4),
        "training_time_sec": round(t_sem, 3)
    }
    
    # --- DETAILED COMPARISON TABLE ---
    print("\n" + "=" * 70)
    print("FINAL INTENT CLASSIFICATION COMPARISON ON GOLDEN SET (N=200)")
    print("=" * 70)
    
    summary_rows = [
        {
            "Model": "1. Majority Baseline",
            "Accuracy": f"{maj_acc * 100:.2f}%",
            "Macro Precision": f"{maj_p * 100:.2f}%",
            "Macro Recall": f"{maj_r * 100:.2f}%",
            "Macro F1": f"{maj_f1 * 100:.2f}%"
        },
        {
            "Model": "2. TF-IDF + Logistic Regression",
            "Accuracy": f"{tfidf_acc * 100:.2f}%",
            "Macro Precision": f"{tfidf_p * 100:.2f}%",
            "Macro Recall": f"{tfidf_r * 100:.2f}%",
            "Macro F1": f"{tfidf_f1 * 100:.2f}%"
        },
        {
            "Model": "3. Proposed Semantic Classifier",
            "Accuracy": f"{sem_acc * 100:.2f}%",
            "Macro Precision": f"{sem_p * 100:.2f}%",
            "Macro Recall": f"{sem_r * 100:.2f}%",
            "Macro F1": f"{sem_f1 * 100:.2f}%"
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save results to JSON
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[REPORT] Saved evaluation metrics to: {output_report_path}")
    
    return results


# Backwards-compatible alias
run_intent_evaluation = evaluate_all_intent_classifiers


if __name__ == "__main__":
    evaluate_all_intent_classifiers()
