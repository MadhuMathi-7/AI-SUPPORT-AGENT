"""
Intent Classifier Module for Customer Support System.

Provides three intent classification models:
1. MajorityBaselineClassifier: Trivial baseline predicting the most frequent class.
2. TfidfLogisticRegressionClassifier: Traditional ML baseline (TF-IDF + Logistic Regression).
3. SemanticIntentClassifier: Proposed system using SentenceTransformer embeddings with calibrated confidence.
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


class MajorityBaselineClassifier:
    """Trivial baseline model that always predicts the majority intent from training data."""
    
    def __init__(self):
        self.majority_class: Optional[str] = None
        self.confidence: float = 0.0
        
    def fit(self, texts: List[str], labels: List[str]):
        """Fit by finding the mode of the training labels."""
        series = pd.Series(labels)
        counts = series.value_counts()
        self.majority_class = counts.index[0]
        self.confidence = float(counts.iloc[0] / len(labels))
        return self
        
    def predict(self, texts: List[str]) -> List[str]:
        """Predict majority class for all inputs."""
        if not self.majority_class:
            raise ValueError("Model is not fitted yet.")
        return [self.majority_class] * len(texts)
        
    def predict_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict majority class with fixed historical frequency confidence."""
        if not self.majority_class:
            raise ValueError("Model is not fitted yet.")
        return [(self.majority_class, self.confidence) for _ in texts]


class TfidfLogisticRegressionClassifier:
    """Traditional ML Baseline: TF-IDF feature extraction + Logistic Regression classifier."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), C: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words='english'
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, texts: List[str], labels: List[str]):
        """Vectorize texts and train Logistic Regression."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = self.classifier.classes_
        return self
        
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Predict intent labels."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        X = self.vectorizer.transform(text_list)
        preds = list(self.classifier.predict(X))
        return preds[0] if single_input else preds
        
    def predict_with_confidence(self, texts: Union[str, List[str]]) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """Predict intent labels and calibrated probability scores."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        X = self.vectorizer.transform(text_list)
        probs = self.classifier.predict_proba(X)
        preds = self.classifier.predict(X)
        max_probs = probs.max(axis=1)
        results = [(str(p), float(round(c, 4))) for p, c in zip(preds, max_probs)]
        return results[0] if single_input else results
        
    def predict_single(self, text: str) -> Tuple[str, float]:
        """Convenience method for single message inference."""
        return self.predict_with_confidence(text)
        
    def save(self, filepath: str):
        """Save pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, filepath)
        
    def load(self, filepath: str):
        """Load pipeline from disk."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except ImportError:
                pass
            data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.classes_ = self.classifier.classes_
        return self


class SemanticIntentClassifier:
    """
    Proposed Semantic Intent Classifier:
    Uses Sentence Transformers to encode dense 384-d semantic representations,
    followed by a calibrated classification head.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.classes_: Optional[np.ndarray] = None
        
    def fit(self, texts: List[str], labels: List[str], batch_size: int = 64):
        """Encode texts into embeddings and train classifier."""
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        self.classifier.fit(embeddings, labels)
        self.classes_ = self.classifier.classes_
        return self
        
    def predict(self, texts: Union[str, List[str]], batch_size: int = 64) -> Union[str, List[str]]:
        """Predict intent labels for a single string or list of strings."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        
        embeddings = self.encoder.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        preds = list(self.classifier.predict(embeddings))
        return preds[0] if single_input else preds
        
    def predict_with_confidence(self, texts: Union[str, List[str]], batch_size: int = 64) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """Predict intent labels with calibrated confidence probabilities."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        
        embeddings = self.encoder.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        probs = self.classifier.predict_proba(embeddings)
        preds = self.classifier.predict(embeddings)
        max_probs = probs.max(axis=1)
        results = [(str(p), float(round(c, 4))) for p, c in zip(preds, max_probs)]
        return results[0] if single_input else results
        
    def predict_single(self, text: str) -> Tuple[str, float]:
        """Convenience method for single message inference."""
        res = self.predict_with_confidence(text)
        return res
        
    def save(self, filepath: str):
        """Save classifier model weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"classifier": self.classifier, "model_name": self.model_name}, filepath)
        
    def load(self, filepath: str):
        """Load classifier model weights."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except ImportError:
                pass
            data = joblib.load(filepath)
        self.classifier = data["classifier"]
        self.classes_ = self.classifier.classes_
        return self
