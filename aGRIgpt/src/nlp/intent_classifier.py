"""ML-Based Intent Classifier for Agricultural Queries.
Uses TF-IDF feature extraction combined with a Calibrated Logistic Regression classifier
to categorize user utterances into domain intents with probabilistic confidence scoring.
"""

from __future__ import annotations

import json
import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.nlp.preprocessor import AgriTextPreprocessor, default_preprocessor


class AgriIntentClassifier:
    """Classifies user queries into agricultural intent categories with confidence scores."""

    DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intents.json")
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "intent_classifier.joblib")

    def __init__(self, data_path: Optional[str] = None, model_path: Optional[str] = None):
        self.data_path = os.path.abspath(data_path or self.DEFAULT_DATA_PATH)
        self.model_path = os.path.abspath(model_path or self.DEFAULT_MODEL_PATH)
        self.preprocessor = default_preprocessor
        self.pipeline: Optional[Pipeline] = None
        self.intent_descriptions: Dict[str, str] = {}
        self.classes_: List[str] = []

        # Load or train automatically
        self._initialize()

    def _initialize(self):
        """Load cached model or train fresh model if missing."""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                return
            except Exception as e:
                print(f"[AgriIntentClassifier] Failed to load cached model ({e}), retraining...")
        
        self.train_and_save()

    def load_training_data(self) -> Tuple[List[str], List[str]]:
        """Load training patterns and intent labels from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Intent dataset not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        labels = []
        self.intent_descriptions = {}

        for item in data.get("intents", []):
            tag = item["tag"]
            self.intent_descriptions[tag] = item.get("description", "")
            for pattern in item.get("patterns", []):
                cleaned = self.preprocessor.preprocess(pattern)["cleaned"]
                texts.append(cleaned)
                labels.append(tag)

        return texts, labels

    def train(self) -> Dict[str, Any]:
        """Train TF-IDF + LogisticRegression model on agricultural intents."""
        texts, labels = self.load_training_data()
        if not texts:
            raise ValueError("Training dataset is empty!")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                max_features=5000
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        # Cross-validation score check
        cv = StratifiedKFold(n_splits=min(5, len(set(labels))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=cv, scoring="accuracy")
        mean_cv_accuracy = float(np.mean(cv_scores))

        # Fit on full data
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)

        return {
            "status": "trained",
            "samples": len(texts),
            "classes": self.classes_,
            "mean_cv_accuracy": round(mean_cv_accuracy, 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores]
        }

    def save_model(self, path: Optional[str] = None):
        """Persist model and metadata to disk."""
        target_path = path or self.model_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classes_": self.classes_,
            "intent_descriptions": self.intent_descriptions
        }
        joblib.dump(payload, target_path)

    def load_model(self, path: Optional[str] = None):
        """Load trained model and metadata."""
        target_path = path or self.model_path
        payload = joblib.load(target_path)
        self.pipeline = payload["pipeline"]
        self.classes_ = payload["classes_"]
        self.intent_descriptions = payload.get("intent_descriptions", {})

    def train_and_save(self) -> Dict[str, Any]:
        """Train and persist."""
        metrics = self.train()
        self.save_model()
        return metrics

    def predict(self, text: str, threshold: float = 0.25) -> Dict[str, Any]:
        """Predict intent and class probabilities for user query."""
        if not self.pipeline:
            self._initialize()

        cleaned = self.preprocessor.preprocess(text)["cleaned"]

        if not cleaned.strip():
            return {
                "intent": "general_greeting_help",
                "confidence": 0.5,
                "probabilities": {c: 0.0 for c in self.classes_},
                "description": self.intent_descriptions.get("general_greeting_help", ""),
                "is_confident": False
            }

        # Predict probabilities
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        prob_dict = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(self.classes_, probabilities)
        }

        # Sort descending
        sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_confidence = sorted_probs[0]

        is_confident = top_confidence >= threshold

        return {
            "intent": top_intent,
            "confidence": top_confidence,
            "probabilities": prob_dict,
            "top_k": sorted_probs[:3],
            "description": self.intent_descriptions.get(top_intent, ""),
            "is_confident": is_confident
        }


# Singleton instance
default_intent_classifier = AgriIntentClassifier()
