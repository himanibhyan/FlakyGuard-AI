"""
Root-Cause Classifier Module (Task B) for FlakyGuard AI.
Implements multi-class root-cause classification on confirmed flaky tests using:
1. TF-IDF + Multinomial Logistic Regression
2. TF-IDF + Calibrated Linear SVM
Preserves all 13 research-defined categories and evaluates per-class metrics.
"""

import os
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import get_tfidf_vectorizer
from src.evaluation import evaluate_multiclass_classifier

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class RootCauseClassifier:
    """Trains, evaluates, and serves Task B root-cause models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = get_tfidf_vectorizer(max_features=8000, min_df=2)
        self.label_encoder = LabelEncoder()
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=1500,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Linear SVM": CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=self.random_state,
                    max_iter=3000
                ),
                cv=3
            )
        }
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.class_names: List[str] = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20
    ) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        """Encodes labels, splits stratified 80/20, and transforms text features."""
        # Encode string categories to integer labels
        y_encoded = self.label_encoder.fit_transform(df["root_cause_category"])
        self.class_names = list(self.label_encoder.classes_)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["combined_text"],
            y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=self.random_state
        )

        # Fit vectorizer strictly on training partition
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)

        return X_train, X_test, y_train, y_test

    def train_and_evaluate_all(
        self,
        X_train: Any,
        X_test: Any,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Trains candidate classifiers, computes macro and per-class metrics."""
        best_macro_f1 = -1.0

        for name, model in self.models.items():
            print(f"[Task B: Root Cause] Fitting {name} on {len(self.class_names)} categories...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = evaluate_multiclass_classifier(
                y_test,
                y_pred,
                self.class_names,
                name
            )
            self.results[name] = metrics

            print(f"  -> {name} | Macro-F1: {metrics['macro_f1']} | Weighted-F1: {metrics['weighted_f1']} | Acc: {metrics['accuracy']}")

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                self.best_model_name = name
                self.best_model = model

        print(f"[Task B] Selected Best Model: {self.best_model_name} (Macro-F1: {best_macro_f1})")
        return self.results

    def save_artifacts(self, models_dir: str = MODELS_DIR) -> None:
        """Serializes vectorizer, label encoder, and root cause models."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(models_dir, "rc_vectorizer.joblib"))
        joblib.dump(self.label_encoder, os.path.join(models_dir, "rc_label_encoder.joblib"))

        for name, model in self.models.items():
            filename = f"rc_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, os.path.join(models_dir, filename))

        joblib.dump(self.best_model, os.path.join(models_dir, "best_rc_model.joblib"))
        metadata = {
            "best_model_name": self.best_model_name,
            "classes": self.class_names,
            "random_state": self.random_state,
            "results": self.results
        }
        joblib.dump(metadata, os.path.join(models_dir, "rc_classification_meta.joblib"))
        print(f"[Task B] Artifacts saved to {models_dir}")
