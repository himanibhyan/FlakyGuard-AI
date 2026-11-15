"""
Flakiness Detection Module (Task A) for FlakyGuard AI.
Implements binary classification (T = Flaky vs F = False Positive) using:
1. Logistic Regression
2. Calibrated Linear Support Vector Machine (Linear SVM)
3. Random Forest Classifier
Conforms strictly to 80/20 stratified splitting and research requirements.
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import get_tfidf_vectorizer
from src.evaluation import evaluate_binary_classifier

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


class FlakyDetector:
    """Trains, evaluates, and serves Task A flakiness detection models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = get_tfidf_vectorizer()
        self.models: Dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
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
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.curves_data: Dict[str, Dict[str, np.ndarray]] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs 80/20 stratified train/test split and TF-IDF vectorization."""
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["combined_text"],
            df["label"].values,
            test_size=test_size,
            stratify=df["label"].values,
            random_state=self.random_state
        )

        # Fit vectorizer only on training set to prevent data leakage
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
        """Trains all candidate models, evaluates metrics, and selects best by Macro-F1."""
        best_macro_f1 = -1.0

        for name, model in self.models.items():
            print(f"[Task A: Flakiness Detection] Fitting {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_binary_classifier(y_test, y_pred, y_prob, name)
            self.results[name] = metrics
            self.curves_data[name] = {
                "y_true": y_test,
                "y_prob": y_prob
            }

            print(f"  -> {name} | Macro-F1: {metrics['macro_f1']} | F1: {metrics['f1_score']} | Acc: {metrics['accuracy']} | ROC-AUC: {metrics['roc_auc']}")

            # Model selection criterion: Macro-F1 (balanced harmonic mean)
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                self.best_model_name = name
                self.best_model = model

        print(f"[Task A] Selected Best Model: {self.best_model_name} (Macro-F1: {best_macro_f1})")
        return self.results

    def save_artifacts(self, models_dir: str = MODELS_DIR) -> None:
        """Serializes vectorizer, individual models, and designated best model."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(models_dir, "flaky_vectorizer.joblib"))

        for name, model in self.models.items():
            filename = f"flaky_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, os.path.join(models_dir, filename))

        joblib.dump(self.best_model, os.path.join(models_dir, "best_flaky_model.joblib"))
        metadata = {
            "best_model_name": self.best_model_name,
            "random_state": self.random_state,
            "results": self.results
        }
        joblib.dump(metadata, os.path.join(models_dir, "flaky_detection_meta.joblib"))
        print(f"[Task A] Artifacts saved to {models_dir}")
