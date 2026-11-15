"""
Explainability Module for FlakyGuard AI.
Implements local and global interpretability using LIME and exact feature weight analysis
for both Task A (Flakiness Detection) and Task B (Root-Cause Classification).
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "figures"))


class FlakyExplainer:
    """Provides local and global explanations for flakiness predictions."""

    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            self.class_names = ["False Positive", "Flaky"]
        else:
            self.class_names = class_names
        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names,
            bow=True,
            random_state=42
        )

    def explain_instance_lime(
        self,
        text: str,
        predict_proba_fn,
        num_features: int = 8
    ) -> List[Tuple[str, float]]:
        """Generates local LIME feature contributions for an input text."""
        exp = self.lime_explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            num_samples=250
        )
        return exp.as_list()

    @staticmethod
    def explain_instance_linear(
        text: str,
        vectorizer,
        model,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Computes exact token-level contribution for linear models (Logistic Regression / Linear SVM):
        contribution_i = x_i * w_i
        Sub-millisecond latency and mathematically exact to the model's decision function.
        """
        x_vec = vectorizer.transform([text])
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Unpack model weights
        if hasattr(model, "coef_"):
            coef = model.coef_[0]
        elif hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV: average base estimators
            coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_]
            coef = np.mean(coefs, axis=0)
        else:
            return []

        # Non-zero indices in input vector
        nonzero_indices = x_vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        tokens_data = []
        for idx in nonzero_indices:
            tfidf_val = x_vec[0, idx]
            weight = coef[idx]
            contribution = tfidf_val * weight
            tokens_data.append({
                "word": feature_names[idx],
                "tfidf": float(round(tfidf_val, 4)),
                "weight": float(round(weight, 4)),
                "contribution": float(round(contribution, 4)),
                "supports": "Flaky" if contribution > 0 else "False Positive"
            })

        tokens_data.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return tokens_data[:top_k]

    @staticmethod
    def plot_global_feature_importance(
        vectorizer,
        model,
        model_name: str,
        save_path: str,
        top_n: int = 15
    ) -> None:
        """Plots top positive and negative vocabulary weights for the model."""
        feature_names = np.array(vectorizer.get_feature_names_out())

        if hasattr(model, "coef_"):
            weights = model.coef_[0]
        elif hasattr(model, "calibrated_classifiers_"):
            coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_]
            weights = np.mean(coefs, axis=0)
        elif hasattr(model, "feature_importances_"):
            # Random Forest
            weights = model.feature_importances_
            top_indices = np.argsort(weights)[-top_n:]
            plt.figure(figsize=(8, 5))
            plt.barh(feature_names[top_indices], weights[top_indices], color="#4361EE")
            plt.title(f"Global Feature Importance: {model_name} (MDI)", weight="bold", pad=12)
            plt.xlabel("Importance")
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            plt.close()
            return
        else:
            return

        # Linear model weights
        top_positive_idx = np.argsort(weights)[-top_n:]
        top_negative_idx = np.argsort(weights)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # Flaky predictors
        axes[0].barh(feature_names[top_positive_idx], weights[top_positive_idx], color="#4361EE")
        axes[0].set_title(f"Top {top_n} Features Predicting Flaky (T)", weight="bold")
        axes[0].set_xlabel("Feature Weight (Coefficient)")

        # False positive predictors
        axes[1].barh(feature_names[top_negative_idx][::-1], weights[top_negative_idx][::-1], color="#E63946")
        axes[1].set_title(f"Top {top_n} Features Predicting False Positive (F)", weight="bold")
        axes[1].set_xlabel("Feature Weight (Coefficient)")

        plt.suptitle(f"Global Interpretability: {model_name} Feature Weights", weight="bold", fontsize=14)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
