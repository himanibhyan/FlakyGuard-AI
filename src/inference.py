"""
Inference and Model Serving Engine for FlakyGuard AI.
Loads persisted artifacts with caching and provides high-speed inference
and token-level explanations for the Streamlit dashboard and test suite.
"""

import os
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

MACRO_CATEGORY_MAP = {
    "Async Wait": "Timing & Asynchrony",
    "Time": "Timing & Asynchrony",
    "Test Case Timeout": "Timing & Asynchrony",
    "Test Suite Timeout": "Timing & Asynchrony",
    "Concurrency": "Concurrency & Non-Determinism",
    "Randomness": "Concurrency & Non-Determinism",
    "Unordered Collections": "Concurrency & Non-Determinism",
    "Network": "Resource & Environment",
    "Resource Leak": "Resource & Environment",
    "Platform Dependency": "Resource & Environment",
    "Test Order Dependency": "Test Logic & Assertions",
    "Floating Point Operations": "Test Logic & Assertions",
    "Too Restrictive Range": "Test Logic & Assertions",
}



class FlakyGuardEngine:
    """Encapsulates cached inference for Flakiness Detection and Root-Cause Classification."""

    _instance = None

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir

        # Load Task A artifacts
        self.flaky_vectorizer = joblib.load(os.path.join(models_dir, "flaky_vectorizer.joblib"))
        self.flaky_model = joblib.load(os.path.join(models_dir, "best_flaky_model.joblib"))
        self.flaky_meta = joblib.load(os.path.join(models_dir, "flaky_detection_meta.joblib"))

        # Load all Task A candidate models
        self.flaky_models = {
            "Random Forest": joblib.load(os.path.join(models_dir, "flaky_random_forest.joblib")),
            "Logistic Regression": joblib.load(os.path.join(models_dir, "flaky_logistic_regression.joblib")),
            "Linear SVM": joblib.load(os.path.join(models_dir, "flaky_linear_svm.joblib"))
        }

        # Load Task B artifacts
        self.rc_vectorizer = joblib.load(os.path.join(models_dir, "rc_vectorizer.joblib"))
        self.rc_label_encoder = joblib.load(os.path.join(models_dir, "rc_label_encoder.joblib"))
        self.rc_model = joblib.load(os.path.join(models_dir, "best_rc_model.joblib"))
        self.rc_meta = joblib.load(os.path.join(models_dir, "rc_classification_meta.joblib"))
        self.rc_classes = list(self.rc_label_encoder.classes_)

        self.rc_models = {
            "Logistic Regression": joblib.load(os.path.join(models_dir, "rc_logistic_regression.joblib")),
            "Linear SVM": joblib.load(os.path.join(models_dir, "rc_linear_svm.joblib"))
        }

    @classmethod
    def get_instance(cls, models_dir: str = MODELS_DIR) -> "FlakyGuardEngine":
        if cls._instance is None:
            cls._instance = FlakyGuardEngine(models_dir)
        return cls._instance

    def predict_flakiness(
        self,
        title: str,
        description: str,
        language: str = "",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs Task A inference.
        Returns prediction, calibrated probability, confidence, and influential words.
        """
        # Construct clean text
        text_parts = []
        if language:
            text_parts.append(language.lower())
        if title:
            text_parts.append(title.strip())
        if description:
            text_parts.append(description.strip())
        combined_text = " ".join(text_parts).strip()
        if not combined_text:
            combined_text = "test failure"

        model = self.flaky_models.get(model_name, self.flaky_model)
        X = self.flaky_vectorizer.transform([combined_text])

        pred_int = int(model.predict(X)[0])
        pred_label = "Flaky" if pred_int == 1 else "False Positive"

        # Obtain probability and confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            flaky_prob = float(probs[1])
            fp_prob = float(probs[0])
            confidence = float(max(flaky_prob, fp_prob))
            has_calibrated_prob = True
        else:
            flaky_prob = 1.0 if pred_int == 1 else 0.0
            confidence = 1.0
            has_calibrated_prob = False

        # Compute influential words
        influential_words = self._extract_token_weights(
            combined_text,
            self.flaky_vectorizer,
            model
        )

        return {
            "is_flaky": bool(pred_int == 1),
            "prediction": pred_label,
            "flaky_probability": round(flaky_prob, 4),
            "false_positive_probability": round(1.0 - flaky_prob, 4),
            "confidence": round(confidence, 4),
            "has_calibrated_probability": has_calibrated_prob,
            "model_used": model_name or self.flaky_meta["best_model_name"],
            "influential_words": influential_words,
            "input_text": combined_text
        }

    def predict_root_cause(
        self,
        title: str,
        description: str,
        language: str = "",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs Task B inference on confirmed flaky issues.
        Returns predicted category and full probability distribution.
        """
        text_parts = []
        if language:
            text_parts.append(language.lower())
        if title:
            text_parts.append(title.strip())
        if description:
            text_parts.append(description.strip())
        combined_text = " ".join(text_parts).strip()
        if not combined_text:
            combined_text = "flaky test timing issue"

        model = self.rc_models.get(model_name, self.rc_model)
        X = self.rc_vectorizer.transform([combined_text])

        pred_idx = int(model.predict(X)[0])
        pred_category = self.rc_classes[pred_idx]

        probs_dist = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            for cls_name, p in zip(self.rc_classes, probs):
                probs_dist[cls_name] = float(round(p, 4))
            confidence = float(round(max(probs), 4))
        else:
            probs_dist[pred_category] = 1.0
            confidence = 1.0

        # Sort distribution descending
        sorted_probs = dict(sorted(probs_dist.items(), key=lambda x: x[1], reverse=True))

        macro_category = MACRO_CATEGORY_MAP.get(pred_category, "Other / Unknown")

        return {
            "predicted_root_cause": pred_category,
            "predicted_macro_category": macro_category,
            "confidence": confidence,
            "probability_distribution": sorted_probs,
            "model_used": model_name or self.rc_meta["best_model_name"]
        }

    def _extract_token_weights(
        self,
        text: str,
        vectorizer,
        model,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Extracts influential tokens from the input text using linear weights or tree importance."""
        feature_names = np.array(vectorizer.get_feature_names_out())
        X = vectorizer.transform([text])
        nonzero = X.nonzero()[1]
        if len(nonzero) == 0:
            return []

        results = []
        if hasattr(model, "coef_"):
            weights = model.coef_[0]
            for idx in nonzero:
                tfidf = X[0, idx]
                w = weights[idx]
                contrib = tfidf * w
                results.append({
                    "word": feature_names[idx],
                    "weight": round(float(w), 4),
                    "contribution": round(float(contrib), 4),
                    "direction": "Flaky" if contrib > 0 else "False Positive"
                })
        elif hasattr(model, "calibrated_classifiers_"):
            coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_]
            avg_coef = np.mean(coefs, axis=0)
            for idx in nonzero:
                tfidf = X[0, idx]
                w = avg_coef[idx]
                contrib = tfidf * w
                results.append({
                    "word": feature_names[idx],
                    "weight": round(float(w), 4),
                    "contribution": round(float(contrib), 4),
                    "direction": "Flaky" if contrib > 0 else "False Positive"
                })
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for idx in nonzero:
                imp = importances[idx]
                tfidf = X[0, idx]
                results.append({
                    "word": feature_names[idx],
                    "weight": round(float(imp), 4),
                    "contribution": round(float(tfidf * imp), 4),
                    "direction": "Flaky Predictor"
                })

        results.sort(key=lambda x: abs(x.get("contribution", 0)), reverse=True)
        return results[:top_k]


def main():
    """Command-line interface for FlakyGuard AI inference."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="FlakyGuard AI: Flaky Test Detection & Root-Cause Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference.py --title "TestTimeout flaky" --description "Intermittent socket timeout on CI"
  python src/inference.py --title "Goroutine leak" --description "Channel deadlock" --language "Go" --root-cause
        """
    )
    parser.add_argument("--title", type=str, default="", help="GitHub issue title")
    parser.add_argument("--description", type=str, default="", help="GitHub issue description or error log")
    parser.add_argument("--language", type=str, default="", help="Programming language (e.g. Python, Java, Go, JS, C)")
    parser.add_argument("--model", type=str, default=None, choices=["Random Forest", "Logistic Regression", "Linear SVM"], help="Model override")
    parser.add_argument("--root-cause", action="store_true", help="Also predict root-cause category")
    parser.add_argument("--json", action="store_true", help="Output results formatted as JSON")

    args = parser.parse_args()

    if not args.title and not args.description:
        print("[!] Warning: Neither --title nor --description provided. Running with sample diagnostic input.")
        args.title = "TestRaceCondition flaky under heavy CI load"
        args.description = "Async wait deadline exceeded; intermittent race condition in goroutine execution."
        args.language = "Go"

    engine = FlakyGuardEngine.get_instance()

    # 1. Binary Flakiness Prediction
    res = engine.predict_flakiness(
        title=args.title,
        description=args.description,
        language=args.language,
        model_name=args.model
    )

    # 2. Root Cause Prediction (if requested or if predicted flaky)
    rc_res = None
    if args.root_cause:
        rc_res = engine.predict_root_cause(
            title=args.title,
            description=args.description,
            language=args.language
        )

    if args.json:
        output_payload = {
            "binary_detection": res,
            "root_cause_classification": rc_res,
            "limitations_notice": (
                "Predictions reflect lexical patterns in developer-written issue text, "
                "not dynamic runtime JVM/OS execution traces. High confidence indicates "
                "strong text correlation, not an absolute guarantee of flakiness."
            )
        }
        print(json.dumps(output_payload, indent=2))
        return

    # Formatted CLI output
    print("=" * 70)
    print(" FLAKYGUARD AI — DIAGNOSTIC INFERENCE REPORT")
    print("=" * 70)
    print(f"Input Title:       {args.title}")
    print(f"Language:          {args.language or 'Not Specified'}")
    print(f"Model Used:        {res['model_used']}")
    print("-" * 70)
    print(f"PREDICTION:        {res['prediction'].upper()}")
    print(f"Flaky Probability: {res['flaky_probability']:.2%}")
    print(f"Confidence Score:  {res['confidence']:.2%}")
    print("-" * 70)
    print("TOP INFLUENTIAL TOKENS:")
    for w in res["influential_words"][:5]:
        direction = w.get("direction", "N/A")
        contrib = w.get("contribution", 0.0)
        print(f"  • {w['word']:<18} | Weight: {w['weight']:>7.4f} | Contribution: {contrib:>7.4f} | Bias: {direction}")

    if rc_res:
        print("-" * 70)
        print("ROOT-CAUSE TAXONOMY PREDICTION (Task B):")
        print(f"Primary Root Cause: {rc_res['predicted_root_cause']}")
        print(f"Macro Category:     {rc_res['predicted_macro_category']}")
        print("Top Candidate Probabilities:")
        sorted_probs = sorted(rc_res["probability_distribution"].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in sorted_probs:
            print(f"  • {cat:<24} : {prob:.2%}")

    print("=" * 70)
    print("[!] LIMITATIONS NOTICE:")
    print("    This prediction is based on natural language keywords in developer issue text.")
    print("    It does not inspect dynamic runtime thread interleavings or memory access traces.")
    print("=" * 70)


if __name__ == "__main__":
    main()

