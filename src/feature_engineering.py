"""
Feature Engineering Module for FlakyGuard AI.
Implements TF-IDF vectorization conforming to research project specifications:
- max_features=10000
- ngram_range=(1, 2)
- min_df=2
- stop_words="english"
"""

import os
from typing import Tuple, Optional
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def get_tfidf_vectorizer(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    stop_words: str = "english"
) -> TfidfVectorizer:
    """Creates a TF-IDF vectorizer matching project specifications."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stop_words,
        sublinear_tf=True
    )


def fit_and_save_vectorizer(
    texts: pd.Series,
    save_path: str,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> Tuple[TfidfVectorizer, any]:
    """Fits vectorizer on corpus and serializes with joblib."""
    vectorizer = get_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    X = vectorizer.fit_transform(texts)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    return vectorizer, X


def load_vectorizer(load_path: str) -> TfidfVectorizer:
    """Loads a serialized vectorizer."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vectorizer artifact not found: {load_path}")
    return joblib.load(load_path)
