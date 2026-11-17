import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from data_config import NUMERIC_COLS, CATEGORICAL_COLS, TEXT_COLS


def make_text(frame):
    return (
        frame["criteria"].fillna("").astype(str)
        + " "
        + frame["brief_summary"].fillna("").astype(str)
    ).str.strip()


class TrialFeatureBuilder:
    """Fit on training data only; reuse fitted transformations on calibration/test."""

    def __init__(self, include_text=True, max_text_features=1500, svd_components=40):
        self.include_text = include_text
        self.max_text_features = max_text_features
        self.svd_components = svd_components
        self.structured_transformer = None
        self.vectorizer = None
        self.svd = None
        self.feature_names = []

    def fit_transform(self, X):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                max_categories=50,
                sparse_output=False
            ))
        ])

        self.structured_transformer = ColumnTransformer([
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS)
        ], remainder="drop", sparse_threshold=0)

        structured = self.structured_transformer.fit_transform(X)
        names = list(self.structured_transformer.get_feature_names_out())

        if not self.include_text:
            self.feature_names = names
            return structured

        text = make_text(X)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_text_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.98,
            sublinear_tf=True
        )
        text_matrix = self.vectorizer.fit_transform(text)

        n_components = min(
            self.svd_components,
            text_matrix.shape[0] - 1,
            text_matrix.shape[1] - 1
        )

        if n_components >= 1:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            text_features = self.svd.fit_transform(text_matrix)
        else:
            text_features = np.zeros((len(X), 1))

        text_names = [f"text_component_{i+1}" for i in range(text_features.shape[1])]
        self.feature_names = names + text_names
        return np.hstack([structured, text_features])

    def transform(self, X):
        structured = self.structured_transformer.transform(X)

        if not self.include_text:
            return structured

        text_matrix = self.vectorizer.transform(make_text(X))
        if self.svd is not None:
            text_features = self.svd.transform(text_matrix)
        else:
            text_features = np.zeros((len(X), 1))

        return np.hstack([structured, text_features])
