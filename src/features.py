from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype


TARGET_COL = "Label"
DROP_COLS = ["SHA1", 'Magic', 'PE_TYPE', 'SizeOfOptionalHeader']
_DLL_RE = re.compile(r"^[a-z0-9_.-]+\.dll$")


@dataclass(frozen=True)
class SpecialCols:
    first_seen: str = "FirstSeenDate"
    identify: str = "Identify"
    imported_dlls: str = "ImportedDlls"
    imported_symbols: str = "ImportedSymbols"


def dll_tokenizer(doc: str):
    if not isinstance(doc, str):
        return []
    doc = doc.lower()
    doc = re.sub(r"[^\x20-\x7E]+", " ", doc)
    tokens = doc.split()
    return [t for t in tokens if _DLL_RE.match(t)]


class To1DText(BaseEstimator, TransformerMixin):
    """Convert ColumnTransformer (n,1) input to a clean 1D array of strings."""
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        arr = np.asarray(X).ravel()

        s = pd.Series(arr)
        s = s.fillna("").astype(str)

        return s.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["text"], dtype=object)
        return np.asarray(input_features, dtype=object)


class DatePartsTransformer(BaseEstimator, TransformerMixin):
    """
    Turns YYYY-MM-DD strings into numeric features: year, month.
    """
    def __init__(self):
        self.feature_names_ = ["first_seen_year", "first_seen_month"]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        dt = pd.to_datetime(s, errors="coerce")
        out = np.column_stack([
            dt.dt.year.fillna(0).astype(int),
            dt.dt.month.fillna(0).astype(int),
        ])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


class MissingFlagTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes missing vs present for a single column as 0/1.
    """
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        return s.isna().astype(int).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name], dtype=object)


class SymbolsSummaryTransformer(BaseEstimator, TransformerMixin):
    """
    Extract cheap numeric summaries from the huge ImportedSymbols string:
      - token count (split on whitespace)
    """
    def __init__(self):
        self.feature_names_ = ["token_count"]

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel()).fillna("")

        tokens = s.str.split()
        token_count = tokens.apply(len).astype(int)

        out = np.column_stack([token_count])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


def build_preprocessor(df: pd.DataFrame, cols: SpecialCols = SpecialCols()) -> ColumnTransformer:
    """
    Returns a sklearn ColumnTransformer that:
      - scales numeric columns
      - extracts date parts from FirstSeenDate
      - adds Identify missing flag
      - multi-hot encodes ImportedDlls via CountVectorizer(binary=True)
      - extracts summary features from ImportedSymbols
      - drops ID + raw string columns not meant for modeling
    """
    exclude = set([TARGET_COL] + DROP_COLS + [
        cols.first_seen, cols.identify, cols.imported_dlls, cols.imported_symbols
    ])

    numeric_cols = [
        c for c in df.columns
        if c not in exclude and is_numeric_dtype(df[c])
    ]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    dll_pipe = Pipeline(steps=[
        ("to_text", To1DText()),
        ("vectorizer", CountVectorizer(
            tokenizer=dll_tokenizer,
            lowercase=False, #lowercased in tokenizer
            token_pattern=None,
            binary=True
        )),
    ])

    transformers = [
        ("num", numeric_pipe, numeric_cols),

        ("first_seen_dateparts", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("dateparts", DatePartsTransformer()),
        ]), [cols.first_seen]),

        ("identify_missing", Pipeline(steps=[
            ("missingflag", MissingFlagTransformer("identify_missing")),
        ]), [cols.identify]),

        ("imported_dlls", dll_pipe, [cols.imported_dlls]),

        ("imported_symbols_summary", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("summaries", SymbolsSummaryTransformer()),
        ]), [cols.imported_symbols]),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Defines the contract: returns X (raw frame) and y (label).
    Preprocessing happens inside the sklearn pipeline, not here.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y
