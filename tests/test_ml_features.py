import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import (
    DatePartsTransformer,
    MissingFlagTransformer,
    SpecialCols,
    SymbolsSummaryTransformer,
    To1DText,
    build_preprocessor,
    dll_tokenizer,
    make_xy,
)


@pytest.fixture
def sample_df():
    base_row = {
        "Size": 1024,
        "Entropy": 5.5,
        "FirstSeenDate": "2024-01-01",
        "Identify": "packed_sample",
        "ImportedDlls": "kernel32.dll ntdll.dll",
        "ImportedSymbols": "CreateFileW CloseHandle",
        "SHA1": None,
        "Magic": None,
        "PE_TYPE": None,
        "SizeOfOptionalHeader": None,
        "Label": 1,
    }
    second_row = {
        **base_row,
        "Size": 2048,
        "Entropy": 3.2,
        "FirstSeenDate": "not-a-date",
        "Identify": None,
        "ImportedDlls": "NOT_A_DLL.TXT",
        "ImportedSymbols": "",
        "Label": 0,
    }
    return pd.DataFrame([base_row, second_row])


def test_dll_tokenizer_normalizes_and_filters():
    text = "KERNEL32.DLL some_lib.dlL notdll.exe"
    tokens = dll_tokenizer(text)
    assert tokens == ["kernel32.dll", "some_lib.dll"]


def test_to1dtext_handles_varied_inputs():
    transformer = To1DText()
    output = transformer.fit_transform([[None], [123], ["abc"]])
    assert output.dtype == object
    assert output.tolist() == ["", "123", "abc"]


def test_dateparts_transformer_coerces_invalid_values():
    transformer = DatePartsTransformer()
    out = transformer.fit_transform(["2025-03-15", None, ""],)
    assert out.shape == (3, 2)
    assert (out[0] == np.array([2025, 3])).all()
    assert (out[1] == np.array([0, 0])).all()


def test_missing_flag_reports_nans():
    transformer = MissingFlagTransformer("flag_missing")
    out = transformer.fit_transform([[None], ["text"], [0]])
    assert out.shape == (3, 1)
    assert out.tolist() == [[1], [0], [0]]


def test_symbols_summary_counts_tokens():
    transformer = SymbolsSummaryTransformer()
    out = transformer.fit_transform(["foo bar", None, "", "baz"])
    assert out.shape == (4, 1)
    assert out.flatten().tolist() == [2, 0, 0, 1]


def test_make_xy_requires_label():
    df = pd.DataFrame({"Size": [1, 2]})
    with pytest.raises(ValueError):
        make_xy(df)


def test_build_preprocessor_creates_numeric_features(sample_df):
    X, y = make_xy(sample_df)
    preprocessor = build_preprocessor(X, SpecialCols())
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()
    numeric_transformer = preprocessor.transformers_[0]
    numeric_cols = set(numeric_transformer[2])
    assert {"Size", "Entropy"}.issubset(numeric_cols)
