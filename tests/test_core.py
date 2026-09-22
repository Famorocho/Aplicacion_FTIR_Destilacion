from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import pytest

from src.core import (
    InputValidationError,
    build_excel_export,
    load_model_bundle,
    parse_spectrum,
    predict_spectrum,
)


os.environ["ALLOW_SKLEARN_VERSION_MISMATCH"] = "1"


@pytest.fixture(scope="module")
def loaded():
    return load_model_bundle()


def horizontal_bytes(metadata, values, rows=1):
    columns = ["CRUDO", *[f"{x:.5f}" for x in metadata["wavenumbers_cm1"]]]
    content = [[f"TEST-{idx + 1}", *values] for idx in range(rows)]
    return pd.DataFrame(content, columns=columns).to_csv(index=False).encode("utf-8")


def test_horizontal_prediction_and_export(loaded):
    bundle, metadata = loaded
    scaler = bundle["modelo"].named_steps["scaler"]
    raw = horizontal_bytes(metadata, scaler.mean_)
    parsed = parse_spectrum(raw, "sample.csv", metadata, scaler)
    result = predict_spectrum(parsed, bundle, metadata)
    assert parsed.sample_id == "TEST-1"
    assert parsed.source_layout == "Horizontal"
    assert result.delivered.shape == (11,)
    assert np.all(np.diff(result.delivered) >= -1e-9)
    exported = build_excel_export(parsed, result, bundle["puntos"])
    assert exported[:2] == b"PK"


def test_vertical_format(loaded):
    bundle, metadata = loaded
    scaler = bundle["modelo"].named_steps["scaler"]
    vertical = pd.DataFrame(
        {
            "numero_onda": metadata["wavenumbers_cm1"],
            "absorbancia": scaler.mean_,
            "CRUDO": ["VERTICAL-1"] * metadata["n_features"],
        }
    )
    raw = vertical.to_csv(index=False).encode("utf-8")
    parsed = parse_spectrum(raw, "vertical.csv", metadata, scaler)
    assert parsed.sample_id == "VERTICAL-1"
    assert parsed.source_layout == "Vertical"


def test_rejects_more_than_one_horizontal_sample(loaded):
    bundle, metadata = loaded
    scaler = bundle["modelo"].named_steps["scaler"]
    raw = horizontal_bytes(metadata, scaler.mean_, rows=2)
    with pytest.raises(InputValidationError):
        parse_spectrum(raw, "two.csv", metadata, scaler)


def test_rejects_missing_spectral_column(loaded):
    bundle, metadata = loaded
    scaler = bundle["modelo"].named_steps["scaler"]
    frame = pd.read_csv(io.BytesIO(horizontal_bytes(metadata, scaler.mean_)))
    raw = frame.drop(columns=frame.columns[-1]).to_csv(index=False).encode("utf-8")
    with pytest.raises(InputValidationError):
        parse_spectrum(raw, "missing.csv", metadata, scaler)


def test_rejects_constant_spectrum(loaded):
    bundle, metadata = loaded
    scaler = bundle["modelo"].named_steps["scaler"]
    raw = horizontal_bytes(metadata, np.zeros(metadata["n_features"]))
    with pytest.raises(InputValidationError):
        parse_spectrum(raw, "constant.csv", metadata, scaler)

