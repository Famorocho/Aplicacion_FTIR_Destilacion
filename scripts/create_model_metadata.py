from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_excel(args.dataset, sheet_name="crudos", nrows=1)
    wavenumbers = [float(value) for value in frame.columns[1:-11]]
    metadata = {
        "application_version": "1.0.0",
        "model_name": "PLSRegression-20",
        "model_version": "pls20-holdout-2026-09",
        "model_sha256": sha256(args.model),
        "sklearn_version": "1.9.0",
        "training_samples": 574,
        "holdout_samples": 144,
        "n_features": 601,
        "n_outputs": 11,
        "wavenumber_tolerance_cm1": 0.02,
        "wavenumbers_cm1": wavenumbers,
        "points": ["IBP", "T10", "T20", "T30", "T40", "T50", "T60", "T70", "T80", "T90", "FBP_T100"],
        "holdout_metrics": {"mae_c": 8.6919, "rmse_c": 12.5151, "r2": 0.94024, "monotonic_curves_pct": 100.0},
        "limitations": [
            "Uso para screening y caracterización preliminar.",
            "No sustituye ASTM D7169 para control final.",
            "FBP/T100 presenta la mayor incertidumbre: RMSE 24.93 °C en hold-out.",
            "La alerta de dominio implementada es preliminar y no reemplaza una validación OOD formal.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

