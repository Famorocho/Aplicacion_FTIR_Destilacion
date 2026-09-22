from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "modelo_seleccionado.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


class InputValidationError(ValueError):
    """Error de estructura o calidad del espectro de entrada."""


class ModelCompatibilityError(RuntimeError):
    """Error de integridad o compatibilidad del modelo serializado."""


@dataclass(frozen=True)
class ParsedSpectrum:
    sample_id: str
    values: np.ndarray
    wavenumbers: np.ndarray
    source_layout: str
    qa: dict


@dataclass(frozen=True)
class PredictionResult:
    sample_id: str
    predicted: np.ndarray
    delivered: np.ndarray
    monotonic_adjusted: bool
    traceability: dict


def _normalize_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def load_metadata(path: Path = METADATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_bundle(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    *,
    strict_version: bool = True,
) -> tuple[dict, dict]:
    metadata = load_metadata(metadata_path)
    actual_hash = sha256_file(model_path)
    if actual_hash != metadata["model_sha256"]:
        raise ModelCompatibilityError(
            "El hash del modelo no coincide con el manifiesto. No se ejecutó la predicción."
        )

    expected_version = metadata["sklearn_version"]
    allow_mismatch = os.getenv("ALLOW_SKLEARN_VERSION_MISMATCH") == "1"
    if strict_version and not allow_mismatch and sklearn.__version__ != expected_version:
        raise ModelCompatibilityError(
            f"El modelo requiere scikit-learn {expected_version}; "
            f"el entorno tiene {sklearn.__version__}."
        )

    with warnings.catch_warnings():
        if allow_mismatch or not strict_version:
            warnings.simplefilter("ignore")
        bundle = joblib.load(model_path)

    required_keys = {"modelo", "nombre_modelo", "puntos", "semilla", "parametros"}
    if not required_keys.issubset(bundle):
        raise ModelCompatibilityError("El archivo del modelo no contiene el pipeline esperado.")
    if int(bundle["modelo"].n_features_in_) != metadata["n_features"]:
        raise ModelCompatibilityError("El número de variables del modelo no coincide con el manifiesto.")
    return bundle, metadata


def _read_csv_flexible(raw: bytes) -> pd.DataFrame:
    attempts: list[pd.DataFrame] = []
    for encoding in ("utf-8-sig", "latin-1"):
        for decimal in (".", ","):
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw), sep=None, engine="python", decimal=decimal, encoding=encoding
                )
                attempts.append(frame)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    if not attempts:
        raise InputValidationError("No fue posible interpretar el archivo CSV.")
    return max(attempts, key=lambda frame: (frame.shape[1], -frame.isna().sum().sum()))


def read_input_file(raw: bytes, filename: str) -> pd.DataFrame:
    if not raw:
        raise InputValidationError("El archivo está vacío.")
    if len(raw) > 10 * 1024 * 1024:
        raise InputValidationError("El archivo supera el límite de 10 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return _read_csv_flexible(raw)
    if suffix in {".xlsx", ".xlsm"}:
        try:
            return pd.read_excel(io.BytesIO(raw), sheet_name=0)
        except Exception as exc:
            raise InputValidationError(f"No fue posible interpretar el archivo Excel: {exc}") from exc
    raise InputValidationError("Formato no admitido. Use .xlsx, .xlsm o .csv.")


def _match_wavenumbers(found: np.ndarray, expected: np.ndarray, tolerance: float) -> np.ndarray:
    if len(found) != len(expected):
        raise InputValidationError(
            f"Se requieren {len(expected)} números de onda y se encontraron {len(found)}."
        )
    order: list[int] = []
    used: set[int] = set()
    for target in expected:
        distances = np.abs(found - target)
        idx = int(np.argmin(distances))
        if distances[idx] > tolerance or idx in used:
            raise InputValidationError(
                "La grilla espectral no coincide con las 601 variables usadas por el modelo."
            )
        used.add(idx)
        order.append(idx)
    return np.asarray(order, dtype=int)


def _to_numeric_vector(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(numeric).any():
        raise InputValidationError("Las absorbancias contienen celdas vacías o valores no numéricos.")
    if not np.isfinite(numeric).all():
        raise InputValidationError("Las absorbancias contienen valores infinitos.")
    return numeric


def _extract_horizontal(
    frame: pd.DataFrame, expected: np.ndarray, filename: str, tolerance: float
) -> tuple[str, np.ndarray]:
    if len(frame) != 1:
        raise InputValidationError("El formato horizontal debe contener exactamente una muestra.")

    id_aliases = {"CRUDO", "ID", "MUESTRA", "ID_MUESTRA", "CODIGO_MUESTRA"}
    target_aliases = {
        "IBP", "0_1", "0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9", "1",
        "T10", "T20", "T30", "T40", "T50", "T60", "T70", "T80", "T90", "FBP_T100",
    }
    sample_id = Path(filename).stem
    numeric_columns: list[object] = []
    found_wavenumbers: list[float] = []
    unknown_columns: list[str] = []

    for column in frame.columns:
        normalized = _normalize_label(column)
        if normalized in id_aliases:
            value = str(frame.iloc[0][column]).strip()
            if value and value.lower() != "nan":
                sample_id = value
            continue
        try:
            wavenumber = float(str(column).strip().replace(",", "."))
            numeric_columns.append(column)
            found_wavenumbers.append(wavenumber)
        except ValueError:
            if normalized not in target_aliases and not normalized.startswith("UNNAMED"):
                unknown_columns.append(str(column))

    if unknown_columns:
        preview = ", ".join(unknown_columns[:5])
        raise InputValidationError(f"Columnas no reconocidas: {preview}.")

    found = np.asarray(found_wavenumbers, dtype=float)
    order = _match_wavenumbers(found, expected, tolerance)
    values_found = _to_numeric_vector(frame.loc[frame.index[0], numeric_columns])
    return sample_id, values_found[order]


def _find_vertical_columns(frame: pd.DataFrame) -> tuple[object, object, object | None]:
    wave_aliases = {
        "NUMERO_ONDA", "NUMERO_DE_ONDA", "WAVENUMBER", "WAVE_NUMBER", "CM_1", "CM1"
    }
    absorbance_aliases = {"ABSORBANCIA", "ABSORBANCE", "ABS", "INTENSIDAD", "INTENSITY"}
    id_aliases = {"CRUDO", "ID", "MUESTRA", "ID_MUESTRA", "CODIGO_MUESTRA"}
    wave = absorbance = sample_id = None
    for column in frame.columns:
        normalized = _normalize_label(column)
        if normalized in wave_aliases:
            wave = column
        elif normalized in absorbance_aliases:
            absorbance = column
        elif normalized in id_aliases:
            sample_id = column
    if wave is None or absorbance is None:
        raise InputValidationError(
            "El formato vertical requiere las columnas 'numero_onda' y 'absorbancia'."
        )
    return wave, absorbance, sample_id


def _extract_vertical(
    frame: pd.DataFrame, expected: np.ndarray, filename: str, tolerance: float
) -> tuple[str, np.ndarray]:
    wave_col, absorbance_col, id_col = _find_vertical_columns(frame)
    wave = _to_numeric_vector(frame[wave_col])
    absorbance = _to_numeric_vector(frame[absorbance_col])
    order = _match_wavenumbers(wave, expected, tolerance)

    sample_id = Path(filename).stem
    if id_col is not None:
        ids = frame[id_col].dropna().astype(str).str.strip()
        ids = ids[ids.ne("")].unique()
        if len(ids) > 1:
            raise InputValidationError("El archivo vertical contiene más de un identificador de muestra.")
        if len(ids) == 1:
            sample_id = ids[0]
    return sample_id, absorbance[order]


def _quality_checks(values: np.ndarray, scaler) -> dict:
    if np.ptp(values) <= 1e-12 or float(np.std(values)) <= 1e-12:
        raise InputValidationError("El espectro no presenta variación; revise la medición o el archivo.")

    scale = np.where(np.asarray(scaler.scale_) == 0, 1.0, scaler.scale_)
    z_scores = np.abs((values - np.asarray(scaler.mean_)) / scale)
    count_over_3 = int(np.sum(z_scores > 3))
    count_over_6 = int(np.sum(z_scores > 6))
    max_z = float(np.max(z_scores))
    preliminary_domain = "Advertencia" if count_over_6 > 0 or count_over_3 > 30 else "Sin alerta"
    return {
        "estado_estructura": "Conforme",
        "valores_finitos": True,
        "desviacion_absorbancia": float(np.std(values)),
        "variables_z_mayor_3": count_over_3,
        "variables_z_mayor_6": count_over_6,
        "z_maximo": max_z,
        "dominio_preliminar": preliminary_domain,
        "nota_dominio": (
            "Comparación preliminar contra la media y desviación del escalador; "
            "no equivale a una detección OOD validada."
        ),
    }


def parse_spectrum(
    raw: bytes,
    filename: str,
    metadata: dict,
    scaler,
) -> ParsedSpectrum:
    frame = read_input_file(raw, filename)
    expected = np.asarray(metadata["wavenumbers_cm1"], dtype=float)
    tolerance = float(metadata.get("wavenumber_tolerance_cm1", 0.02))

    if len(frame) == 1:
        sample_id, values = _extract_horizontal(frame, expected, filename, tolerance)
        layout = "Horizontal"
    else:
        sample_id, values = _extract_vertical(frame, expected, filename, tolerance)
        layout = "Vertical"

    qa = _quality_checks(values, scaler)
    qa["formato_entrada"] = layout
    qa["variables_recibidas"] = int(len(values))
    return ParsedSpectrum(sample_id, values, expected, layout, qa)


def predict_spectrum(
    parsed: ParsedSpectrum,
    bundle: dict,
    metadata: dict,
) -> PredictionResult:
    predicted = np.asarray(bundle["modelo"].predict(parsed.values.reshape(1, -1))).reshape(-1)
    if len(predicted) != len(bundle["puntos"]):
        raise ModelCompatibilityError("El modelo no devolvió los 11 puntos esperados.")

    monotonic = np.all(np.diff(predicted) >= -1e-9)
    if monotonic:
        delivered = predicted.copy()
    else:
        delivered = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
            np.arange(len(predicted)), predicted
        )

    traceability = {
        "id_muestra": parsed.sample_id,
        "fecha_hora_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": metadata["model_name"],
        "version_modelo": metadata["model_version"],
        "hash_modelo_sha256": metadata["model_sha256"],
        "scikit_learn": metadata["sklearn_version"],
        "muestras_entrenamiento": metadata["training_samples"],
        "variables_entrada": metadata["n_features"],
        "semilla": bundle["semilla"],
        "ajuste_monotonia": not monotonic,
        "estado_qa": parsed.qa["estado_estructura"],
        "alerta_dominio_preliminar": parsed.qa["dominio_preliminar"],
    }
    return PredictionResult(parsed.sample_id, predicted, delivered, not monotonic, traceability)


def results_dataframe(result: PredictionResult, points: list[str]) -> pd.DataFrame:
    fractions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    display_points = ["IBP", "T10", "T20", "T30", "T40", "T50", "T60", "T70", "T80", "T90", "FBP/T100"]
    return pd.DataFrame(
        {
            "Punto": display_points,
            "Fracción destilada (%)": fractions,
            "Temperatura estimada (°C)": np.round(result.delivered, 2),
            "Observación": [""] * 10 + ["Mayor incertidumbre del modelo; confirmar cuando la decisión sea crítica."],
        }
    )


def build_excel_export(
    parsed: ParsedSpectrum,
    result: PredictionResult,
    points: list[str],
) -> bytes:
    output = io.BytesIO()
    result_table = results_dataframe(result, points)
    trace = pd.DataFrame(
        {"Campo": list(result.traceability.keys()), "Valor": list(result.traceability.values())}
    )
    qa = pd.DataFrame({"Control": list(parsed.qa.keys()), "Resultado": list(parsed.qa.values())})
    spectrum = pd.DataFrame(
        {"Número de onda (cm⁻¹)": parsed.wavenumbers, "Absorbancia": parsed.values}
    )
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_table.to_excel(writer, sheet_name="Resultados", index=False)
        trace.to_excel(writer, sheet_name="Trazabilidad", index=False)
        qa.to_excel(writer, sheet_name="QA", index=False)
        spectrum.to_excel(writer, sheet_name="Espectro", index=False)
        workbook = writer.book
        header = workbook.add_format(
            {"bold": True, "font_color": "white", "bg_color": "#145A32", "border": 1}
        )
        warning_fmt = workbook.add_format({"bg_color": "#FFF2CC", "text_wrap": True})
        for sheet_name, frame in {
            "Resultados": result_table,
            "Trazabilidad": trace,
            "QA": qa,
            "Espectro": spectrum,
        }.items():
            sheet = writer.sheets[sheet_name]
            for col, name in enumerate(frame.columns):
                sheet.write(0, col, name, header)
                width = min(max(len(str(name)) + 3, 16), 48)
                sheet.set_column(col, col, width)
            sheet.freeze_panes(1, 0)
        writer.sheets["Resultados"].set_column(3, 3, 52, warning_fmt)
    return output.getvalue()


def build_template(metadata: dict, file_format: str) -> bytes:
    columns = ["CRUDO", *[f"{value:.5f}" for value in metadata["wavenumbers_cm1"]]]
    template = pd.DataFrame([["MUESTRA-001", *([np.nan] * metadata["n_features"])]], columns=columns)
    if file_format == "csv":
        return template.to_csv(index=False).encode("utf-8-sig")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        template.to_excel(writer, sheet_name="espectro", index=False)
        sheet = writer.sheets["espectro"]
        sheet.freeze_panes(1, 1)
        sheet.set_column(0, 0, 18)
        sheet.set_column(1, len(columns) - 1, 12)
    return output.getvalue()

