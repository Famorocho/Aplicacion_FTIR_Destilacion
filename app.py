from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from src.core import (
    InputValidationError,
    ModelCompatibilityError,
    build_excel_export,
    build_template,
    load_model_bundle,
    parse_spectrum,
    predict_spectrum,
    results_dataframe,
)


st.set_page_config(
    page_title="Curva de destilación por ATR-FTIR",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f7f9f8; }
      [data-testid="stSidebar"] { background: #eef4f0; }
      h1, h2, h3 { color: #123c2c; }
      .block-container { padding-top: 2rem; max-width: 1280px; }
      .app-subtitle { color: #49665b; font-size: 1.05rem; margin-top: -0.7rem; }
      .screening-note { border-left: 5px solid #e0a800; padding: .8rem 1rem; background: #fffaf0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model_bundle()


try:
    bundle, metadata = get_model()
except ModelCompatibilityError as exc:
    st.error(str(exc))
    st.stop()


st.title("Estimación de la curva de destilación ASTM D7169")
st.markdown(
    '<div class="app-subtitle">Predicción a partir de un espectro ATR-FTIR mediante PLS con 20 componentes</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Modelo")
    st.write("**PLS Regression:** 20 componentes")
    st.write(f"**Entrenamiento:** {metadata['training_samples']} muestras")
    st.write(f"**Prueba reservada:** {metadata['holdout_samples']} muestras")
    st.metric("RMSE global", f"{metadata['holdout_metrics']['rmse_c']:.2f} °C")
    st.metric("MAE global", f"{metadata['holdout_metrics']['mae_c']:.2f} °C")
    st.metric("R² global", f"{metadata['holdout_metrics']['r2']:.3f}")
    st.caption(f"Versión: {metadata['model_version']}")
    st.caption(f"Hash: {metadata['model_sha256'][:12]}...")
    st.divider()
    st.markdown("**Formatos de entrada**")
    st.write("• Excel o CSV horizontal: una fila y 601 columnas espectrales.")
    st.write("• Excel o CSV vertical: 601 filas con número de onda y absorbancia.")

st.markdown(
    '<div class="screening-note"><b>Alcance:</b> resultado para screening y caracterización preliminar. '
    'No sustituye el ensayo ASTM D7169 para control final.</div>',
    unsafe_allow_html=True,
)

st.subheader("1. Cargar espectro")
left, middle, right = st.columns([2, 1, 1])

with left:
    uploaded = st.file_uploader(
        "Seleccione un archivo con una sola muestra",
        type=["xlsx", "xlsm", "csv"],
        help="Tamaño máximo: 10 MB. La grilla debe coincidir con las 601 variables del modelo.",
    )

with middle:
    st.download_button(
        "Descargar plantilla Excel",
        data=build_template(metadata, "xlsx"),
        file_name="plantilla_espectro_ftir.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with right:
    st.download_button(
        "Descargar plantilla CSV",
        data=build_template(metadata, "csv"),
        file_name="plantilla_espectro_ftir.csv",
        mime="text/csv",
        width="stretch",
    )

if uploaded is None:
    st.info("Cargue un archivo para validar el espectro y generar la estimación.")
    st.stop()

raw = uploaded.getvalue()
file_key = hashlib.sha256(raw).hexdigest()

try:
    scaler = bundle["modelo"].named_steps["scaler"]
    parsed = parse_spectrum(raw, uploaded.name, metadata, scaler)
except InputValidationError as exc:
    st.error(f"Archivo rechazado: {exc}")
    st.stop()

sample_id = st.text_input(
    "Identificador de la muestra",
    value=parsed.sample_id,
    key=f"id-{file_key}",
)

if sample_id.strip() != parsed.sample_id:
    parsed = type(parsed)(
        sample_id.strip() or parsed.sample_id,
        parsed.values,
        parsed.wavenumbers,
        parsed.source_layout,
        parsed.qa,
    )

st.success(
    f"Estructura válida: {len(parsed.values)} absorbancias "
    f"en formato {parsed.source_layout.lower()}."
)

qa_col, spectrum_col = st.columns([1, 2])

with qa_col:
    st.markdown("**Control de entrada**")
    st.write(f"Estado estructural: **{parsed.qa['estado_estructura']}**")
    st.write("Valores finitos: **Sí**")
    st.write(f"Variables con |z| > 3: **{parsed.qa['variables_z_mayor_3']}**")
    st.write(f"Alerta preliminar de dominio: **{parsed.qa['dominio_preliminar']}**")
    st.caption(parsed.qa["nota_dominio"])

with spectrum_col:
    spectral_chart = pd.DataFrame(
        {
            "Número de onda (cm⁻¹)": parsed.wavenumbers,
            "Absorbancia": parsed.values,
        }
    )
    st.line_chart(
        spectral_chart,
        x="Número de onda (cm⁻¹)",
        y="Absorbancia",
        height=280,
    )

result = predict_spectrum(parsed, bundle, metadata)
table = results_dataframe(result, bundle["puntos"])

st.subheader("2. Resultado de la estimación")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("IBP", f"{result.delivered[0]:.1f} °C")
metric_2.metric("T50", f"{result.delivered[5]:.1f} °C")
metric_3.metric("T90", f"{result.delivered[9]:.1f} °C")
metric_4.metric("FBP/T100", f"{result.delivered[10]:.1f} °C")

chart_col, table_col = st.columns([1.25, 1])

with chart_col:
    curve_chart = table[
        ["Fracción destilada (%)", "Temperatura estimada (°C)"]
    ]
    st.line_chart(
        curve_chart,
        x="Fracción destilada (%)",
        y="Temperatura estimada (°C)",
        height=400,
    )

with table_col:
    st.dataframe(
        table[
            ["Punto", "Fracción destilada (%)", "Temperatura estimada (°C)"]
        ],
        hide_index=True,
        width="stretch",
        height=400,
    )

if result.monotonic_adjusted:
    st.warning(
        "La predicción original requirió corrección isotónica "
        "para garantizar monotonicidad."
    )
else:
    st.success(
        "La curva estimada es monótona y no requirió corrección."
    )

if parsed.qa["dominio_preliminar"] == "Advertencia":
    st.warning(
        "El espectro presenta diferencias elevadas frente al dominio de calibración. "
        "El resultado debe confirmarse mediante el ensayo de referencia."
    )

st.warning(
    "FBP/T100 es el punto de mayor incertidumbre del modelo "
    "(RMSE de validación: 24,93 °C). "
    "Debe confirmarse cuando soporte una decisión crítica."
)

export = build_excel_export(parsed, result, bundle["puntos"])

st.download_button(
    "Descargar resultado y trazabilidad",
    data=export,
    file_name=f"resultado_{result.sample_id.replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)

with st.expander("Ver trazabilidad técnica"):
    trace_df = pd.DataFrame(
        {
            "Campo": [str(x) for x in result.traceability.keys()],
            "Valor": [str(x) for x in result.traceability.values()],
        }
    )

    st.dataframe(
        trace_df,
        hide_index=True,
        width="stretch",
    )