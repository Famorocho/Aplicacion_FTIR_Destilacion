# Estimación de la curva de destilación mediante ATR-FTIR

Aplicación Streamlit para estimar los 11 puntos de la curva de destilación ASTM D7169 a partir de un espectro ATR-FTIR. Utiliza el pipeline PLS con 20 componentes seleccionado durante el módulo de modelado.

## Alcance

La aplicación está destinada a screening y caracterización preliminar. No sustituye el ensayo ASTM D7169 para control final. El punto FBP/T100 tiene la mayor incertidumbre del modelo y se presenta con una advertencia específica.

El repositorio no contiene la base `PRO_CRUD2.xlsx`. Solo incluye el pipeline serializado y el manifiesto mínimo necesario para ejecutar inferencias reproducibles.

## Funcionalidades

- Carga de una sola muestra en Excel o CSV.
- Validación de las 601 variables espectrales, valores faltantes, duplicados, grilla y calidad básica.
- Compatibilidad con estructura horizontal o vertical.
- Predicción de IBP, T10, T20, T30, T40, T50, T60, T70, T80, T90 y FBP/T100.
- Verificación y corrección isotónica de monotonicidad cuando sea necesaria.
- Gráficas del espectro y de la curva estimada.
- Trazabilidad de la versión y hash del modelo.
- Descarga de resultados, controles QA, trazabilidad y espectro en Excel.

## Formatos de entrada

### Horizontal

Una fila por archivo:

| CRUDO | 692.39409 | 694.32277 | ... | 3078.16428 |
|---|---:|---:|---:|---:|
| MUESTRA-001 | valor | valor | ... | valor |

Las 601 columnas espectrales pueden estar en otro orden; la aplicación las reorganiza contra la grilla del modelo. Se admite una tolerancia de 0,02 cm⁻¹ en los encabezados.

### Vertical

Una fila por número de onda:

| CRUDO | numero_onda | absorbancia |
|---|---:|---:|
| MUESTRA-001 | 692.39409 | valor |
| MUESTRA-001 | 694.32277 | valor |

El archivo debe tener exactamente 601 mediciones y un único identificador de muestra.

## Ejecución local

Requiere Python 3.11. La versión de scikit-learn debe permanecer en 1.9.0 porque el modelo fue serializado con esa versión.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows, la activación del entorno es:

```powershell
.venv\Scripts\activate
```

## Ejecución con Docker

```bash
docker build -t ftir-destilacion .
docker run --rm -p 8501:8501 ftir-destilacion
```

La aplicación queda disponible en `http://localhost:8501`.

## Despliegue desde un repositorio privado

El contenedor puede desplegarse en un servidor corporativo o en un servicio de contenedores. Para Streamlit Community Cloud se debe conectar la cuenta de GitHub, autorizar acceso al repositorio privado y seleccionar `app.py` como archivo de entrada. La documentación oficial explica la [conexión con repositorios privados](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account) y el [proceso de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Antes de usar una nube externa, se debe confirmar que la política de manejo de información permite almacenar el modelo y procesar espectros fuera de la infraestructura corporativa.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

Las pruebas verifican ambos formatos de entrada, la predicción de 11 puntos, la monotonicidad, la exportación y el rechazo de archivos incompletos, constantes o con más de una muestra.

## Evidencia del modelo

- Entrenamiento: 574 muestras.
- Prueba reservada: 144 muestras.
- MAE global: 8,69 °C.
- RMSE global: 12,52 °C.
- R² global: 0,940.
- Curvas monótonas en hold-out: 100 %.
- FBP/T100: RMSE 24,93 °C; requiere advertencia y confirmación cuando la decisión sea crítica.

## Seguridad y trazabilidad

- El modelo se carga únicamente desde `artifacts/modelo_seleccionado.joblib`.
- Antes de cargarlo se verifica su hash SHA-256 contra `model_metadata.json`.
- La aplicación rechaza una versión de scikit-learn distinta de 1.9.0.
- No se debe aceptar ni cargar un archivo `joblib` suministrado por usuarios.
- Cada exportación registra ID de muestra, fecha UTC, versión, hash, QA y estado de monotonicidad.

