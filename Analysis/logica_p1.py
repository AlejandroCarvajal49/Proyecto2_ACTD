"""
Lógica de la Pregunta 1 - Equidad regional urbano/rural.

Responsable: Santiago Arias.

Este archivo contiene:
    - PARTE 1 (Proyecto 1): carga de datos, graficas descriptivas y prueba t.
    - PARTE 2 (Proyecto 2): pipeline predictivo completo (P3-style):
        * Multiples configuraciones de redes neuronales (regresion + clasificacion).
        * Multiples feature sets (basico / socioeconomico / completo).
        * Tracking en MLflow bajo experimento "pregunta_1".
        * Laboratorio de entrenamiento personalizado desde el tablero.
        * Simulador de escenarios A/B y analisis OLS de significancia.
"""

import itertools
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import tensorflow as tf


# ===========================================================================
# ===========================================================================
# PARTE 1 - PROYECTO 1: ANÁLISIS DESCRIPTIVO (sin cambios)
# ===========================================================================
# ===========================================================================

def cargar_datos_p1():
    # 1. Cargar datos de Saber 11
    df = pd.read_csv('Data/saber11_Antioquia_clean.csv', dtype={'cole_cod_mcpio_ubicacion': str})
    df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].str.upper().str.strip()

    # Manejar el nombre de la columna de área de residencia (puede variar según el dataset de ICFES)
    col_area = 'estu_areareside' if 'estu_areareside' in df.columns else 'cole_area_ubicacion'

    # Limpieza: Filtrar 'Sin Información' y estandarizar a Urbano/Rural
    if col_area in df.columns:
        df = df[~df[col_area].astype(str).str.upper().isin(['SIN INFORMACIÓN', 'SIN INFORMACION', 'NAN'])]
        df['Area'] = df[col_area].apply(
            lambda x: 'Urbano' if 'CABECERA' in str(x).upper() or 'URBAN' in str(x).upper() else 'Rural'
        )
    else:
        df['Area'] = 'Desconocido'

    # Limpieza: Tratamiento de nulos en estrato socioeconómico
    col_estrato = 'fami_estratovivienda'
    if col_estrato in df.columns:
        df = df.dropna(subset=[col_estrato])
        df = df[~df[col_estrato].astype(str).str.upper().isin(['SIN INFORMACION', 'SIN INFORMACIÓN'])]

    # 2. Cargar y cruzar datos de PIB
    try:
        df_pib = pd.read_csv('Data/PIB_municipios.csv', sep=None, engine='python', encoding='utf-8-sig')
        df_pib.columns = df_pib.columns.str.strip()
        df_pib['Municipio'] = df_pib['Municipio'].str.upper().str.strip()
        df = pd.merge(df, df_pib, left_on='cole_mcpio_ubicacion', right_on='Municipio', how='left')
    except KeyError as e:
        print(f"Error de columna en PIB_municipios.csv. Las columnas detectadas son: {df_pib.columns.tolist()}")
        df['PIB miles de millones'] = np.nan
    except FileNotFoundError:
        print("Advertencia: No se encontró 'PIB_municipios.csv'. Verifica la ruta.")
        df['PIB miles de millones'] = np.nan

    # 3. Cargar y cruzar coordenadas espaciales
    try:
        df_coord = pd.read_csv('Data/municipios_unicos.csv')
        df = pd.merge(df, df_coord, on='cole_mcpio_ubicacion', how='left')
    except FileNotFoundError:
        print("Advertencia: No se encontró 'municipios_unicos.csv'. Verifica la ruta.")
        df['lat'] = np.nan
        df['lon'] = np.nan

    return df


def obtener_lista_municipios_p1(df):
    municipios = df['cole_mcpio_ubicacion'].dropna().unique().tolist()
    municipios.sort()
    return ['TODOS'] + municipios


def generar_boxplot_brecha(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.box(
        dff,
        x='Area',
        y='punt_global',
        color='Area',
        title=f'Distribución del Puntaje Global: Urbano vs Rural ({municipio})',
        labels={'punt_global': 'Puntaje Global', 'Area': 'Zona de Residencia'},
        color_discrete_map={'Urbano': '#1f77b4', 'Rural': '#2ca02c'}
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def generar_dispersion_pib_brecha(df):
    agrupado = df.groupby(['cole_mcpio_ubicacion', 'Area'])['punt_global'].mean().unstack()

    if 'Urbano' in agrupado.columns and 'Rural' in agrupado.columns:
        agrupado['Brecha_Puntos'] = agrupado['Urbano'] - agrupado['Rural']
    else:
        agrupado['Brecha_Puntos'] = np.nan

    agrupado = agrupado.reset_index()

    if 'PIB miles de millones' in df.columns:
        pib_df = df[['cole_mcpio_ubicacion', 'PIB miles de millones']].drop_duplicates()
        agrupado = pd.merge(agrupado, pib_df, on='cole_mcpio_ubicacion', how='left')

        fig = px.scatter(
            agrupado,
            x='PIB miles de millones',
            y='Brecha_Puntos',
            hover_name='cole_mcpio_ubicacion',
            trendline='ols',
            title='Correlación: Brecha Urbano-Rural vs PIB Municipal',
            labels={
                'Brecha_Puntos': 'Brecha (Puntos Urbano - Rural)',
                'PIB miles de millones': 'PIB (Miles de Millones)'
            },
            opacity=0.7
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        return fig

    return px.scatter(title="Datos de PIB no disponibles para graficar")


def calcular_estadisticas_brecha(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    urbano = dff[dff['Area'] == 'Urbano']['punt_global'].dropna()
    rural = dff[dff['Area'] == 'Rural']['punt_global'].dropna()

    if len(urbano) < 2 or len(rural) < 2:
        return "Insight: No hay suficientes datos en ambas zonas para calcular la brecha estadística en este municipio."

    brecha = urbano.mean() - rural.mean()
    t_stat, p_val = stats.ttest_ind(urbano, rural, equal_var=False)
    significancia = "SIGNIFICATIVA" if p_val < 0.05 else "NO significativa"

    return f"Insight: Se evidencia una diferencia de {brecha:.1f} puntos promedio a favor de las zonas urbanas. La brecha es estadísticamente {significancia} (p-valor: {p_val:.4f})."


def generar_barras_brecha_error(df, municipio):
    dff = df.copy()

    if municipio == 'TODOS':
        agrupado = dff.groupby(['cole_mcpio_ubicacion', 'Area'])['punt_global'].agg(['mean', 'std']).reset_index()
        brechas = agrupado.pivot(index='cole_mcpio_ubicacion', columns='Area', values='mean').dropna()
        brechas['Diferencia'] = brechas['Urbano'] - brechas['Rural']
        top_municipios = brechas.sort_values(by='Diferencia', ascending=False).head(10).index
        dff_plot = agrupado[agrupado['cole_mcpio_ubicacion'].isin(top_municipios)]
        titulo = "Top 10 Municipios con Mayor Brecha (Promedio y Desviación)"
        x_col = 'cole_mcpio_ubicacion'
    else:
        dpto_promedio = df.groupby('Area')['punt_global'].agg(['mean', 'std']).reset_index()
        dpto_promedio['cole_mcpio_ubicacion'] = 'PROMEDIO ANTIOQUIA'
        mpio_promedio = dff[dff['cole_mcpio_ubicacion'] == municipio].groupby('Area')['punt_global'].agg(['mean', 'std']).reset_index()
        mpio_promedio['cole_mcpio_ubicacion'] = municipio
        dff_plot = pd.concat([mpio_promedio, dpto_promedio])
        titulo = f"Comparación Local vs Departamental ({municipio})"
        x_col = 'cole_mcpio_ubicacion'

    fig = px.bar(
        dff_plot,
        x=x_col,
        y='mean',
        color='Area',
        barmode='group',
        error_y='std',
        title=titulo,
        labels={'mean': 'Puntaje Global Promedio', x_col: 'Municipio/Región', 'Area': 'Zona'},
        color_discrete_map={'Urbano': '#1f77b4', 'Rural': '#2ca02c'}
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def generar_mapa_pib_puntaje(df, municipio):
    dff = df.copy()

    df_mapa = dff.groupby(['cole_mcpio_ubicacion', 'lat', 'lon']).agg(
        punt_global=('punt_global', 'mean'),
        pib=('PIB miles de millones', 'first')
    ).reset_index()

    min_puntaje = df_mapa['punt_global'].min()
    max_puntaje = df_mapa['punt_global'].max()

    if municipio != 'TODOS':
        df_mapa['opacidad'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 1.0 if x == municipio else 0.15)
        df_mapa['tamano'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 15 if x == municipio else 5)
    else:
        df_mapa['opacidad'] = 0.8
        df_mapa['tamano'] = 8

    fig = px.scatter_mapbox(
        df_mapa,
        lat='lat',
        lon='lon',
        color='punt_global',
        hover_name='cole_mcpio_ubicacion',
        hover_data={
            'punt_global': ':.1f',
            'pib': ':.2f',
            'lat': False,
            'lon': False,
            'tamano': False,
            'opacidad': False
        },
        labels={'punt_global': 'Puntaje Global Prom.', 'pib': 'PIB (Miles de Millones)'},
        color_continuous_scale='Viridis',
        range_color=[min_puntaje, max_puntaje],
        mapbox_style='carto-positron',
        zoom=6.0,
        center={"lat": 6.2518, "lon": -75.5636},
        title=f'Mapa Espacial: Puntaje Global y PIB ({municipio})',
        size='tamano',
        size_max=15
    )

    fig.update_traces(marker=dict(opacity=df_mapa['opacidad']))
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig


# ===========================================================================
# ===========================================================================
# PARTE 2 - PROYECTO 2: PIPELINE PREDICTIVO (REFACTOR P3-STYLE)
# ===========================================================================
# ===========================================================================
#
# Esta seccion implementa el laboratorio de modelos para responder operativamente:
#   "Existe una brecha rural/urbana que justifique intervencion diferenciada
#    en municipios con menor PIB?"
#
# Convenciones:
#   - Experiment MLflow:  "pregunta_1"   (bajo mlruns/pregunta_1/)
#   - Mejores modelos:    models/pregunta_1/<task>/best/
#                         + custom (entrenados desde la UI):
#                         models/pregunta_1/custom/<task>/best/
#   - Tasks:              "regresion" (punt_global)
#                         "clasificacion_binaria" (prioridad alta = punt_global < P25)
# ===========================================================================

MODELS_DIR = "models"
P1_EXPERIMENT_NAME = "pregunta_1"

# Variables que el usuario podra ingresar en el simulador del tablero.
COLUMNAS_CATEGORICAS_P2 = [
    "Area",                    # Urbano / Rural
    "fami_estratovivienda",    # Estrato 1..6
    "fami_educacionpadre",     # Nivel educativo del padre
    "fami_educacionmadre",     # Nivel educativo de la madre
    "cole_naturaleza",         # Oficial / No oficial
    "cole_jornada",            # Manana, tarde, completa, sabatina, noche
    "cole_genero",             # Masculino, femenino, mixto
]

COLUMNAS_NUMERICAS_P2 = [
    "PIB miles de millones",
    "PIB per capita",
]

# Etiquetas legibles para variables del simulador (usadas en la UI).
ETIQUETAS_VARIABLES_P1 = {
    "Area": "Zona (Urbano/Rural)",
    "fami_estratovivienda": "Estrato vivienda",
    "fami_educacionpadre": "Educacion padre",
    "fami_educacionmadre": "Educacion madre",
    "cole_naturaleza": "Naturaleza colegio",
    "cole_jornada": "Jornada",
    "cole_genero": "Genero colegio",
    "PIB miles de millones": "PIB municipal",
    "PIB per capita": "PIB per capita",
}


# ===========================================================================
# 2.1 HELPERS BASICOS
# ===========================================================================

def _first_present_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def _serializable_params(config):
    params = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            params[key] = "-".join([str(v) for v in value])
        else:
            params[key] = value
    return params


def _coerce_numeric_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _sanitize_missing_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return df


# ===========================================================================
# 2.2 FEATURE SETS
# ===========================================================================
#
# Tres conjuntos progresivamente mas ricos. El laboratorio entrena los modelos
# con cada uno y selecciona el mejor por tarea (igual que P3).
# ===========================================================================

def _build_feature_sets_p1(df):
    feature_sets = {
        "basico": {
            "numeric": ["PIB miles de millones"],
            "categorical": ["Area", "fami_estratovivienda"],
        },
        "socioeconomico": {
            "numeric": ["PIB miles de millones", "PIB per capita"],
            "categorical": [
                "Area",
                "fami_estratovivienda",
                "fami_educacionpadre",
                "fami_educacionmadre",
            ],
        },
        "completo": {
            "numeric": COLUMNAS_NUMERICAS_P2,
            "categorical": COLUMNAS_CATEGORICAS_P2,
        },
    }
    for key, cfg in feature_sets.items():
        cfg["numeric"] = [c for c in cfg["numeric"] if c in df.columns]
        cfg["categorical"] = [c for c in cfg["categorical"] if c in df.columns]
    return feature_sets


# ===========================================================================
# 2.3 PATHS Y MLflow (URI, UI launcher, info)
# ===========================================================================

def _p1_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p1_models_dir():
    return os.path.join(_p1_base_dir(), "models", "pregunta_1")


def _resolve_mlflow_tracking_uri_p1(base_dir):
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return (Path(base_dir) / "mlruns" / "pregunta_1").as_uri()


def obtener_mlflow_info_p1():
    base_dir = _p1_base_dir()
    mlruns_path = Path(base_dir) / "mlruns" / "pregunta_1"
    tracking_uri = _resolve_mlflow_tracking_uri_p1(base_dir)
    ui_url = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
    return {
        "tracking_uri": tracking_uri,
        "mlruns_path": str(mlruns_path),
        "ui_url": ui_url,
        "experiment": P1_EXPERIMENT_NAME,
    }


def _parse_mlflow_ui_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000
    return host, port


def verificar_mlflow_ui_p1():
    info = obtener_mlflow_info_p1()
    host, port = _parse_mlflow_ui_url(info["ui_url"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True, f"MLflow UI activo en {info['ui_url']}"
        except Exception:
            return False, f"MLflow UI no responde en {info['ui_url']}"


def iniciar_mlflow_ui_p1():
    info = obtener_mlflow_info_p1()
    tracking_uri = info["tracking_uri"]
    host, port = _parse_mlflow_ui_url(info["ui_url"])

    is_up, message = verificar_mlflow_ui_p1()
    if is_up:
        return True, message

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--host", host,
        "--port", str(port),
    ]
    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "cwd": _p1_base_dir(),
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        subprocess.Popen(cmd, **kwargs)
        return True, f"Iniciando MLflow UI en {info['ui_url']}"
    except Exception as exc:
        return False, f"No se pudo iniciar MLflow UI ({exc})."


def _artifact_uri_to_path(uri):
    if uri.startswith("file:"):
        parsed = urlparse(uri)
        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/"):
            path = path.lstrip("/")
        return Path(path)
    return Path(uri)


# ===========================================================================
# 2.4 ARQUITECTURA MLP GENERICA + CONFIGURACIONES
# ===========================================================================
#
# `_crear_mlp_p1` arma una red densa parametrizada por un dict de config
# (layers, activation, dropout, l2, optimizer, lr, loss). Se usan tanto en el
# laboratorio (entrenamiento desde la UI) como en el script de entrenamiento.
# ===========================================================================

def _crear_optimizer_p1(tf, config):
    lr = float(config.get("learning_rate", 1e-3))
    opt = str(config.get("optimizer", "adam")).lower()
    if opt == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if opt == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _loss_from_config(tf, loss_name):
    if loss_name == "huber":
        return tf.keras.losses.Huber()
    return loss_name


def _crear_mlp_p1(tf, input_dim, config, output_dim, output_activation, loss, metrics=None):
    regularizer = None
    l2_val = float(config.get("l2", 0.0))
    if l2_val > 0:
        regularizer = tf.keras.regularizers.l2(l2_val)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in config.get("layers", []):
        model.add(tf.keras.layers.Dense(
            int(units),
            activation=config.get("activation", "relu"),
            kernel_regularizer=regularizer,
        ))
        if float(config.get("dropout", 0.0)) > 0:
            model.add(tf.keras.layers.Dropout(float(config["dropout"])))
    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    model.compile(optimizer=_crear_optimizer_p1(tf, config), loss=loss, metrics=metrics or [])
    return model


def _configuraciones_modelos_p1():
    return {
        "regresion": [
            {
                "name": "mlp_reg_mse_base",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "mse",
                "epochs": 30,
                "batch_size": 512,
            },
            {
                "name": "mlp_reg_huber",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.25,
                "l2": 1e-3,
                "optimizer": "rmsprop",
                "learning_rate": 0.0005,
                "loss": "huber",
                "epochs": 30,
                "batch_size": 256,
            },
            {
                "name": "mlp_reg_mae_shallow",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "l2": 0.0,
                "optimizer": "adam",
                "learning_rate": 0.002,
                "loss": "mae",
                "epochs": 25,
                "batch_size": 512,
            },
        ],
        "clasificacion_binaria": [
            {
                "name": "mlp_bin_base",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.3,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "binary_crossentropy",
                "epochs": 30,
                "batch_size": 512,
            },
            {
                "name": "mlp_bin_deep",
                "layers": [128, 64, 32, 16],
                "activation": "elu",
                "dropout": 0.35,
                "l2": 1e-4,
                "optimizer": "rmsprop",
                "learning_rate": 0.0005,
                "loss": "binary_crossentropy",
                "epochs": 35,
                "batch_size": 256,
            },
            {
                "name": "mlp_bin_shallow",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-3,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "binary_crossentropy",
                "epochs": 25,
                "batch_size": 512,
            },
        ],
    }


# ===========================================================================
# 2.5 EVALUADORES
# ===========================================================================

def _evaluar_regresion(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _evaluar_clasificacion_binaria(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        metricas["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metricas["roc_auc"] = None
    return metricas


# ===========================================================================
# 2.6 PERSISTENCIA: MEJOR MODELO POR TASK + CARGADOR
# ===========================================================================
#
# Los mejores modelos viven en:
#     models/pregunta_1/<task>/best/             (entrenamiento offline)
#     models/pregunta_1/custom/<task>/best/      (entrenamiento desde la UI)
# Cada directorio contiene: model.keras, preprocessor.pkl, metadata.json.
# ===========================================================================

def _guardar_mejor_modelo_p1(task, model, preprocessor, config, metrics,
                              feature_columns, umbral_p25=None, base_dir=None):
    if base_dir is None:
        base_dir = _p1_models_dir()
    task_dir = os.path.join(base_dir, task, "best")
    os.makedirs(task_dir, exist_ok=True)

    model.save(os.path.join(task_dir, "model.keras"))
    joblib.dump(preprocessor, os.path.join(task_dir, "preprocessor.pkl"))

    metadata = {
        "task": task,
        "config": _serializable_params(config),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "umbral_p25": umbral_p25,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def cargar_modelos_p1():
    """
    Carga los mejores modelos guardados. Prioriza la carpeta `custom/`
    (entrenamiento desde el laboratorio) sobre la base. Devuelve siempre
    un dict con flag `disponible` + `error` legible.
    """
    base_dir = _p1_models_dir()
    custom_dir = os.path.join(base_dir, "custom")
    tasks = ["regresion", "clasificacion_binaria"]

    artefactos = {"disponible": True, "error": None}

    try:
        import tensorflow as _tf
    except Exception as exc:
        return {"disponible": False, "error": f"TensorFlow no disponible ({exc})."}

    for task in tasks:
        candidatos = [
            ("custom", os.path.join(custom_dir, task, "best")),
            ("base", os.path.join(base_dir, task, "best")),
        ]
        seleccionado = None
        for source, task_dir in candidatos:
            model_path = os.path.join(task_dir, "model.keras")
            preproc_path = os.path.join(task_dir, "preprocessor.pkl")
            metadata_path = os.path.join(task_dir, "metadata.json")
            if all(os.path.exists(p) for p in (model_path, preproc_path, metadata_path)):
                seleccionado = (source, model_path, preproc_path, metadata_path)
                break

        if seleccionado is None:
            return {
                "disponible": False,
                "error": (
                    f"No se encontro el mejor modelo para '{task}'. "
                    f"Ejecute el entrenamiento o use el laboratorio en el tablero."
                ),
            }

        source, model_path, preproc_path, metadata_path = seleccionado
        try:
            artefactos[task] = {
                "model": _tf.keras.models.load_model(model_path),
                "preprocessor": joblib.load(preproc_path),
                "metadata": json.load(open(metadata_path, "r", encoding="utf-8")),
                "source": source,
            }
        except Exception as exc:
            return {"disponible": False, "error": f"Error cargando {task}: {exc}"}

    return artefactos


# ===========================================================================
# 2.7 OPCIONES PARA EL SIMULADOR + INPUT BUILDERS + PIB LOOKUP
# ===========================================================================

def obtener_opciones_simulador_p1(df):
    """Categorias unicas de cada variable, ya limpias para el dropdown."""
    opciones = {}
    for col in COLUMNAS_CATEGORICAS_P2:
        if col in df.columns:
            valores = df[col].dropna().astype(str).str.strip().unique().tolist()
            valores = [v for v in valores if v.upper() not in
                       ("NAN", "SIN INFORMACION", "SIN INFORMACION", "")]
            valores.sort()
            opciones[col] = valores
        else:
            opciones[col] = []
    return opciones


def construir_input_p1(valores_form, df_base=None):
    """
    Toma el dict del formulario y arma un DataFrame de una fila con TODAS las
    columnas que cualquiera de los preprocesadores podria pedir. El
    preprocesador del modelo elegido se encarga de seleccionar las que use.
    Si `df_base` se pasa y el form trae `municipio`, completa el PIB.
    """
    pib_m = valores_form.get("PIB miles de millones")
    pib_pc = valores_form.get("PIB per capita")
    if (pib_m is None or pib_pc is None) and df_base is not None and valores_form.get("municipio"):
        pib_m_lookup, pib_pc_lookup = obtener_pib_municipio(df_base, valores_form["municipio"])
        if pib_m is None:
            pib_m = pib_m_lookup
        if pib_pc is None:
            pib_pc = pib_pc_lookup

    fila = {
        "Area": valores_form.get("Area"),
        "fami_estratovivienda": valores_form.get("fami_estratovivienda"),
        "fami_educacionpadre": valores_form.get("fami_educacionpadre"),
        "fami_educacionmadre": valores_form.get("fami_educacionmadre"),
        "cole_naturaleza": valores_form.get("cole_naturaleza"),
        "cole_jornada": valores_form.get("cole_jornada"),
        "cole_genero": valores_form.get("cole_genero"),
        "PIB miles de millones": pib_m,
        "PIB per capita": pib_pc,
    }
    return pd.DataFrame([fila])


def obtener_pib_municipio(df, municipio):
    """Devuelve (PIB miles de millones, PIB per capita) del municipio o medianas."""
    if municipio and "cole_mcpio_ubicacion" in df.columns:
        sub = df[df["cole_mcpio_ubicacion"] == municipio]
        if not sub.empty:
            pib = sub["PIB miles de millones"].dropna().median()
            per_cap = (sub["PIB per capita"].dropna().median()
                       if "PIB per capita" in df.columns else np.nan)
            if pd.notna(pib):
                return float(pib), float(per_cap) if pd.notna(per_cap) else None

    pib_med = (df["PIB miles de millones"].dropna().median()
               if "PIB miles de millones" in df.columns else 0.0)
    per_med = (df["PIB per capita"].dropna().median()
               if "PIB per capita" in df.columns else None)
    return float(pib_med), float(per_med) if per_med is not None and pd.notna(per_med) else None


# ===========================================================================
# 2.8 CARGA DE RESULTADOS DESDE MLflow
# ===========================================================================

def cargar_resultados_mlflow_p1(max_runs=200):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        return pd.DataFrame(), f"MLflow no disponible ({exc})."

    base_dir = _p1_base_dir()
    mlflow.set_tracking_uri(_resolve_mlflow_tracking_uri_p1(base_dir))
    client = MlflowClient()

    exp = client.get_experiment_by_name(P1_EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame(), f"No se encontro el experimento '{P1_EXPERIMENT_NAME}' en MLflow."

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )
    rows = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags
        rows.append({
            "task": tags.get("task"),
            "config_id": run.info.run_name,
            "model": tags.get("model_name") or params.get("name"),
            "feature_set": tags.get("feature_set"),
            "selected_vars": tags.get("selected_vars"),
            "layers": params.get("layers"),
            "dropout": params.get("dropout"),
            "activation": params.get("activation"),
            "optimizer": params.get("optimizer"),
            "learning_rate": params.get("learning_rate"),
            "epochs": params.get("epochs"),
            "batch_size": params.get("batch_size"),
            "loss": params.get("loss"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "run_id": run.info.run_id,
        })

    resumen = pd.DataFrame(rows)
    if resumen.empty:
        return resumen, "No hay corridas registradas en MLflow."
    return resumen, f"Corridas cargadas: {len(resumen)}"


def cargar_historial_mlflow_p1(resumen_df, max_runs=30):
    """Recupera el history.csv (loss/val_loss por epoca) de cada run para graficar."""
    if resumen_df.empty:
        return pd.DataFrame()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception:
        return pd.DataFrame()

    base_dir = _p1_base_dir()
    mlflow.set_tracking_uri(_resolve_mlflow_tracking_uri_p1(base_dir))
    client = MlflowClient()

    rows = []
    for _, row in resumen_df.head(max_runs).iterrows():
        run_id = row.get("run_id")
        if not run_id:
            continue
        try:
            run = client.get_run(run_id)
            history_dir = _artifact_uri_to_path(run.info.artifact_uri) / "history"
            if not history_dir.exists():
                continue
            csvs = list(history_dir.glob("*.csv"))
            if not csvs:
                continue
            hist = pd.read_csv(csvs[0])
        except Exception:
            continue

        loss_serie = hist.get("loss", [])
        val_loss_serie = hist.get("val_loss", [None] * len(loss_serie))
        for idx, loss in enumerate(loss_serie, start=1):
            rows.append({
                "epoch": idx,
                "loss": loss,
                "val_loss": val_loss_serie[idx - 1] if idx - 1 < len(val_loss_serie) else None,
                "run_label": row.get("config_id") or run.info.run_name,
                "task": row.get("task"),
            })
    return pd.DataFrame(rows)


def seleccionar_mejores_modelos_resultados_p1(resumen_df):
    if resumen_df.empty:
        return pd.DataFrame()

    mejores = []
    for task, group in resumen_df.groupby("task"):
        if task == "regresion":
            group = group.dropna(subset=["rmse"])
            if group.empty:
                continue
            best = group.sort_values(["rmse", "r2"], ascending=[True, False]).head(1)
            metric, value = "rmse", float(best["rmse"].iloc[0])
        else:
            group = group.dropna(subset=["f1"])
            if group.empty:
                continue
            best = group.sort_values("f1", ascending=False).head(1)
            metric, value = "f1", float(best["f1"].iloc[0])

        mejores.append({
            "task": task,
            "model": best["model"].iloc[0],
            "feature_set": best["feature_set"].iloc[0],
            "metric": metric,
            "metric_value": round(value, 4),
            "run_id": best["run_id"].iloc[0],
        })
    return pd.DataFrame(mejores)


def seleccionar_top_modelos_resultados_p1(resumen_df, top_n=2):
    if resumen_df.empty:
        return pd.DataFrame()

    rows = []
    for task, group in resumen_df.groupby("task"):
        if task == "regresion":
            metric, label = "rmse", "RMSE"
            group = group.dropna(subset=[metric]).sort_values([metric, "r2"], ascending=[True, False])
        else:
            metric, label = "f1", "F1"
            group = group.dropna(subset=[metric]).sort_values(metric, ascending=False)

        if group.empty:
            continue
        for rank, (_, row) in enumerate(group.head(top_n).iterrows(), start=1):
            rows.append({
                "task": task,
                "rank": rank,
                "model": row.get("model"),
                "feature_set": row.get("feature_set"),
                "selected_vars": row.get("selected_vars"),
                "metric": label,
                "metric_value": round(float(row.get(metric)), 4) if pd.notna(row.get(metric)) else None,
                "config_id": row.get("config_id"),
                "run_id": row.get("run_id"),
            })
    return pd.DataFrame(rows)


def construir_figuras_comparativas_p1(resumen_df, history_df):
    fig_reg, fig_bin, fig_loss = go.Figure(), go.Figure(), go.Figure()

    if resumen_df.empty:
        fig_reg.update_layout(title="Sin corridas de regresion")
        fig_bin.update_layout(title="Sin corridas de clasificacion binaria")
        fig_loss.update_layout(title="Sin curvas de entrenamiento")
        return fig_reg, fig_bin, fig_loss

    reg = resumen_df[resumen_df["task"] == "regresion"].copy()
    if not reg.empty:
        fig_reg = px.bar(
            reg, x="config_id", y="rmse", color="feature_set",
            title="RMSE por configuracion (regresion - menor es mejor)",
            labels={"config_id": "Configuracion", "rmse": "RMSE"},
        )

    bin_df = resumen_df[resumen_df["task"] == "clasificacion_binaria"].copy()
    if not bin_df.empty:
        fig_bin = px.bar(
            bin_df, x="config_id", y="f1", color="feature_set",
            title="F1 por configuracion (clasificacion binaria - mayor es mejor)",
            labels={"config_id": "Configuracion", "f1": "F1"},
        )

    if not history_df.empty and "loss" in history_df.columns:
        fig_loss = px.line(
            history_df, x="epoch", y="loss", color="run_label",
            title="Perdida por epoca (todas las corridas)",
            labels={"epoch": "Epoca", "loss": "Loss"},
        )
        if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
            fig_loss.add_traces(px.line(
                history_df.dropna(subset=["val_loss"]),
                x="epoch", y="val_loss", color="run_label",
                line_dash_sequence=["dash"],
            ).data)

    for fig in (fig_reg, fig_bin, fig_loss):
        fig.update_layout(margin={"r": 10, "t": 40, "l": 10, "b": 10})
    return fig_reg, fig_bin, fig_loss


# ===========================================================================
# 2.9 LABORATORIO DE ENTRENAMIENTO PERSONALIZADO
# ===========================================================================

def _build_preprocessor_p1(num_features, cat_features):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if num_features:
        transformers.append((
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())]),
            num_features,
        ))
    if cat_features:
        transformers.append((
            "cat",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
            cat_features,
        ))
    return ColumnTransformer(transformers=transformers) if transformers else None


def entrenar_modelos_p1_personalizados(df, variables, max_rows=20000, random_state=42,
                                        umbral_p25=None):
    """
    Entrena el catalogo de configuraciones con las variables que el usuario
    elige en la UI. Registra cada corrida en MLflow (mlruns/pregunta_1/) y
    guarda el mejor modelo por task en models/pregunta_1/custom/<task>/best/.
    """
    try:
        import mlflow
        from sklearn.model_selection import train_test_split
        import tensorflow as _tf
    except Exception as exc:
        empty = pd.DataFrame()
        return empty, empty, empty, "", f"Dependencias no disponibles ({exc})."

    if not variables:
        empty = pd.DataFrame()
        return empty, empty, empty, "", "Selecciona al menos una variable."

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, "", "No hay datos para modelar."

    df_pred = df.copy()
    if max_rows and len(df_pred) > max_rows:
        df_pred = df_pred.sample(n=max_rows, random_state=random_state)

    numeric_vars = [v for v in COLUMNAS_NUMERICAS_P2 if v in variables and v in df_pred.columns]
    categorical_vars = [v for v in COLUMNAS_CATEGORICAS_P2 if v in variables and v in df_pred.columns]
    features = numeric_vars + categorical_vars
    if not features:
        empty = pd.DataFrame()
        return empty, empty, empty, "", "Las variables seleccionadas no estan disponibles."

    base_dir = _p1_base_dir()
    mlflow.set_tracking_uri(_resolve_mlflow_tracking_uri_p1(base_dir))
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs(Path(base_dir) / "mlruns" / "pregunta_1", exist_ok=True)
    mlflow.set_experiment(P1_EXPERIMENT_NAME)

    configs = _configuraciones_modelos_p1()
    results, history_rows, errores = [], [], []
    best_tracker = {
        "regresion": {"metric": np.inf},
        "clasificacion_binaria": {"metric": -np.inf},
    }

    if umbral_p25 is None and "punt_global" in df_pred.columns:
        umbral_p25 = float(pd.to_numeric(df_pred["punt_global"], errors="coerce").quantile(0.25))

    feature_set_name = "custom"
    selected_vars_tag = ",".join(features)
    custom_models_dir = os.path.join(_p1_models_dir(), "custom")

    # ----------------------- REGRESION -----------------------
    if "punt_global" in df_pred.columns:
        data_reg = df_pred[features + ["punt_global"]].dropna(subset=["punt_global"])
        if not data_reg.empty:
            X = data_reg[features]
            y = pd.to_numeric(data_reg["punt_global"], errors="coerce")
            mask = y.notna()
            X, y = X.loc[mask], y.loc[mask]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)

            X_tr = _coerce_numeric_columns(X_tr, numeric_vars)
            X_te = _coerce_numeric_columns(X_te, numeric_vars)
            X_tr = _sanitize_missing_columns(X_tr, categorical_vars)
            X_te = _sanitize_missing_columns(X_te, categorical_vars)

            preprocessor = _build_preprocessor_p1(numeric_vars, categorical_vars)
            X_tr_proc = _to_dense(preprocessor.fit_transform(X_tr))
            X_te_proc = _to_dense(preprocessor.transform(X_te))

            for idx, config in enumerate(configs["regresion"], start=1):
                run_name = f"p1_custom_reg_{idx}"
                config_id = f"custom_reg_{idx}"
                try:
                    with mlflow.start_run(run_name=run_name):
                        loss_fn = _loss_from_config(_tf, config["loss"])
                        model = _crear_mlp_p1(
                            _tf, X_tr_proc.shape[1], config,
                            output_dim=1, output_activation="linear", loss=loss_fn,
                        )
                        history = model.fit(
                            X_tr_proc, y_tr,
                            validation_split=0.15,
                            epochs=config["epochs"], batch_size=config["batch_size"],
                            verbose=0,
                            callbacks=[_tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
                        )
                        preds = model.predict(X_te_proc, verbose=0).reshape(-1)
                        metricas = _evaluar_regresion(y_te, preds)

                        mlflow.set_tag("task", "regresion")
                        mlflow.set_tag("feature_set", feature_set_name)
                        mlflow.set_tag("model_name", config["name"])
                        mlflow.set_tag("selected_vars", selected_vars_tag)
                        mlflow.log_params(_serializable_params(config))
                        mlflow.log_metrics(metricas)
                        run_id = mlflow.active_run().info.run_id

                        with tempfile.TemporaryDirectory() as tmpdir:
                            hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                            pd.DataFrame(history.history).to_csv(hist_path, index=False)
                            mlflow.log_artifact(hist_path, artifact_path="history")
                            cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                            with open(cfg_path, "w", encoding="utf-8") as f:
                                json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                            mlflow.log_artifact(cfg_path, artifact_path="config")
                            model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                            model.save(model_path)
                            mlflow.log_artifact(model_path, artifact_path="model")
                            preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                            joblib.dump(preprocessor, preproc_path)
                            mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                        for ep, loss in enumerate(history.history.get("loss", []), start=1):
                            val_losses = history.history.get("val_loss", [None] * len(history.history.get("loss", [])))
                            history_rows.append({
                                "task": "regresion", "feature_set": feature_set_name,
                                "config_id": config_id, "run_id": run_id,
                                "run_label": f"reg-{config_id}",
                                "epoch": ep, "loss": loss,
                                "val_loss": val_losses[ep - 1] if ep - 1 < len(val_losses) else None,
                            })

                        results.append({
                            "task": "regresion", "model": config["name"],
                            "feature_set": feature_set_name, "config_id": config_id,
                            "layers": "-".join([str(x) for x in config["layers"]]),
                            "dropout": config["dropout"], "activation": config["activation"],
                            "optimizer": config["optimizer"], "learning_rate": config["learning_rate"],
                            "epochs": config["epochs"], "batch_size": config["batch_size"],
                            "loss": config["loss"],
                            "rmse": metricas["rmse"], "mae": metricas["mae"], "r2": metricas["r2"],
                            "accuracy": None, "f1": None, "roc_auc": None,
                            "run_id": run_id,
                        })

                        if metricas["rmse"] < best_tracker["regresion"]["metric"]:
                            best_tracker["regresion"] = {"metric": metricas["rmse"], "metrics": metricas}
                            _guardar_mejor_modelo_p1(
                                "regresion", model, preprocessor, config, metricas,
                                features, umbral_p25=umbral_p25, base_dir=custom_models_dir,
                            )
                except Exception as exc:
                    errores.append(f"{run_name}: {exc}")

    # ----------------------- CLASIFICACION BINARIA -----------------------
    if "punt_global" in df_pred.columns and umbral_p25 is not None:
        df_pred["__prioridad_alta__"] = (
            pd.to_numeric(df_pred["punt_global"], errors="coerce") < umbral_p25
        ).astype(int)
        data_bin = df_pred[features + ["__prioridad_alta__"]].dropna(subset=["__prioridad_alta__"])
        if not data_bin.empty and data_bin["__prioridad_alta__"].nunique() > 1:
            X = data_bin[features]
            y = data_bin["__prioridad_alta__"]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y,
            )
            X_tr = _coerce_numeric_columns(X_tr, numeric_vars)
            X_te = _coerce_numeric_columns(X_te, numeric_vars)
            X_tr = _sanitize_missing_columns(X_tr, categorical_vars)
            X_te = _sanitize_missing_columns(X_te, categorical_vars)

            preprocessor = _build_preprocessor_p1(numeric_vars, categorical_vars)
            X_tr_proc = _to_dense(preprocessor.fit_transform(X_tr))
            X_te_proc = _to_dense(preprocessor.transform(X_te))

            for idx, config in enumerate(configs["clasificacion_binaria"], start=1):
                run_name = f"p1_custom_bin_{idx}"
                config_id = f"custom_bin_{idx}"
                try:
                    with mlflow.start_run(run_name=run_name):
                        model = _crear_mlp_p1(
                            _tf, X_tr_proc.shape[1], config,
                            output_dim=1, output_activation="sigmoid",
                            loss=config["loss"], metrics=["accuracy"],
                        )
                        history = model.fit(
                            X_tr_proc, y_tr,
                            validation_split=0.15,
                            epochs=config["epochs"], batch_size=config["batch_size"],
                            verbose=0,
                            callbacks=[_tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
                        )
                        proba = model.predict(X_te_proc, verbose=0).reshape(-1)
                        pred = (proba >= 0.5).astype(int)
                        metricas = _evaluar_clasificacion_binaria(y_te, pred, proba)

                        mlflow.set_tag("task", "clasificacion_binaria")
                        mlflow.set_tag("feature_set", feature_set_name)
                        mlflow.set_tag("model_name", config["name"])
                        mlflow.set_tag("selected_vars", selected_vars_tag)
                        mlflow.log_params(_serializable_params(config))
                        mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None})
                        run_id = mlflow.active_run().info.run_id

                        with tempfile.TemporaryDirectory() as tmpdir:
                            hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                            pd.DataFrame(history.history).to_csv(hist_path, index=False)
                            mlflow.log_artifact(hist_path, artifact_path="history")
                            cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                            with open(cfg_path, "w", encoding="utf-8") as f:
                                json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                            mlflow.log_artifact(cfg_path, artifact_path="config")
                            model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                            model.save(model_path)
                            mlflow.log_artifact(model_path, artifact_path="model")
                            preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                            joblib.dump(preprocessor, preproc_path)
                            mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                        for ep, loss in enumerate(history.history.get("loss", []), start=1):
                            val_losses = history.history.get("val_loss", [None] * len(history.history.get("loss", [])))
                            history_rows.append({
                                "task": "clasificacion_binaria", "feature_set": feature_set_name,
                                "config_id": config_id, "run_id": run_id,
                                "run_label": f"bin-{config_id}",
                                "epoch": ep, "loss": loss,
                                "val_loss": val_losses[ep - 1] if ep - 1 < len(val_losses) else None,
                            })

                        results.append({
                            "task": "clasificacion_binaria", "model": config["name"],
                            "feature_set": feature_set_name, "config_id": config_id,
                            "layers": "-".join([str(x) for x in config["layers"]]),
                            "dropout": config["dropout"], "activation": config["activation"],
                            "optimizer": config["optimizer"], "learning_rate": config["learning_rate"],
                            "epochs": config["epochs"], "batch_size": config["batch_size"],
                            "loss": config["loss"],
                            "rmse": None, "mae": None, "r2": None,
                            "accuracy": metricas["accuracy"], "f1": metricas["f1"],
                            "roc_auc": metricas.get("roc_auc"),
                            "run_id": run_id,
                        })

                        if metricas["f1"] > best_tracker["clasificacion_binaria"]["metric"]:
                            best_tracker["clasificacion_binaria"] = {"metric": metricas["f1"], "metrics": metricas}
                            _guardar_mejor_modelo_p1(
                                "clasificacion_binaria", model, preprocessor, config, metricas,
                                features, umbral_p25=umbral_p25, base_dir=custom_models_dir,
                            )
                except Exception as exc:
                    errores.append(f"{run_name}: {exc}")

    resumen_df = pd.DataFrame(results)
    history_df = pd.DataFrame(history_rows)
    mejores_df = seleccionar_mejores_modelos_resultados_p1(resumen_df)
    interpretacion = generar_interpretacion_p1(resumen_df, mejores_df, df_pred)

    if resumen_df.empty:
        return resumen_df, history_df, mejores_df, interpretacion, "No se generaron corridas."

    mensaje = f"Corridas personalizadas: {len(resumen_df)}"
    if errores:
        mensaje = f"{mensaje} | Errores: {len(errores)}"
    return resumen_df, history_df, mejores_df, interpretacion, mensaje


# ===========================================================================
# 2.10 SIMULADOR A/B
# ===========================================================================

def predecir_escenarios_p1(valores_a, valores_b, df_base=None):
    """
    Corre las predicciones de regresion + clasificacion para dos escenarios
    distintos (perfiles A vs B) y devuelve un dict por escenario. Si los
    modelos no existen, devuelve un mensaje de error en el ultimo campo.
    """
    artefactos = cargar_modelos_p1()
    if not artefactos.get("disponible"):
        return None, None, None, None, artefactos.get("error")

    def _pred_uno(valores):
        df_input = construir_input_p1(valores, df_base=df_base)

        reg = artefactos["regresion"]
        feat_reg = reg["metadata"]["feature_columns"]
        X_reg = _to_dense(reg["preprocessor"].transform(df_input[feat_reg]))
        puntaje = float(reg["model"].predict(X_reg, verbose=0).reshape(-1)[0])

        clf = artefactos["clasificacion_binaria"]
        feat_clf = clf["metadata"]["feature_columns"]
        X_clf = _to_dense(clf["preprocessor"].transform(df_input[feat_clf]))
        proba = float(clf["model"].predict(X_clf, verbose=0).reshape(-1)[0])

        return {
            "puntaje": puntaje,
            "proba_prioridad_alta": proba,
            "etiqueta": "Prioridad ALTA" if proba >= 0.5 else "Prioridad estándar",
        }

    pred_a = _pred_uno(valores_a)
    pred_b = _pred_uno(valores_b)
    fig_reg, fig_clf = grafica_comparacion_escenarios_p1(pred_a, pred_b)
    return pred_a, pred_b, fig_reg, fig_clf, None


def grafica_comparacion_escenarios_p1(pred_a, pred_b):
    if not pred_a or not pred_b:
        empty = go.Figure().update_layout(title="Modelos no disponibles")
        return empty, empty

    df_reg = pd.DataFrame({
        "Escenario": ["A", "B"],
        "Puntaje Global": [pred_a["puntaje"], pred_b["puntaje"]],
    })
    fig_reg = px.bar(
        df_reg, x="Escenario", y="Puntaje Global",
        text="Puntaje Global", color="Escenario",
        title="Comparacion de escenarios - Puntaje global predicho",
        color_discrete_map={"A": "#1f77b4", "B": "#c0392b"},
    )
    fig_reg.update_traces(texttemplate="%{text:.1f}", textposition="outside")

    df_clf = pd.DataFrame({
        "Escenario": ["A", "B"],
        "Prob. prioridad alta (%)": [pred_a["proba_prioridad_alta"] * 100,
                                       pred_b["proba_prioridad_alta"] * 100],
    })
    fig_clf = px.bar(
        df_clf, x="Escenario", y="Prob. prioridad alta (%)",
        text="Prob. prioridad alta (%)", color="Escenario",
        title="Comparacion de escenarios - Probabilidad de prioridad alta",
        color_discrete_map={"A": "#1f77b4", "B": "#c0392b"},
    )
    fig_clf.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    for fig in (fig_reg, fig_clf):
        fig.update_layout(margin={"r": 10, "t": 40, "l": 10, "b": 10}, showlegend=False)
    return fig_reg, fig_clf


# ===========================================================================
# 2.11 ESTIMADOR DE IMPACTO ZONA (contrafactual Urbano vs Rural agregado)
# ===========================================================================

def estimar_impacto_zona_p1(modelo, preprocessor, feature_columns, base_df, n_muestras=800):
    """
    Para una muestra del dataset, predice el puntaje con Area='Urbano' y con
    Area='Rural' manteniendo el resto fijo. Devuelve el impacto medio (Urbano-Rural).
    """
    if base_df.empty or "Area" not in feature_columns:
        return None
    missing = [c for c in feature_columns if c not in base_df.columns]
    if missing:
        return None

    muestra = base_df[feature_columns].dropna()
    if muestra.empty:
        return None

    muestra = muestra.sample(n=min(n_muestras, len(muestra)), random_state=42)
    base_urb = muestra.copy(); base_urb["Area"] = "Urbano"
    base_rur = muestra.copy(); base_rur["Area"] = "Rural"

    try:
        X_urb = _to_dense(preprocessor.transform(base_urb[feature_columns]))
        X_rur = _to_dense(preprocessor.transform(base_rur[feature_columns]))
        pred_urb = modelo.predict(X_urb, verbose=0).reshape(-1)
        pred_rur = modelo.predict(X_rur, verbose=0).reshape(-1)
        impacto = float(np.nanmean(pred_urb - pred_rur))
        if np.isnan(impacto):
            return None
        return impacto
    except Exception:
        return None


# ===========================================================================
# 2.12 ANALISIS DE SIGNIFICANCIA OLS (t-test y F-test)
# ===========================================================================

def analizar_significancia_ols_p1(df, variables_seleccionadas, sample_rows=20000):
    """Pruebas t (numericas individuales) y F (variables agrupadas + combinaciones)."""
    try:
        import statsmodels.api as sm
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), f"Statsmodels no disponible ({exc})."

    if df.empty or "punt_global" not in df.columns:
        return {}, pd.DataFrame(), pd.DataFrame(), "No hay datos para regresion."

    dff = df.copy()
    y = pd.to_numeric(dff["punt_global"], errors="coerce")
    mask = y.notna()
    dff = dff.loc[mask].copy()
    y = y.loc[mask]

    if sample_rows and len(dff) > sample_rows:
        dff = dff.sample(n=sample_rows, random_state=42)
        y = y.loc[dff.index]

    numeric_vars = [v for v in COLUMNAS_NUMERICAS_P2 if v in dff.columns]
    categorical_vars = [v for v in COLUMNAS_CATEGORICAS_P2 if v in dff.columns]

    dff = _coerce_numeric_columns(dff, numeric_vars)
    for col in numeric_vars:
        dff[col] = dff[col].fillna(dff[col].median())
    for col in categorical_vars:
        dff[col] = dff[col].astype("string").fillna("SIN_INFO")

    X_num = dff[numeric_vars]
    X_cat = (pd.get_dummies(dff[categorical_vars], prefix=categorical_vars,
                             prefix_sep="__", drop_first=True)
             if categorical_vars else pd.DataFrame(index=dff.index))
    X_full = pd.concat([X_num, X_cat], axis=1).loc[y.index]
    X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y_num = pd.to_numeric(y, errors="coerce").astype(float)

    if X_full.empty or y_num.isna().all():
        return {}, pd.DataFrame(), pd.DataFrame(), "No hay datos numericos suficientes para OLS."

    try:
        model_full = sm.OLS(y_num, sm.add_constant(X_full)).fit()
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), f"OLS fallo ({exc})."

    group_map = {}
    for var in numeric_vars:
        if var in X_full.columns:
            group_map[var] = [var]
    for var in categorical_vars:
        prefix = f"{var}__"
        cols = [c for c in X_full.columns if c.startswith(prefix)]
        if cols:
            group_map[var] = cols

    selected_vars = variables_seleccionadas or []
    selected_cols = []
    for var in selected_vars:
        selected_cols.extend(group_map.get(var, []))
    selected_cols = list(dict.fromkeys(selected_cols))
    if not selected_cols:
        return {}, pd.DataFrame(), pd.DataFrame(), "Selecciona al menos una variable."

    X_sel = X_full[selected_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    try:
        model_sel = sm.OLS(y_num, sm.add_constant(X_sel)).fit()
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), f"OLS sobre seleccion fallo ({exc})."

    resumen = {
        "n_obs": int(model_sel.nobs),
        "r2": float(model_sel.rsquared),
        "adj_r2": float(model_sel.rsquared_adj),
    }

    ttest_rows = []
    for var in numeric_vars:
        if var not in selected_vars or var not in model_sel.params:
            continue
        ttest_rows.append({
            "variable": var,
            "coef": float(model_sel.params[var]),
            "t_stat": float(model_sel.tvalues[var]),
            "p_value": float(model_sel.pvalues[var]),
            "significativa": "Si" if model_sel.pvalues[var] < 0.05 else "No",
        })
    ttest_df = pd.DataFrame(ttest_rows)

    ftest_rows = []
    exog_names = model_full.model.exog_names

    def _f_test_for_columns(cols):
        idx = [exog_names.index(col) for col in cols if col in exog_names]
        if not idx:
            return None
        r_matrix = np.zeros((len(idx), len(exog_names)))
        for i, col_idx in enumerate(idx):
            r_matrix[i, col_idx] = 1
        try:
            return float(model_full.f_test(r_matrix).pvalue)
        except Exception:
            return None

    for var, cols in group_map.items():
        if not cols:
            continue
        p_val = _f_test_for_columns(cols)
        ftest_rows.append({
            "variable": var,
            "p_value": p_val,
            "incluida": "Si" if var in selected_vars else "No",
            "significativa": "Si" if p_val is not None and p_val < 0.05 else "No",
        })

    comb_vars = [v for v in selected_vars if v in group_map]
    if len(comb_vars) >= 2:
        for k in range(2, len(comb_vars) + 1):
            for combo in itertools.combinations(comb_vars, k):
                cols_combo = []
                for var in combo:
                    cols_combo.extend(group_map.get(var, []))
                p_val = _f_test_for_columns(cols_combo)
                ftest_rows.append({
                    "variable": " + ".join(combo),
                    "p_value": p_val,
                    "incluida": "Si",
                    "significativa": "Si" if p_val is not None and p_val < 0.05 else "No",
                })

    ftest_df = pd.DataFrame(ftest_rows)
    return resumen, ttest_df, ftest_df, "Analisis OLS listo."


# ===========================================================================
# 2.13 INTERPRETACION AUTOMATICA
# ===========================================================================

def generar_interpretacion_p1(resumen_df, mejores_df, base_df):
    if resumen_df is None or resumen_df.empty:
        return "Aun no hay corridas registradas para interpretar."

    artefactos = cargar_modelos_p1()
    modelos_disponibles = artefactos.get("disponible", False)
    partes = []

    best_reg = mejores_df[mejores_df["task"] == "regresion"] if not mejores_df.empty else pd.DataFrame()
    if not best_reg.empty:
        if modelos_disponibles:
            info = artefactos["regresion"]["metadata"]
            metricas = info.get("metrics", {})
            partes.append(
                f"Mejor regresion: {info.get('config', {}).get('name')} con "
                f"RMSE={metricas.get('rmse', 0.0):.2f}, R2={metricas.get('r2', 0.0):.3f}."
            )
            impacto = estimar_impacto_zona_p1(
                artefactos["regresion"]["model"],
                artefactos["regresion"]["preprocessor"],
                info.get("feature_columns", []),
                base_df,
            )
            if impacto is not None:
                signo = "favorable a zonas urbanas" if impacto > 0 else "favorable a zonas rurales"
                partes.append(
                    f"Impacto promedio de la zona: {abs(impacto):.1f} puntos {signo}."
                )
        else:
            row = best_reg.iloc[0]
            partes.append(f"Mejor regresion (MLflow): {row['model']} con {row['metric']}={row['metric_value']}.")

    best_bin = mejores_df[mejores_df["task"] == "clasificacion_binaria"] if not mejores_df.empty else pd.DataFrame()
    if not best_bin.empty:
        if modelos_disponibles:
            info = artefactos["clasificacion_binaria"]["metadata"]
            metricas = info.get("metrics", {})
            partes.append(
                f"Mejor clasificacion binaria: {info.get('config', {}).get('name')} con "
                f"F1={metricas.get('f1', 0.0):.3f} y accuracy={metricas.get('accuracy', 0.0):.3f}."
            )
        else:
            row = best_bin.iloc[0]
            partes.append(f"Mejor clasificacion binaria (MLflow): {row['model']} con {row['metric']}={row['metric_value']}.")

    if not partes:
        return "No se encontraron resultados suficientes para interpretar."

    partes.append(
        "Conclusion: la zona (urbano/rural), el estrato y el contexto socioeconomico "
        "ayudan a explicar el puntaje global y a priorizar municipios para intervencion."
    )
    if not modelos_disponibles:
        partes.append("Nota: el simulador A/B requiere que se haya entrenado al menos un mejor modelo localmente.")
    return " ".join(partes)


# ===========================================================================
# 2.14 RETROCOMPATIBILIDAD
# ===========================================================================
# Aliases para no romper imports antiguos durante el periodo de transicion.
# El script de entrenamiento usa estos nombres.

def build_modelo_regresion(input_dim, compile_model=True):
    """Backward-compat: arquitectura fija usada por el viejo script de entrenamiento."""
    config = _configuraciones_modelos_p1()["regresion"][0]
    model = _crear_mlp_p1(
        tf, input_dim, config,
        output_dim=1, output_activation="linear",
        loss=_loss_from_config(tf, config["loss"]), metrics=["mae"],
    )
    if not compile_model:
        # Reconstruir sin compilar.
        return _crear_mlp_p1_uncompiled(input_dim, config, output_dim=1, output_activation="linear")
    return model


def build_modelo_clasificacion(input_dim, compile_model=True):
    config = _configuraciones_modelos_p1()["clasificacion_binaria"][0]
    model = _crear_mlp_p1(
        tf, input_dim, config,
        output_dim=1, output_activation="sigmoid",
        loss=config["loss"], metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    if not compile_model:
        return _crear_mlp_p1_uncompiled(input_dim, config, output_dim=1, output_activation="sigmoid")
    return model


def _crear_mlp_p1_uncompiled(input_dim, config, output_dim, output_activation):
    """Igual que _crear_mlp_p1 pero sin compile (para load_weights puro)."""
    regularizer = None
    l2_val = float(config.get("l2", 0.0))
    if l2_val > 0:
        regularizer = tf.keras.regularizers.l2(l2_val)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in config.get("layers", []):
        model.add(tf.keras.layers.Dense(
            int(units),
            activation=config.get("activation", "relu"),
            kernel_regularizer=regularizer,
        ))
        if float(config.get("dropout", 0.0)) > 0:
            model.add(tf.keras.layers.Dropout(float(config["dropout"])))
    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))
    return model