import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tempfile
from urllib.parse import urlparse, unquote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def cargar_datos_p3():
    df = pd.read_csv('Data/saber11_Antioquia_clean.csv', dtype={'cole_cod_mcpio_ubicacion': str})
    df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].str.upper().str.strip()
    
    df['Acceso_TIC'] = df.apply(
        lambda x: 'Internet y Computador' if x['fami_tieneinternet'] == 'Si' and x['fami_tienecomputador'] == 'Si'
        else ('Solo Internet' if x['fami_tieneinternet'] == 'Si'
              else ('Solo Computador' if x['fami_tienecomputador'] == 'Si' else 'Sin Acceso TIC')),
        axis=1
    )
    
    df_coord = pd.read_csv('Data/municipios_unicos.csv')
    df = pd.merge(df, df_coord, on='cole_mcpio_ubicacion', how='left')
    
    return df

def obtener_lista_municipios(df):
    municipios = df['cole_mcpio_ubicacion'].dropna().unique().tolist()
    municipios.sort()
    return ['TODOS'] + municipios

def generar_mapa_antioquia(df, municipio):
    dff = df.copy()
    
    df_mapa = dff.groupby(['cole_mcpio_ubicacion', 'lat', 'lon'])['punt_ingles'].mean().reset_index()
    min_ingles = df_mapa['punt_ingles'].min()
    max_ingles = df_mapa['punt_ingles'].max()
    
    if municipio != 'TODOS':
        df_mapa['opacidad'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 1.0 if x == municipio else 0.1)
        df_mapa['tamano'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 15 if x == municipio else 5)
    else:
        df_mapa['opacidad'] = 0.8
        df_mapa['tamano'] = 8

    fig = px.scatter_mapbox(
        df_mapa,
        lat='lat',
        lon='lon',
        color='punt_ingles',
        hover_name='cole_mcpio_ubicacion',
        color_continuous_scale='Viridis',
        range_color=[min_ingles, max_ingles],
        mapbox_style='carto-positron',
        zoom=6.0,
        center={"lat": 6.2518, "lon": -75.5636},
        title=f'Promedio de Puntaje en Inglés por Municipio ({municipio})',
        size='tamano',
        size_max=15
    )
    fig.update_traces(marker=dict(opacity=df_mapa['opacidad']))
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def generar_ranking_municipios_estatico(df):
    dff = df.copy()
    df_rank = dff.groupby('cole_mcpio_ubicacion', as_index=False)['punt_ingles'].mean()
    df_rank = df_rank.sort_values('punt_ingles', ascending=True)

    fig = px.bar(
        df_rank,
        x='punt_ingles',
        y='cole_mcpio_ubicacion',
        orientation='h',
        labels={'punt_ingles': 'Promedio Puntaje Inglés', 'cole_mcpio_ubicacion': 'Municipio'},
        title='Ranking de Municipios por Promedio de Puntaje en Inglés (Antioquia)',
        color='punt_ingles',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=900)
    return fig

def generar_histograma_tic(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.histogram(
        dff,
        x='desemp_ingles',
        color='Acceso_TIC',
        barmode='group',
        category_orders={"desemp_ingles": ["A-", "A1", "A2", "B1", "B+"]},
        title=f'Distribución del Nivel de Inglés vs. Acceso TIC ({municipio})',
        labels={'desemp_ingles': 'Nivel de Inglés', 'count': 'Frecuencia'},
        color_discrete_map={
            'Internet y Computador': '#2ca02c',
            'Solo Internet': '#1f77b4',
            'Solo Computador': '#ff7f0e',
            'Sin Acceso TIC': '#d62728'
        }
    )
    return fig

def generar_dispersion_regresion(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.scatter(
        dff,
        x='punt_ingles',
        y='punt_global',
        color='Acceso_TIC',
        trendline='ols',
        opacity=0.5,
        title=f'Regresión: Puntaje de Inglés vs Puntaje Global ICFES ({municipio})',
        labels={'punt_ingles': 'Puntaje Inglés', 'punt_global': 'Puntaje Global'}
    )
    return fig

def generar_dispersion_clusters(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    # Dicotomización estricta para clústeres rojo/verde
    dff['Tiene_Internet'] = dff['fami_tieneinternet'].apply(
        lambda x: 'Con Internet' if x == 'Si' else 'Sin Internet'
    )

    fig = px.scatter(
        dff,
        x='punt_ingles',
        y='punt_global',
        color='Tiene_Internet',
        opacity=0.6,
        marginal_x='box', # Añade visualización de densidad en los ejes
        marginal_y='box',
        title=f'Clusters de Desempeño: Con vs Sin Internet ({municipio})',
        labels={'punt_ingles': 'Puntaje Inglés', 'punt_global': 'Puntaje Global'},
        color_discrete_map={
            'Con Internet': '#2ca02c', # Verde
            'Sin Internet': '#d62728'  # Rojo
        }
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def calcular_probabilidad_b1(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    con_internet = dff[dff['fami_tieneinternet'] == 'Si']
    sin_internet = dff[dff['fami_tieneinternet'] == 'No']

    if len(con_internet) == 0 or len(sin_internet) == 0:
        return 0.0

    # Cálculo de probabilidad marginal P(B1 U B+ | Internet)
    prob_con = len(con_internet[con_internet['desemp_ingles'].isin(['B1', 'B+'])]) / len(con_internet) * 100
    prob_sin = len(sin_internet[sin_internet['desemp_ingles'].isin(['B1', 'B+'])]) / len(sin_internet) * 100

    # Diferencial de probabilidad (Z%)
    diferencia_z = prob_con - prob_sin
    return round(diferencia_z, 2)


def generar_serie_tic_ingles_por_periodo(df, municipio='TODOS'):
    """Genera una serie temporal del promedio de `punt_ingles` por año,
    separada por categorías de `Acceso_TIC`.

    Se asume que `df` tiene la columna `periodo` y se usan los primeros 4
    caracteres como año (ej: '2019-1' -> 2019).
    """
    dff = df.copy()
    # Asegurar que exista la columna Acceso_TIC (si no, derivarla)
    if 'Acceso_TIC' not in dff.columns:
        dff['Acceso_TIC'] = dff.apply(
            lambda x: 'Internet y Computador' if x.get('fami_tieneinternet') == 'Si' and x.get('fami_tienecomputador') == 'Si'
            else ('Solo Internet' if x.get('fami_tieneinternet') == 'Si'
                  else ('Solo Computador' if x.get('fami_tienecomputador') == 'Si' else 'Sin Acceso TIC')),
            axis=1
        )

    # Extraer año de la columna periodo (primeros 4 dígitos)
    if 'periodo' in dff.columns:
        dff['year'] = dff['periodo'].astype(str).str[:4]
        # filtrar años válidos numéricos
        dff = dff[dff['year'].str.isnumeric()]
        dff['year'] = dff['year'].astype(int)
    else:
        dff['year'] = None

    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    # Agrupar por año y acceso TIC
    if dff['year'].notna().any():
        df_g = dff.groupby(['year', 'Acceso_TIC'], as_index=False)['punt_ingles'].mean()
        df_g = df_g.sort_values('year')
        fig = px.line(
            df_g,
            x='year',
            y='punt_ingles',
            color='Acceso_TIC',
            markers=True,
            title=f'Promedio Puntaje Inglés por Año y Acceso TIC ({municipio})',
            labels={'punt_ingles': 'Promedio Puntaje Inglés', 'year': 'Año'}
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    else:
        fig = px.line(title='No hay datos de periodo para construir la serie temporal')

    return fig


def _first_present_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _map_yes_no(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in {"SI", "S", "1", "TRUE", "T", "Y", "YES"}:
        return 1
    if text in {"NO", "N", "0", "FALSE", "F"}:
        return 0
    return np.nan


def _standardize_zone(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if "RURAL" in text:
        return "RURAL"
    if "URB" in text or "CABECERA" in text:
        return "URBANO"
    return text


def _standardize_school_type(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if "NO OFICIAL" in text or "PRIVAD" in text:
        return "PRIVADO"
    if "OFICIAL" in text or "PUBLIC" in text:
        return "PUBLICO"
    return text


def _build_modeling_frame(df):
    dff = df.copy()

    col_internet = _first_present_column(dff, ["fami_tieneinternet"])
    col_computador = _first_present_column(dff, ["fami_tienecomputador"])
    col_estrato = _first_present_column(dff, ["fami_estratovivienda"])
    col_edu_padre = _first_present_column(dff, ["fami_educacionpadre"])
    col_edu_madre = _first_present_column(dff, ["fami_educacionmadre"])
    col_area = _first_present_column(dff, ["cole_area_ubicacion", "estu_areareside"])
    col_naturaleza = _first_present_column(dff, ["cole_naturaleza"])
    col_jornada = _first_present_column(dff, ["cole_jornada"])
    col_genero = _first_present_column(dff, ["estu_genero", "cole_genero"])
    col_bilingue = _first_present_column(dff, ["cole_bilingue"])
    col_periodo = _first_present_column(dff, ["periodo"])
    col_municipio = _first_present_column(dff, ["cole_mcpio_ubicacion"])

    dff["internet_flag"] = dff[col_internet].map(_map_yes_no) if col_internet else np.nan
    dff["computador_flag"] = dff[col_computador].map(_map_yes_no) if col_computador else np.nan

    dff["tic_score"] = dff[["internet_flag", "computador_flag"]].sum(axis=1, min_count=1)
    dff["tic_interaccion"] = dff["internet_flag"] * dff["computador_flag"]

    if col_estrato:
        dff["estrato_cat"] = dff[col_estrato].astype("string").str.upper().str.strip()
    else:
        dff["estrato_cat"] = pd.NA

    if col_area:
        dff["zona"] = dff[col_area].map(_standardize_zone)
    else:
        dff["zona"] = pd.NA

    if col_naturaleza:
        dff["tipo_colegio"] = dff[col_naturaleza].map(_standardize_school_type)
    else:
        dff["tipo_colegio"] = pd.NA

    if col_jornada:
        dff["jornada"] = dff[col_jornada].astype("string").str.upper().str.strip()
    else:
        dff["jornada"] = pd.NA

    if col_genero:
        dff["genero"] = dff[col_genero].astype("string").str.upper().str.strip()
    else:
        dff["genero"] = pd.NA

    if col_edu_padre:
        dff["edu_padre"] = dff[col_edu_padre].astype("string").str.upper().str.strip()
    else:
        dff["edu_padre"] = pd.NA

    if col_edu_madre:
        dff["edu_madre"] = dff[col_edu_madre].astype("string").str.upper().str.strip()
    else:
        dff["edu_madre"] = pd.NA

    dff["bilingue_flag"] = dff[col_bilingue].map(_map_yes_no) if col_bilingue else np.nan

    if col_periodo:
        dff["periodo_year"] = pd.to_numeric(
            dff[col_periodo].astype("string").str.slice(0, 4),
            errors="coerce"
        )
    else:
        dff["periodo_year"] = np.nan

    if col_municipio:
        dff["municipio"] = dff[col_municipio].astype("string").str.upper().str.strip()
    else:
        dff["municipio"] = pd.NA

    return dff


def _build_feature_sets(df):
    base_numeric = ["internet_flag", "computador_flag", "tic_score", "tic_interaccion"]
    contexto_numeric = base_numeric + ["bilingue_flag"]
    contexto_categoric = [
        "estrato_cat",
        "zona",
        "tipo_colegio",
        "jornada",
        "genero",
        "edu_padre",
        "edu_madre",
    ]

    feature_sets = {
        "tic_basico": {
            "numeric": base_numeric,
            "categorical": [],
        },
        "tic_contexto": {
            "numeric": contexto_numeric,
            "categorical": contexto_categoric,
        },
    }

    # Filtrar solo columnas existentes
    for key, cfg in feature_sets.items():
        cfg["numeric"] = [c for c in cfg["numeric"] if c in df.columns]
        cfg["categorical"] = [c for c in cfg["categorical"] if c in df.columns]

    return feature_sets


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


def _flag_from_si_no(value):
    if pd.isna(value):
        return 0
    text = str(value).strip().upper()
    return 1 if text == "SI" else 0


def obtener_opciones_simulador_p3(df):
    dff = _build_modeling_frame(df)
    opciones = {
        "estrato": sorted([v for v in dff["estrato_cat"].dropna().unique().tolist()]),
        "zona": sorted([v for v in dff["zona"].dropna().unique().tolist()]),
        "tipo_colegio": sorted([v for v in dff["tipo_colegio"].dropna().unique().tolist()]),
        "jornada": sorted([v for v in dff["jornada"].dropna().unique().tolist()]),
        "genero": sorted([v for v in dff["genero"].dropna().unique().tolist()]),
        "edu_padre": sorted([v for v in dff["edu_padre"].dropna().unique().tolist()]),
        "edu_madre": sorted([v for v in dff["edu_madre"].dropna().unique().tolist()]),
    }

    opciones["estrato"] = opciones["estrato"] or ["ESTRATO 1", "ESTRATO 2"]
    opciones["zona"] = opciones["zona"] or ["URBANO", "RURAL"]
    opciones["tipo_colegio"] = opciones["tipo_colegio"] or ["PUBLICO", "PRIVADO"]
    opciones["jornada"] = opciones["jornada"] or ["COMPLETA", "MANANA"]
    opciones["genero"] = opciones["genero"] or ["MASCULINO", "FEMENINO", "MIXTO"]
    opciones["edu_padre"] = opciones["edu_padre"] or ["SECUNDARIA", "TECNICA"]
    opciones["edu_madre"] = opciones["edu_madre"] or ["SECUNDARIA", "TECNICA"]

    return opciones


def construir_input_p3(valores_form):
    internet_flag = _flag_from_si_no(valores_form.get("internet"))
    computador_flag = _flag_from_si_no(valores_form.get("computador"))
    bilingue_flag = _flag_from_si_no(valores_form.get("bilingue"))

    fila = {
        "internet_flag": internet_flag,
        "computador_flag": computador_flag,
        "tic_score": internet_flag + computador_flag,
        "tic_interaccion": internet_flag * computador_flag,
        "bilingue_flag": bilingue_flag,
        "estrato_cat": valores_form.get("estrato"),
        "zona": valores_form.get("zona"),
        "tipo_colegio": valores_form.get("tipo_colegio"),
        "jornada": valores_form.get("jornada"),
        "genero": valores_form.get("genero"),
        "edu_padre": valores_form.get("edu_padre"),
        "edu_madre": valores_form.get("edu_madre"),
    }
    return pd.DataFrame([fila])


P3_EXPERIMENT_NAME = "pregunta_3"


def _p3_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p3_models_dir():
    return os.path.join(_p3_base_dir(), "models", "pregunta_3")


def _resolve_mlflow_tracking_uri(base_dir):
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return (Path(base_dir) / "mlruns" / "pregunta_3").as_uri()


def obtener_mlflow_info():
    base_dir = _p3_base_dir()
    mlruns_path = Path(base_dir) / "mlruns" / "pregunta_3"
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    ui_url = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
    return {
        "tracking_uri": tracking_uri,
        "mlruns_path": str(mlruns_path),
        "ui_url": ui_url,
    }


def _parse_mlflow_ui_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000
    return host, port


def verificar_mlflow_ui():
    info = obtener_mlflow_info()
    host, port = _parse_mlflow_ui_url(info["ui_url"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True, f"MLflow UI activo en {info['ui_url']}"
        except Exception:
            return False, f"MLflow UI no responde en {info['ui_url']}"


def iniciar_mlflow_ui():
    info = obtener_mlflow_info()
    tracking_uri = info["tracking_uri"]
    host, port = _parse_mlflow_ui_url(info["ui_url"])

    is_up, message = verificar_mlflow_ui()
    if is_up:
        return True, message

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        tracking_uri,
        "--host",
        host,
        "--port",
        str(port),
    ]

    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "cwd": _p3_base_dir(),
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


def cargar_resultados_mlflow_p3(max_runs=200):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        return pd.DataFrame(), f"MLflow no disponible ({exc})."

    base_dir = _p3_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    exp = client.get_experiment_by_name(P3_EXPERIMENT_NAME)
    if exp is None:
        exp_ids = [e.experiment_id for e in client.search_experiments()]
        if not exp_ids:
            return pd.DataFrame(), "No se encontraron experimentos en MLflow."
    else:
        exp_ids = [exp.experiment_id]

    runs = client.search_runs(
        exp_ids,
        order_by=["attributes.start_time DESC"],
        max_results=max_runs,
    )

    rows = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags
        run_name = run.info.run_name or ""
        task = tags.get("task")

        if not task and run_name.startswith("p3_"):
            if run_name.startswith("p3_custom_reg") or run_name.startswith("p3_reg"):
                task = "regresion"
            elif run_name.startswith("p3_custom_bin") or run_name.startswith("p3_bin"):
                task = "clasificacion_binaria"
            elif run_name.startswith("p3_custom_multi") or run_name.startswith("p3_multi"):
                task = "clasificacion_multiclase"

        feature_set = tags.get("feature_set")
        if not feature_set and run_name.startswith("p3_"):
            parts = run_name.split("_")
            if len(parts) >= 3:
                feature_set = "custom" if parts[1] == "custom" else parts[2]

        if task is None and not run_name.startswith("p3_"):
            continue

        rows.append(
            {
                "task": task,
                "config_id": run_name,
                "model": tags.get("model_name") or params.get("name"),
                "feature_set": feature_set,
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
                "f1_macro": metrics.get("f1_macro"),
                "log_loss": metrics.get("log_loss"),
                "run_id": run.info.run_id,
            }
        )

    resumen = pd.DataFrame(rows)
    if resumen.empty:
        return resumen, "No hay corridas registradas en MLflow."
    return resumen, f"Corridas cargadas: {len(resumen)}"


def cargar_historial_mlflow_p3(resumen_df, max_runs=30):
    if resumen_df.empty:
        return pd.DataFrame()

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception:
        return pd.DataFrame()

    base_dir = _p3_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    rows = []
    for _, row in resumen_df.head(max_runs).iterrows():
        run_id = row.get("run_id")
        if not run_id:
            continue
        run = client.get_run(run_id)
        artifact_path = _artifact_uri_to_path(run.info.artifact_uri)
        history_dir = artifact_path / "history"
        if not history_dir.exists():
            continue

        csv_files = list(history_dir.glob("*.csv"))
        if not csv_files:
            continue

        try:
            hist = pd.read_csv(csv_files[0])
        except Exception:
            continue

        for idx, loss in enumerate(hist.get("loss", []), start=1):
            rows.append(
                {
                    "epoch": idx,
                    "loss": loss,
                    "val_loss": hist.get("val_loss", [None] * len(hist))[idx - 1],
                    "run_label": row.get("config_id") or run.info.run_name,
                }
            )

    return pd.DataFrame(rows)


def obtener_resumen_mlflow_p3(max_runs=200):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        return pd.DataFrame(), f"MLflow no disponible ({exc})."

    base_dir = _p3_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    exp = client.get_experiment_by_name(P3_EXPERIMENT_NAME)
    rows = []
    if exp is not None:
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_runs,
        )

        for run in runs:
            metrics = run.data.metrics
            tags = run.data.tags
            start_time = run.info.start_time
            if start_time:
                start_time = datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d %H:%M:%S")

            rows.append(
                {
                    "experiment": P3_EXPERIMENT_NAME,
                    "task": tags.get("task"),
                    "model": tags.get("model_name"),
                    "feature_set": tags.get("feature_set"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "r2": metrics.get("r2"),
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "f1_macro": metrics.get("f1_macro"),
                    "log_loss": metrics.get("log_loss"),
                    "status": run.info.status,
                    "start_time": start_time,
                    "run_id": run.info.run_id,
                }
            )

    resumen = pd.DataFrame(rows)
    if resumen.empty:
        return resumen, "No hay corridas registradas en MLflow."

    return resumen, f"Corridas encontradas: {len(resumen)}"


def obtener_mejores_modelos_mlflow(resumen_df):
    if resumen_df.empty:
        return pd.DataFrame()

    mejores = []
    for task, grupo in resumen_df.groupby("task"):
        if task == "regresion":
            metric = "rmse"
            grupo = grupo.dropna(subset=[metric])
            if grupo.empty:
                continue
            best = grupo.sort_values(metric).head(1)
            metric_value = float(best[metric].iloc[0])
        elif task == "clasificacion_multiclase":
            metric = "f1_macro"
            grupo = grupo.dropna(subset=[metric])
            if grupo.empty:
                continue
            best = grupo.sort_values(metric, ascending=False).head(1)
            metric_value = float(best[metric].iloc[0])
        else:
            metric = "f1"
            grupo = grupo.dropna(subset=[metric])
            if grupo.empty:
                continue
            best = grupo.sort_values(metric, ascending=False).head(1)
            metric_value = float(best[metric].iloc[0])

        mejores.append(
            {
                "task": task,
                "model": best["model"].iloc[0],
                "feature_set": best["feature_set"].iloc[0],
                "metric": metric,
                "metric_value": round(metric_value, 4),
                "run_id": best["run_id"].iloc[0],
            }
        )

    return pd.DataFrame(mejores)


def _ejecutar_experimentos_mlflow_p3_legacy(df, max_rows=60000, random_state=42):
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            f"No se pudo cargar MLflow o scikit-learn ({exc}). Instala dependencias y reintenta."
        )

    if df.empty:
        return pd.DataFrame(), "No hay datos disponibles para modelar."

    dff = _build_modeling_frame(df)

    # Limite de filas para ejecuciones rapidas
    if max_rows and len(dff) > max_rows:
        dff = dff.sample(n=max_rows, random_state=random_state)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_path = Path(base_dir) / "mlruns"
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlruns_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    feature_sets = _build_feature_sets(dff)
    results = []
    errores = []

    def _build_preprocess(num_features, cat_features):
        transformers = []
        if num_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                )
            )
        if cat_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_features,
                )
            )
        if not transformers:
            return None
        return ColumnTransformer(transformers=transformers)

    def _log_params(model, extra_params):
        params = {}
        if hasattr(model, "get_params"):
            params.update(model.get_params())
        params.update(extra_params)
        for key, val in params.items():
            if isinstance(val, (str, int, float, bool)):
                mlflow.log_param(key, val)

    def _append_result(row):
        results.append(row)

    # Regresion: puntaje de ingles
    if "punt_ingles" in dff.columns:
        target_reg = dff["punt_ingles"].dropna()
        if not target_reg.empty:
            reg_models = [
                ("LinearRegression", LinearRegression()),
                ("Ridge", Ridge(alpha=1.0)),
                ("Lasso", Lasso(alpha=0.001)),
                (
                    "RandomForestRegressor",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=None,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "GradientBoostingRegressor",
                    GradientBoostingRegressor(random_state=random_state),
                ),
            ]

            mlflow.set_experiment("P3_ingles_regresion")

            for feat_name, cfg in feature_sets.items():
                num_features = cfg["numeric"]
                cat_features = cfg["categorical"]
                if not num_features and not cat_features:
                    continue

                features = num_features + cat_features
                data = dff[features + ["punt_ingles"]].dropna(subset=["punt_ingles"])
                if data.empty:
                    continue

                X = data[features]
                y = data["punt_ingles"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=random_state,
                )

                preprocess = _build_preprocess(num_features, cat_features)

                for model_name, model in reg_models:
                    run_name = f"reg_{model_name}_{feat_name}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            if preprocess is None:
                                pipeline = model
                            else:
                                pipeline = Pipeline(
                                    steps=[("preprocess", preprocess), ("model", model)]
                                )

                            pipeline.fit(X_train, y_train)
                            preds = pipeline.predict(X_test)

                            try:
                                rmse = mean_squared_error(y_test, preds, squared=False)
                            except TypeError:
                                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                            mae = mean_absolute_error(y_test, preds)
                            r2 = r2_score(y_test, preds)

                            mlflow.set_tag("task", "regresion")
                            mlflow.set_tag("feature_set", feat_name)
                            mlflow.set_tag("model_name", model_name)
                            mlflow.log_metric("rmse", rmse)
                            mlflow.log_metric("mae", mae)
                            mlflow.log_metric("r2", r2)

                            _log_params(
                                model,
                                {
                                    "n_train": len(X_train),
                                    "n_test": len(X_test),
                                    "num_features": len(num_features),
                                    "cat_features": len(cat_features),
                                },
                            )

                            try:
                                mlflow.sklearn.log_model(pipeline, "model")
                            except Exception:
                                pass

                            run_id = mlflow.active_run().info.run_id
                            _append_result(
                                {
                                    "task": "regresion",
                                    "model": model_name,
                                    "feature_set": feat_name,
                                    "metric_name": "rmse",
                                    "metric_value": round(rmse, 4),
                                    "rmse": round(rmse, 4),
                                    "mae": round(mae, 4),
                                    "r2": round(r2, 4),
                                    "accuracy": None,
                                    "f1": None,
                                    "roc_auc": None,
                                    "f1_macro": None,
                                    "log_loss": None,
                                    "run_id": run_id,
                                }
                            )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

    # Clasificacion: niveles de ingles
    if "desemp_ingles" in dff.columns:
        niveles_validos = ["A-", "A1", "A2", "B1", "B+"]
        base = dff[dff["desemp_ingles"].isin(niveles_validos)].copy()

        if not base.empty:
            base["target_b1_plus"] = base["desemp_ingles"].isin(["B1", "B+"]).astype(int)

            clf_models = [
                (
                    "LogisticRegression",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
                (
                    "RandomForestClassifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "GradientBoostingClassifier",
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]

            # Binaria B1 o superior
            mlflow.set_experiment("P3_ingles_clasificacion_binaria")
            for feat_name, cfg in feature_sets.items():
                num_features = cfg["numeric"]
                cat_features = cfg["categorical"]
                if not num_features and not cat_features:
                    continue

                features = num_features + cat_features
                data = base[features + ["target_b1_plus"]].dropna(subset=["target_b1_plus"])
                if data.empty:
                    continue

                X = data[features]
                y = data["target_b1_plus"]
                if y.nunique() < 2:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=random_state,
                    stratify=y,
                )

                preprocess = _build_preprocess(num_features, cat_features)

                for model_name, model in clf_models:
                    run_name = f"clf_bin_{model_name}_{feat_name}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            if preprocess is None:
                                pipeline = model
                            else:
                                pipeline = Pipeline(
                                    steps=[("preprocess", preprocess), ("model", model)]
                                )

                            pipeline.fit(X_train, y_train)
                            preds = pipeline.predict(X_test)

                            probas = None
                            if hasattr(pipeline, "predict_proba"):
                                probas = pipeline.predict_proba(X_test)[:, 1]

                            accuracy = accuracy_score(y_test, preds)
                            f1 = f1_score(y_test, preds)
                            roc_auc = None
                            if probas is not None:
                                roc_auc = roc_auc_score(y_test, probas)

                            mlflow.set_tag("task", "clasificacion_binaria")
                            mlflow.set_tag("feature_set", feat_name)
                            mlflow.set_tag("model_name", model_name)
                            mlflow.log_metric("accuracy", accuracy)
                            mlflow.log_metric("f1", f1)
                            if roc_auc is not None:
                                mlflow.log_metric("roc_auc", roc_auc)

                            _log_params(
                                model,
                                {
                                    "n_train": len(X_train),
                                    "n_test": len(X_test),
                                    "num_features": len(num_features),
                                    "cat_features": len(cat_features),
                                },
                            )

                            try:
                                mlflow.sklearn.log_model(pipeline, "model")
                            except Exception:
                                pass

                            run_id = mlflow.active_run().info.run_id
                            _append_result(
                                {
                                    "task": "clasificacion_binaria",
                                    "model": model_name,
                                    "feature_set": feat_name,
                                    "metric_name": "f1",
                                    "metric_value": round(f1, 4),
                                    "rmse": None,
                                    "mae": None,
                                    "r2": None,
                                    "accuracy": round(accuracy, 4),
                                    "f1": round(f1, 4),
                                    "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                                    "f1_macro": None,
                                    "log_loss": None,
                                    "run_id": run_id,
                                }
                            )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

            # Multiclase niveles A-, A1, A2, B1, B+
            mlflow.set_experiment("P3_ingles_clasificacion_multiclase")
            for feat_name, cfg in feature_sets.items():
                num_features = cfg["numeric"]
                cat_features = cfg["categorical"]
                if not num_features and not cat_features:
                    continue

                features = num_features + cat_features
                data = base[features + ["desemp_ingles"]].dropna(subset=["desemp_ingles"])
                if data.empty:
                    continue

                X = data[features]
                y = data["desemp_ingles"]
                if y.nunique() < 2:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=random_state,
                    stratify=y,
                )

                preprocess = _build_preprocess(num_features, cat_features)

                for model_name, model in clf_models:
                    run_name = f"clf_multi_{model_name}_{feat_name}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            if preprocess is None:
                                pipeline = model
                            else:
                                pipeline = Pipeline(
                                    steps=[("preprocess", preprocess), ("model", model)]
                                )

                            pipeline.fit(X_train, y_train)
                            preds = pipeline.predict(X_test)

                            probas = None
                            if hasattr(pipeline, "predict_proba"):
                                probas = pipeline.predict_proba(X_test)

                            accuracy = accuracy_score(y_test, preds)
                            f1_macro = f1_score(y_test, preds, average="macro")
                            loss = None
                            if probas is not None:
                                loss = log_loss(y_test, probas)

                            mlflow.set_tag("task", "clasificacion_multiclase")
                            mlflow.set_tag("feature_set", feat_name)
                            mlflow.set_tag("model_name", model_name)
                            mlflow.log_metric("accuracy", accuracy)
                            mlflow.log_metric("f1_macro", f1_macro)
                            if loss is not None:
                                mlflow.log_metric("log_loss", loss)

                            _log_params(
                                model,
                                {
                                    "n_train": len(X_train),
                                    "n_test": len(X_test),
                                    "num_features": len(num_features),
                                    "cat_features": len(cat_features),
                                },
                            )

                            try:
                                mlflow.sklearn.log_model(pipeline, "model")
                            except Exception:
                                pass

                            run_id = mlflow.active_run().info.run_id
                            _append_result(
                                {
                                    "task": "clasificacion_multiclase",
                                    "model": model_name,
                                    "feature_set": feat_name,
                                    "metric_name": "f1_macro",
                                    "metric_value": round(f1_macro, 4),
                                    "rmse": None,
                                    "mae": None,
                                    "r2": None,
                                    "accuracy": round(accuracy, 4),
                                    "f1": None,
                                    "roc_auc": None,
                                    "f1_macro": round(f1_macro, 4),
                                    "log_loss": round(loss, 4) if loss is not None else None,
                                    "run_id": run_id,
                                }
                            )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

    if not results:
        return pd.DataFrame(), "No se generaron corridas. Revisa las variables disponibles."

    resumen = pd.DataFrame(results)

    # Seleccion rapida de mejores por tarea
    resumen_texto = []
    for task, group in resumen.groupby("task"):
        if task == "regresion":
            best = group.sort_values("rmse").head(1)
            metric_label = "RMSE"
            metric_value = best["rmse"].iloc[0]
        elif task == "clasificacion_multiclase":
            best = group.sort_values("f1_macro", ascending=False).head(1)
            metric_label = "F1 macro"
            metric_value = best["f1_macro"].iloc[0]
        else:
            best = group.sort_values("f1", ascending=False).head(1)
            metric_label = "F1"
            metric_value = best["f1"].iloc[0]

        resumen_texto.append(
            f"{task}: mejor {best['model'].iloc[0]} ({best['feature_set'].iloc[0]}) - {metric_label}={metric_value}"
        )

    texto = " | ".join(resumen_texto)
    if errores:
        texto = f"{texto} | Corridas con error: {len(errores)}"
    return resumen, texto


def _configuraciones_modelos_p3():
    return {
        "regresion": [
            {
                "name": "mlp_reg_mse",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "mse",
                "epochs": 40,
                "batch_size": 64,
            },
            {
                "name": "mlp_reg_huber",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-3,
                "optimizer": "rmsprop",
                "learning_rate": 0.0005,
                "loss": "huber",
                "epochs": 60,
                "batch_size": 128,
            },
            {
                "name": "mlp_reg_mae",
                "layers": [64],
                "activation": "tanh",
                "dropout": 0.0,
                "l2": 0.0,
                "optimizer": "adam",
                "learning_rate": 0.002,
                "loss": "mae",
                "epochs": 30,
                "batch_size": 64,
            },
        ],
        "clasificacion_binaria": [
            {
                "name": "mlp_bin_1",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "binary_crossentropy",
                "epochs": 40,
                "batch_size": 64,
            },
            {
                "name": "mlp_bin_2",
                "layers": [128, 64, 32],
                "activation": "elu",
                "dropout": 0.3,
                "l2": 1e-4,
                "optimizer": "rmsprop",
                "learning_rate": 0.0005,
                "loss": "binary_crossentropy",
                "epochs": 60,
                "batch_size": 128,
            },
            {
                "name": "mlp_bin_3",
                "layers": [32, 16],
                "activation": "relu",
                "dropout": 0.1,
                "l2": 1e-3,
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "loss": "binary_crossentropy",
                "epochs": 50,
                "batch_size": 32,
            },
        ],
        "clasificacion_multiclase": [
            {
                "name": "mlp_multi_1",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "categorical_crossentropy",
                "epochs": 40,
                "batch_size": 64,
            },
            {
                "name": "mlp_multi_2",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.3,
                "l2": 1e-4,
                "optimizer": "rmsprop",
                "learning_rate": 0.0005,
                "loss": "categorical_crossentropy",
                "epochs": 60,
                "batch_size": 128,
            },
            {
                "name": "mlp_multi_3",
                "layers": [64, 32],
                "activation": "tanh",
                "dropout": 0.1,
                "l2": 1e-3,
                "optimizer": "sgd",
                "learning_rate": 0.01,
                "loss": "categorical_crossentropy",
                "epochs": 50,
                "batch_size": 64,
            },
        ],
    }


def _configuraciones_modelos_p3_ampliadas():
    base = _configuraciones_modelos_p3()
    base["regresion"].extend(
        [
            {
                "name": "mlp_reg_mse_deep",
                "layers": [128, 64, 32, 16],
                "activation": "relu",
                "dropout": 0.25,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.0008,
                "loss": "mse",
                "epochs": 80,
                "batch_size": 128,
            },
            {
                "name": "mlp_reg_huber_sgd",
                "layers": [64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "l2": 1e-3,
                "optimizer": "sgd",
                "learning_rate": 0.005,
                "loss": "huber",
                "epochs": 60,
                "batch_size": 64,
            },
        ]
    )

    base["clasificacion_binaria"].extend(
        [
            {
                "name": "mlp_bin_4",
                "layers": [128, 64],
                "activation": "relu",
                "dropout": 0.2,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.0008,
                "loss": "binary_crossentropy",
                "epochs": 70,
                "batch_size": 128,
            },
            {
                "name": "mlp_bin_5",
                "layers": [64, 32, 16],
                "activation": "tanh",
                "dropout": 0.15,
                "l2": 1e-3,
                "optimizer": "rmsprop",
                "learning_rate": 0.0007,
                "loss": "binary_crossentropy",
                "epochs": 60,
                "batch_size": 64,
            },
        ]
    )

    base["clasificacion_multiclase"].extend(
        [
            {
                "name": "mlp_multi_4",
                "layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.25,
                "l2": 1e-4,
                "optimizer": "adam",
                "learning_rate": 0.0008,
                "loss": "categorical_crossentropy",
                "epochs": 70,
                "batch_size": 128,
            },
            {
                "name": "mlp_multi_5",
                "layers": [64, 32, 16],
                "activation": "tanh",
                "dropout": 0.2,
                "l2": 1e-3,
                "optimizer": "rmsprop",
                "learning_rate": 0.0007,
                "loss": "categorical_crossentropy",
                "epochs": 60,
                "batch_size": 64,
            },
        ]
    )

    return base


def _serializable_params(config):
    params = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            params[key] = "-".join([str(v) for v in value])
        else:
            params[key] = value
    return params


def _to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def _guardar_mejor_modelo_p3(task, model, preprocessor, config, metrics, feature_columns, label_mapping=None, base_dir=None):
    if base_dir is None:
        base_dir = _p3_models_dir()
    task_dir = os.path.join(base_dir, task, "best")
    os.makedirs(task_dir, exist_ok=True)

    model_path = os.path.join(task_dir, "model.keras")
    model.save(model_path)

    try:
        import joblib
        joblib.dump(preprocessor, os.path.join(task_dir, "preprocessor.pkl"))
    except Exception:
        pass

    metadata = {
        "task": task,
        "config": _serializable_params(config),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "label_mapping": label_mapping,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(task_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


def cargar_modelos_p3():
    try:
        import tensorflow as tf
        import joblib
    except Exception as exc:
        return {"disponible": False, "error": f"Dependencias no disponibles ({exc})."}

    base_dir = _p3_models_dir()
    custom_dir = os.path.join(base_dir, "custom")
    tasks = ["regresion", "clasificacion_binaria", "clasificacion_multiclase"]

    artefactos = {"disponible": True, "error": None}

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
            if os.path.exists(model_path) and os.path.exists(preproc_path) and os.path.exists(metadata_path):
                seleccionado = (source, model_path, preproc_path, metadata_path)
                break

        if seleccionado is None:
            artefactos["disponible"] = False
            artefactos["error"] = (
                "Modelos de Pregunta 3 no encontrados. Entrena modelos desde el panel de variables."
            )
            return artefactos

        source, model_path, preproc_path, metadata_path = seleccionado
        artefactos[task] = {
            "model": tf.keras.models.load_model(model_path),
            "preprocessor": joblib.load(preproc_path),
            "metadata": json.load(open(metadata_path, "r", encoding="utf-8")),
            "source": source,
        }

    return artefactos


def _evaluar_regresion(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def _evaluar_clasificacion_binaria(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def _evaluar_clasificacion_multi(y_true, y_pred, y_proba, labels):
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    except Exception:
        metrics["log_loss"] = None
    return metrics


def _crear_optimizer(tf, config):
    lr = float(config.get("learning_rate", 0.001))
    opt = str(config.get("optimizer", "adam")).lower()
    if opt == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if opt == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _crear_mlp(tf, input_dim, config, output_dim, output_activation, loss, metrics=None):
    regularizer = None
    l2_val = float(config.get("l2", 0.0))
    if l2_val > 0:
        regularizer = tf.keras.regularizers.l2(l2_val)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in config.get("layers", []):
        model.add(tf.keras.layers.Dense(int(units), activation=config.get("activation", "relu"),
                                         kernel_regularizer=regularizer))
        if float(config.get("dropout", 0.0)) > 0:
            model.add(tf.keras.layers.Dropout(float(config["dropout"])))
    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))

    optimizer = _crear_optimizer(tf, config)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])
    return model


def _loss_from_config(tf, loss_name):
    if loss_name == "huber":
        return tf.keras.losses.Huber()
    return loss_name


def seleccionar_mejores_modelos_resultados(resumen_df):
    if resumen_df.empty:
        return pd.DataFrame()

    mejores = []
    for task, group in resumen_df.groupby("task"):
        if task == "regresion":
            group = group.dropna(subset=["rmse"])
            if group.empty:
                continue
            best = group.sort_values(["rmse", "r2"], ascending=[True, False]).head(1)
            metric = "rmse"
            metric_value = float(best["rmse"].iloc[0])
        elif task == "clasificacion_multiclase":
            group = group.dropna(subset=["f1_macro"])
            if group.empty:
                continue
            best = group.sort_values("f1_macro", ascending=False).head(1)
            metric = "f1_macro"
            metric_value = float(best["f1_macro"].iloc[0])
        else:
            group = group.dropna(subset=["f1"])
            if group.empty:
                continue
            best = group.sort_values("f1", ascending=False).head(1)
            metric = "f1"
            metric_value = float(best["f1"].iloc[0])

        mejores.append(
            {
                "task": task,
                "model": best["model"].iloc[0],
                "feature_set": best["feature_set"].iloc[0],
                "metric": metric,
                "metric_value": round(metric_value, 4),
                "run_id": best["run_id"].iloc[0],
            }
        )

    return pd.DataFrame(mejores)


def construir_figuras_comparativas(resumen_df, history_df):
    fig_reg = go.Figure()
    fig_bin = go.Figure()
    fig_multi = go.Figure()
    fig_loss = go.Figure()

    if resumen_df.empty:
        fig_reg.update_layout(title="Sin corridas de regresion")
        fig_bin.update_layout(title="Sin corridas de clasificacion binaria")
        fig_multi.update_layout(title="Sin corridas de clasificacion multiclase")
        fig_loss.update_layout(title="Sin curvas de entrenamiento")
        return fig_reg, fig_bin, fig_multi, fig_loss

    reg = resumen_df[resumen_df["task"] == "regresion"].copy()
    if not reg.empty:
        fig_reg = px.bar(
            reg,
            x="config_id",
            y="rmse",
            color="feature_set",
            title="RMSE por configuracion (regresion)",
            labels={"config_id": "Configuracion", "rmse": "RMSE"},
        )

    bin_df = resumen_df[resumen_df["task"] == "clasificacion_binaria"].copy()
    if not bin_df.empty:
        fig_bin = px.bar(
            bin_df,
            x="config_id",
            y="f1",
            color="feature_set",
            title="F1-score por configuracion (clasificacion binaria)",
            labels={"config_id": "Configuracion", "f1": "F1-score"},
        )

    multi_df = resumen_df[resumen_df["task"] == "clasificacion_multiclase"].copy()
    if not multi_df.empty:
        fig_multi = px.bar(
            multi_df,
            x="config_id",
            y="f1_macro",
            color="feature_set",
            title="F1 macro por configuracion (clasificacion multiclase)",
            labels={"config_id": "Configuracion", "f1_macro": "F1 macro"},
        )

    if not history_df.empty:
        fig_loss = px.line(
            history_df,
            x="epoch",
            y="loss",
            color="run_label",
            title="Perdida por epoca (todas las corridas)",
            labels={"epoch": "Epoca", "loss": "Loss"},
        )
        if "val_loss" in history_df.columns:
            fig_loss.add_traces(
                px.line(
                    history_df,
                    x="epoch",
                    y="val_loss",
                    color="run_label",
                    line_dash_sequence=["dash"],
                ).data
            )

    for fig in [fig_reg, fig_bin, fig_multi, fig_loss]:
        fig.update_layout(margin={"r": 10, "t": 40, "l": 10, "b": 10})

    return fig_reg, fig_bin, fig_multi, fig_loss


def estimar_impacto_tic(model, preprocessor, feature_columns, base_df, task="regresion", n_muestras=800):
    if base_df.empty:
        return None

    if task == "clasificacion_multiclase":
        return None

    missing = [c for c in feature_columns if c not in base_df.columns]
    if missing:
        return None

    muestra = base_df[feature_columns].dropna()
    if muestra.empty:
        return None

    muestra = muestra.sample(n=min(n_muestras, len(muestra)), random_state=42)
    base = muestra.copy()
    alt = muestra.copy()
    if "internet_flag" in alt.columns:
        alt["internet_flag"] = 1
    if "computador_flag" in alt.columns:
        alt["computador_flag"] = 1

    if "tic_score" in alt.columns:
        internet_val = alt["internet_flag"] if "internet_flag" in alt.columns else 0
        computador_val = alt["computador_flag"] if "computador_flag" in alt.columns else 0
        alt["tic_score"] = internet_val + computador_val
    if "tic_interaccion" in alt.columns:
        internet_val = alt["internet_flag"] if "internet_flag" in alt.columns else 0
        computador_val = alt["computador_flag"] if "computador_flag" in alt.columns else 0
        alt["tic_interaccion"] = internet_val * computador_val

    try:
        X_base = _to_dense(preprocessor.transform(base[feature_columns]))
        X_alt = _to_dense(preprocessor.transform(alt[feature_columns]))
        pred_base = model.predict(X_base, verbose=0).reshape(-1)
        pred_alt = model.predict(X_alt, verbose=0).reshape(-1)
        impacto = np.nanmean(pred_alt - pred_base)
        if np.isnan(impacto):
            return None
        return float(impacto)
    except Exception:
        return None


def generar_interpretacion_p3(resumen_df, mejores_df, base_df):
    if resumen_df.empty:
        return "No hay corridas registradas para interpretar."

    texto_partes = []
    artefactos = cargar_modelos_p3()
    modelos_disponibles = artefactos.get("disponible", False)

    def _metricas_mlflow(task_name, metric_key):
        best_row = mejores_df[mejores_df["task"] == task_name]
        if best_row.empty:
            return None
        run_id = best_row["run_id"].iloc[0]
        run_row = resumen_df[resumen_df["run_id"] == run_id]
        if run_row.empty:
            return None
        return run_row.iloc[0].get(metric_key)

    best_reg = mejores_df[mejores_df["task"] == "regresion"]
    if not best_reg.empty:
        if modelos_disponibles:
            info = artefactos["regresion"]["metadata"]
            metricas = info.get("metrics", {})
            texto_partes.append(
                "Mejor regresion: {model} con RMSE={rmse:.2f} y R2={r2:.2f}.".format(
                    model=info.get("config", {}).get("name"),
                    rmse=metricas.get("rmse", 0.0),
                    r2=metricas.get("r2", 0.0),
                )
            )
            impacto = estimar_impacto_tic(
                artefactos["regresion"]["model"],
                artefactos["regresion"]["preprocessor"],
                info.get("feature_columns", []),
                base_df,
                task="regresion",
            )
            if impacto is not None:
                texto_partes.append(
                    f"Impacto TIC estimado en puntaje: {impacto:.2f} puntos."
                )
        else:
            rmse = _metricas_mlflow("regresion", "rmse")
            r2 = _metricas_mlflow("regresion", "r2")
            if rmse is not None and r2 is not None:
                texto_partes.append(
                    f"Mejor regresion (MLflow): RMSE={rmse:.2f}, R2={r2:.2f}."
                )

    best_bin = mejores_df[mejores_df["task"] == "clasificacion_binaria"]
    if not best_bin.empty:
        if modelos_disponibles:
            info = artefactos["clasificacion_binaria"]["metadata"]
            metricas = info.get("metrics", {})
            texto_partes.append(
                "Mejor clasificacion binaria: {model} con F1={f1:.2f} y accuracy={acc:.2f}.".format(
                    model=info.get("config", {}).get("name"),
                    f1=metricas.get("f1", 0.0),
                    acc=metricas.get("accuracy", 0.0),
                )
            )
            impacto = estimar_impacto_tic(
                artefactos["clasificacion_binaria"]["model"],
                artefactos["clasificacion_binaria"]["preprocessor"],
                info.get("feature_columns", []),
                base_df,
                task="clasificacion_binaria",
            )
            if impacto is not None:
                texto_partes.append(
                    f"Impacto TIC estimado en probabilidad B1+: {impacto * 100:.2f} pts porcentuales."
                )
        else:
            f1 = _metricas_mlflow("clasificacion_binaria", "f1")
            acc = _metricas_mlflow("clasificacion_binaria", "accuracy")
            if f1 is not None and acc is not None:
                texto_partes.append(
                    f"Mejor clasificacion binaria (MLflow): F1={f1:.2f}, accuracy={acc:.2f}."
                )

    best_multi = mejores_df[mejores_df["task"] == "clasificacion_multiclase"]
    if not best_multi.empty:
        if modelos_disponibles:
            info = artefactos["clasificacion_multiclase"]["metadata"]
            metricas = info.get("metrics", {})
            texto_partes.append(
                "Mejor clasificacion multiclase: {model} con F1 macro={f1:.2f} y accuracy={acc:.2f}.".format(
                    model=info.get("config", {}).get("name"),
                    f1=metricas.get("f1_macro", 0.0),
                    acc=metricas.get("accuracy", 0.0),
                )
            )
        else:
            f1m = _metricas_mlflow("clasificacion_multiclase", "f1_macro")
            acc = _metricas_mlflow("clasificacion_multiclase", "accuracy")
            if f1m is not None and acc is not None:
                texto_partes.append(
                    f"Mejor clasificacion multiclase (MLflow): F1 macro={f1m:.2f}, accuracy={acc:.2f}."
                )

    if not texto_partes:
        return "No se encontraron resultados suficientes para interpretar."

    texto_partes.append(
        "Conclusión: las variables de TIC (internet/computador) y contexto escolar ayudan a explicar el desempeño en ingles."
    )

    if not modelos_disponibles:
        texto_partes.append("Impactos TIC no disponibles sin modelos guardados localmente.")

    return " ".join(texto_partes)


def ejecutar_experimentos_mlflow_p3(df, max_rows=20000, random_state=42):
    try:
        import mlflow
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        import tensorflow as tf
        import joblib
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            f"No se pudo iniciar el laboratorio ({exc}).",
        )

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", "No hay datos para modelar."

    dff = _build_modeling_frame(df)
    if max_rows and len(dff) > max_rows:
        dff = dff.sample(n=max_rows, random_state=random_state)

    base_dir = _p3_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs(Path(base_dir) / "mlruns" / "pregunta_3", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(P3_EXPERIMENT_NAME)

    feature_sets = _build_feature_sets(dff)
    configs = _configuraciones_modelos_p3()
    results = []
    history_rows = []
    errores = []

    best_tracker = {
        "regresion": {"metric": np.inf},
        "clasificacion_binaria": {"metric": -np.inf},
        "clasificacion_multiclase": {"metric": -np.inf},
    }

    def _build_preprocess(num_features, cat_features):
        transformers = []
        if num_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                )
            )
        if cat_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_features,
                )
            )
        return ColumnTransformer(transformers=transformers) if transformers else None

    for feature_set_name, cfg in feature_sets.items():
        num_features = cfg["numeric"]
        cat_features = cfg["categorical"]
        if not num_features and not cat_features:
            continue

        features = num_features + cat_features

        # --------------------- REGRESION ---------------------
        if "punt_ingles" in dff.columns:
            data_reg = dff[features + ["punt_ingles"]].dropna(subset=["punt_ingles"])
            if not data_reg.empty:
                X = data_reg[features]
                y = data_reg["punt_ingles"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state
                )

                X_train = _coerce_numeric_columns(X_train, num_features)
                X_test = _coerce_numeric_columns(X_test, num_features)
                X_train = _sanitize_missing_columns(X_train, cat_features)
                X_test = _sanitize_missing_columns(X_test, cat_features)

                preprocessor = _build_preprocess(num_features, cat_features)
                X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
                X_test_proc = _to_dense(preprocessor.transform(X_test))

                for idx, config in enumerate(configs["regresion"], start=1):
                    run_name = f"p3_reg_{feature_set_name}_{idx}"
                    config_id = f"reg_{feature_set_name}_{idx}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            loss_fn = _loss_from_config(tf, config["loss"])
                            model = _crear_mlp(
                                tf,
                                X_train_proc.shape[1],
                                config,
                                output_dim=1,
                                output_activation="linear",
                                loss=loss_fn,
                            )

                            history = model.fit(
                                X_train_proc,
                                y_train,
                                validation_split=0.2,
                                epochs=config["epochs"],
                                batch_size=config["batch_size"],
                                verbose=0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        patience=5,
                                        restore_best_weights=True,
                                    )
                                ],
                            )

                            preds = model.predict(X_test_proc, verbose=0).reshape(-1)
                            metricas = _evaluar_regresion(y_test, preds)

                            mlflow.set_tag("task", "regresion")
                            mlflow.set_tag("feature_set", feature_set_name)
                            mlflow.set_tag("model_name", config["name"])
                            mlflow.log_params(_serializable_params(config))
                            mlflow.log_metrics(metricas)

                            run_id = mlflow.active_run().info.run_id

                            # Guardar historia y config como artefactos
                            with tempfile.TemporaryDirectory() as tmpdir:
                                hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                                pd.DataFrame(history.history).to_csv(hist_path, index=False)
                                cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                                with open(cfg_path, "w", encoding="utf-8") as f:
                                    json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                                mlflow.log_artifact(hist_path, artifact_path="history")
                                mlflow.log_artifact(cfg_path, artifact_path="config")

                                model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                                model.save(model_path)
                                mlflow.log_artifact(model_path, artifact_path="model")

                                preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                                joblib.dump(preprocessor, preproc_path)
                                mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                            for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                                history_rows.append(
                                    {
                                        "task": "regresion",
                                        "feature_set": feature_set_name,
                                        "config_id": config_id,
                                        "run_id": run_id,
                                        "run_label": f"reg-{config_id}",
                                        "epoch": epoch_idx,
                                        "loss": loss,
                                        "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                    }
                                )

                            results.append(
                                {
                                    "task": "regresion",
                                    "model": config["name"],
                                    "feature_set": feature_set_name,
                                    "config_id": config_id,
                                    "layers": "-".join([str(x) for x in config["layers"]]),
                                    "dropout": config["dropout"],
                                    "activation": config["activation"],
                                    "optimizer": config["optimizer"],
                                    "learning_rate": config["learning_rate"],
                                    "epochs": config["epochs"],
                                    "batch_size": config["batch_size"],
                                    "loss": config["loss"],
                                    "rmse": metricas["rmse"],
                                    "mae": metricas["mae"],
                                    "r2": metricas["r2"],
                                    "accuracy": None,
                                    "f1": None,
                                    "roc_auc": None,
                                    "f1_macro": None,
                                    "log_loss": None,
                                    "run_id": run_id,
                                }
                            )

                            if metricas["rmse"] < best_tracker["regresion"]["metric"]:
                                best_tracker["regresion"] = {
                                    "metric": metricas["rmse"],
                                    "metrics": metricas,
                                }
                                _guardar_mejor_modelo_p3(
                                    "regresion",
                                    model,
                                    preprocessor,
                                    config,
                                    metricas,
                                    features,
                                )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

        # --------------------- CLASIFICACION BINARIA ---------------------
        if "desemp_ingles" in dff.columns:
            niveles_validos = ["A-", "A1", "A2", "B1", "B+"]
            data_clf = dff[dff["desemp_ingles"].isin(niveles_validos)].copy()
            if not data_clf.empty:
                data_clf["target_bin"] = data_clf["desemp_ingles"].isin(["B1", "B+"]).astype(int)
                data_bin = data_clf[features + ["target_bin"]].dropna(subset=["target_bin"])
                if not data_bin.empty and data_bin["target_bin"].nunique() > 1:
                    X = data_bin[features]
                    y = data_bin["target_bin"]
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=random_state, stratify=y
                    )

                    X_train = _coerce_numeric_columns(X_train, num_features)
                    X_test = _coerce_numeric_columns(X_test, num_features)
                    X_train = _sanitize_missing_columns(X_train, cat_features)
                    X_test = _sanitize_missing_columns(X_test, cat_features)

                    preprocessor = _build_preprocess(num_features, cat_features)
                    X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
                    X_test_proc = _to_dense(preprocessor.transform(X_test))

                    for idx, config in enumerate(configs["clasificacion_binaria"], start=1):
                        run_name = f"p3_bin_{feature_set_name}_{idx}"
                        config_id = f"bin_{feature_set_name}_{idx}"
                        try:
                            with mlflow.start_run(run_name=run_name):
                                model = _crear_mlp(
                                    tf,
                                    X_train_proc.shape[1],
                                    config,
                                    output_dim=1,
                                    output_activation="sigmoid",
                                    loss=config["loss"],
                                    metrics=["accuracy"],
                                )

                                history = model.fit(
                                    X_train_proc,
                                    y_train,
                                    validation_split=0.2,
                                    epochs=config["epochs"],
                                    batch_size=config["batch_size"],
                                    verbose=0,
                                    callbacks=[
                                        tf.keras.callbacks.EarlyStopping(
                                            patience=5,
                                            restore_best_weights=True,
                                        )
                                    ],
                                )

                                proba = model.predict(X_test_proc, verbose=0).reshape(-1)
                                pred = (proba >= 0.5).astype(int)
                                metricas = _evaluar_clasificacion_binaria(y_test, pred, proba)

                                mlflow.set_tag("task", "clasificacion_binaria")
                                mlflow.set_tag("feature_set", feature_set_name)
                                mlflow.set_tag("model_name", config["name"])
                                mlflow.log_params(_serializable_params(config))
                                mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None})

                                run_id = mlflow.active_run().info.run_id

                                with tempfile.TemporaryDirectory() as tmpdir:
                                    hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                                    pd.DataFrame(history.history).to_csv(hist_path, index=False)
                                    cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                                    with open(cfg_path, "w", encoding="utf-8") as f:
                                        json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                                    mlflow.log_artifact(hist_path, artifact_path="history")
                                    mlflow.log_artifact(cfg_path, artifact_path="config")

                                    model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                                    model.save(model_path)
                                    mlflow.log_artifact(model_path, artifact_path="model")

                                    preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                                    joblib.dump(preprocessor, preproc_path)
                                    mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                                for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                                    history_rows.append(
                                        {
                                            "task": "clasificacion_binaria",
                                            "feature_set": feature_set_name,
                                            "config_id": config_id,
                                            "run_id": run_id,
                                            "run_label": f"bin-{config_id}",
                                            "epoch": epoch_idx,
                                            "loss": loss,
                                            "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                        }
                                    )

                                results.append(
                                    {
                                        "task": "clasificacion_binaria",
                                        "model": config["name"],
                                        "feature_set": feature_set_name,
                                        "config_id": config_id,
                                        "layers": "-".join([str(x) for x in config["layers"]]),
                                        "dropout": config["dropout"],
                                        "activation": config["activation"],
                                        "optimizer": config["optimizer"],
                                        "learning_rate": config["learning_rate"],
                                        "epochs": config["epochs"],
                                        "batch_size": config["batch_size"],
                                        "loss": config["loss"],
                                        "rmse": None,
                                        "mae": None,
                                        "r2": None,
                                        "accuracy": metricas["accuracy"],
                                        "f1": metricas["f1"],
                                        "roc_auc": metricas.get("roc_auc"),
                                        "f1_macro": None,
                                        "log_loss": None,
                                        "run_id": run_id,
                                    }
                                )

                                if metricas["f1"] > best_tracker["clasificacion_binaria"]["metric"]:
                                    best_tracker["clasificacion_binaria"] = {
                                        "metric": metricas["f1"],
                                        "metrics": metricas,
                                    }
                                    _guardar_mejor_modelo_p3(
                                        "clasificacion_binaria",
                                        model,
                                        preprocessor,
                                        config,
                                        metricas,
                                        features,
                                    )
                        except Exception as exc:
                            errores.append(f"{run_name}: {exc}")

                # --------------------- CLASIFICACION MULTICLASE ---------------------
                data_multi = data_clf[features + ["desemp_ingles"]].dropna(subset=["desemp_ingles"])
                if not data_multi.empty and data_multi["desemp_ingles"].nunique() > 1:
                    X = data_multi[features]
                    y_raw = data_multi["desemp_ingles"].astype(str)
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=random_state, stratify=y
                    )

                    X_train = _coerce_numeric_columns(X_train, num_features)
                    X_test = _coerce_numeric_columns(X_test, num_features)
                    X_train = _sanitize_missing_columns(X_train, cat_features)
                    X_test = _sanitize_missing_columns(X_test, cat_features)

                    preprocessor = _build_preprocess(num_features, cat_features)
                    X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
                    X_test_proc = _to_dense(preprocessor.transform(X_test))

                    y_train_cat = tf.keras.utils.to_categorical(y_train)
                    y_test_cat = tf.keras.utils.to_categorical(y_test)

                    for idx, config in enumerate(configs["clasificacion_multiclase"], start=1):
                        run_name = f"p3_multi_{feature_set_name}_{idx}"
                        config_id = f"multi_{feature_set_name}_{idx}"
                        try:
                            with mlflow.start_run(run_name=run_name):
                                model = _crear_mlp(
                                    tf,
                                    X_train_proc.shape[1],
                                    config,
                                    output_dim=len(le.classes_),
                                    output_activation="softmax",
                                    loss=config["loss"],
                                    metrics=["accuracy"],
                                )

                                history = model.fit(
                                    X_train_proc,
                                    y_train_cat,
                                    validation_split=0.2,
                                    epochs=config["epochs"],
                                    batch_size=config["batch_size"],
                                    verbose=0,
                                    callbacks=[
                                        tf.keras.callbacks.EarlyStopping(
                                            patience=5,
                                            restore_best_weights=True,
                                        )
                                    ],
                                )

                                proba = model.predict(X_test_proc, verbose=0)
                                pred = np.argmax(proba, axis=1)
                                metricas = _evaluar_clasificacion_multi(y_test, pred, proba, labels=list(range(len(le.classes_))))

                                mlflow.set_tag("task", "clasificacion_multiclase")
                                mlflow.set_tag("feature_set", feature_set_name)
                                mlflow.set_tag("model_name", config["name"])
                                mlflow.log_params(_serializable_params(config))
                                mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None})

                                run_id = mlflow.active_run().info.run_id

                                with tempfile.TemporaryDirectory() as tmpdir:
                                    hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                                    pd.DataFrame(history.history).to_csv(hist_path, index=False)
                                    cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                                    with open(cfg_path, "w", encoding="utf-8") as f:
                                        json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                                    mlflow.log_artifact(hist_path, artifact_path="history")
                                    mlflow.log_artifact(cfg_path, artifact_path="config")

                                    model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                                    model.save(model_path)
                                    mlflow.log_artifact(model_path, artifact_path="model")

                                    preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                                    joblib.dump(preprocessor, preproc_path)
                                    mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                                for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                                    history_rows.append(
                                        {
                                            "task": "clasificacion_multiclase",
                                            "feature_set": feature_set_name,
                                            "config_id": config_id,
                                            "run_id": run_id,
                                            "run_label": f"multi-{config_id}",
                                            "epoch": epoch_idx,
                                            "loss": loss,
                                            "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                        }
                                    )

                                results.append(
                                    {
                                        "task": "clasificacion_multiclase",
                                        "model": config["name"],
                                        "feature_set": feature_set_name,
                                        "config_id": config_id,
                                        "layers": "-".join([str(x) for x in config["layers"]]),
                                        "dropout": config["dropout"],
                                        "activation": config["activation"],
                                        "optimizer": config["optimizer"],
                                        "learning_rate": config["learning_rate"],
                                        "epochs": config["epochs"],
                                        "batch_size": config["batch_size"],
                                        "loss": config["loss"],
                                        "rmse": None,
                                        "mae": None,
                                        "r2": None,
                                        "accuracy": metricas["accuracy"],
                                        "f1": None,
                                        "roc_auc": None,
                                        "f1_macro": metricas["f1_macro"],
                                        "log_loss": metricas.get("log_loss"),
                                        "run_id": run_id,
                                    }
                                )

                                if metricas["f1_macro"] > best_tracker["clasificacion_multiclase"]["metric"]:
                                    best_tracker["clasificacion_multiclase"] = {
                                        "metric": metricas["f1_macro"],
                                        "metrics": metricas,
                                    }
                                    _guardar_mejor_modelo_p3(
                                        "clasificacion_multiclase",
                                        model,
                                        preprocessor,
                                        config,
                                        metricas,
                                        features,
                                        label_mapping=list(le.classes_),
                                    )
                        except Exception as exc:
                            errores.append(f"{run_name}: {exc}")

    resumen_df = pd.DataFrame(results)
    history_df = pd.DataFrame(history_rows)
    mejores_df = seleccionar_mejores_modelos_resultados(resumen_df)
    interpretacion = generar_interpretacion_p3(resumen_df, mejores_df, dff)

    if resumen_df.empty:
        return resumen_df, history_df, mejores_df, interpretacion, "No se generaron corridas."

    mensaje = f"Corridas ejecutadas: {len(resumen_df)}"
    if errores:
        mensaje = f"{mensaje} | Corridas con error: {len(errores)}"

    return resumen_df, history_df, mejores_df, interpretacion, mensaje


def entrenar_modelos_p3_personalizados(df, variables, max_rows=20000, random_state=42):
    try:
        import mlflow
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        import tensorflow as tf
        import joblib
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", f"Dependencias no disponibles ({exc})."

    if not variables:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", "Selecciona al menos una variable."

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", "No hay datos para modelar."

    dff = _build_modeling_frame(df)
    if max_rows and len(dff) > max_rows:
        dff = dff.sample(n=max_rows, random_state=random_state)

    numeric_vars = [
        v for v in ["internet_flag", "computador_flag", "tic_score", "tic_interaccion", "bilingue_flag"]
        if v in variables
    ]
    categorical_vars = [
        v for v in ["estrato_cat", "zona", "tipo_colegio", "jornada", "genero", "edu_padre", "edu_madre"]
        if v in variables
    ]

    features = numeric_vars + categorical_vars
    if not features:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", "No hay variables validas seleccionadas."

    base_dir = _p3_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs(Path(base_dir) / "mlruns" / "pregunta_3", exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(P3_EXPERIMENT_NAME)

    configs = _configuraciones_modelos_p3_ampliadas()
    results = []
    history_rows = []
    errores = []

    best_tracker = {
        "regresion": {"metric": np.inf},
        "clasificacion_binaria": {"metric": -np.inf},
        "clasificacion_multiclase": {"metric": -np.inf},
    }

    def _build_preprocess(num_features, cat_features):
        transformers = []
        if num_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                )
            )
        if cat_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_features,
                )
            )
        return ColumnTransformer(transformers=transformers) if transformers else None

    feature_set_name = "custom"
    selected_vars_tag = ",".join(features)
    custom_models_dir = os.path.join(_p3_models_dir(), "custom")

    # --------------------- REGRESION ---------------------
    if "punt_ingles" in dff.columns:
        data_reg = dff[features + ["punt_ingles"]].dropna(subset=["punt_ingles"])
        if not data_reg.empty:
            X = data_reg[features]
            y = data_reg["punt_ingles"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            X_train = _coerce_numeric_columns(X_train, numeric_vars)
            X_test = _coerce_numeric_columns(X_test, numeric_vars)
            X_train = _sanitize_missing_columns(X_train, categorical_vars)
            X_test = _sanitize_missing_columns(X_test, categorical_vars)

            preprocessor = _build_preprocess(numeric_vars, categorical_vars)
            X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
            X_test_proc = _to_dense(preprocessor.transform(X_test))

            for idx, config in enumerate(configs["regresion"], start=1):
                run_name = f"p3_custom_reg_{idx}"
                config_id = f"custom_reg_{idx}"
                try:
                    with mlflow.start_run(run_name=run_name):
                        loss_fn = _loss_from_config(tf, config["loss"])
                        model = _crear_mlp(
                            tf,
                            X_train_proc.shape[1],
                            config,
                            output_dim=1,
                            output_activation="linear",
                            loss=loss_fn,
                        )

                        history = model.fit(
                            X_train_proc,
                            y_train,
                            validation_split=0.2,
                            epochs=config["epochs"],
                            batch_size=config["batch_size"],
                            verbose=0,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    patience=5,
                                    restore_best_weights=True,
                                )
                            ],
                        )

                        preds = model.predict(X_test_proc, verbose=0).reshape(-1)
                        metricas = _evaluar_regresion(y_test, preds)

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
                            cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                            with open(cfg_path, "w", encoding="utf-8") as f:
                                json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                            mlflow.log_artifact(hist_path, artifact_path="history")
                            mlflow.log_artifact(cfg_path, artifact_path="config")

                            model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                            model.save(model_path)
                            mlflow.log_artifact(model_path, artifact_path="model")

                            preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                            joblib.dump(preprocessor, preproc_path)
                            mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                        for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                            history_rows.append(
                                {
                                    "task": "regresion",
                                    "feature_set": feature_set_name,
                                    "config_id": config_id,
                                    "run_id": run_id,
                                    "run_label": f"reg-{config_id}",
                                    "epoch": epoch_idx,
                                    "loss": loss,
                                    "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                }
                            )

                        results.append(
                            {
                                "task": "regresion",
                                "model": config["name"],
                                "feature_set": feature_set_name,
                                "config_id": config_id,
                                "layers": "-".join([str(x) for x in config["layers"]]),
                                "dropout": config["dropout"],
                                "activation": config["activation"],
                                "optimizer": config["optimizer"],
                                "learning_rate": config["learning_rate"],
                                "epochs": config["epochs"],
                                "batch_size": config["batch_size"],
                                "loss": config["loss"],
                                "rmse": metricas["rmse"],
                                "mae": metricas["mae"],
                                "r2": metricas["r2"],
                                "accuracy": None,
                                "f1": None,
                                "roc_auc": None,
                                "f1_macro": None,
                                "log_loss": None,
                                "run_id": run_id,
                            }
                        )

                        if metricas["rmse"] < best_tracker["regresion"]["metric"]:
                            best_tracker["regresion"] = {"metric": metricas["rmse"], "metrics": metricas}
                            _guardar_mejor_modelo_p3(
                                "regresion",
                                model,
                                preprocessor,
                                config,
                                metricas,
                                features,
                                base_dir=custom_models_dir,
                            )
                except Exception as exc:
                    errores.append(f"{run_name}: {exc}")

    # --------------------- CLASIFICACION BINARIA ---------------------
    if "desemp_ingles" in dff.columns:
        niveles_validos = ["A-", "A1", "A2", "B1", "B+"]
        data_clf = dff[dff["desemp_ingles"].isin(niveles_validos)].copy()
        if not data_clf.empty:
            data_clf["target_bin"] = data_clf["desemp_ingles"].isin(["B1", "B+"]).astype(int)
            data_bin = data_clf[features + ["target_bin"]].dropna(subset=["target_bin"])
            if not data_bin.empty and data_bin["target_bin"].nunique() > 1:
                X = data_bin[features]
                y = data_bin["target_bin"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state, stratify=y
                )

                X_train = _coerce_numeric_columns(X_train, numeric_vars)
                X_test = _coerce_numeric_columns(X_test, numeric_vars)
                X_train = _sanitize_missing_columns(X_train, categorical_vars)
                X_test = _sanitize_missing_columns(X_test, categorical_vars)

                preprocessor = _build_preprocess(numeric_vars, categorical_vars)
                X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
                X_test_proc = _to_dense(preprocessor.transform(X_test))

                for idx, config in enumerate(configs["clasificacion_binaria"], start=1):
                    run_name = f"p3_custom_bin_{idx}"
                    config_id = f"custom_bin_{idx}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            model = _crear_mlp(
                                tf,
                                X_train_proc.shape[1],
                                config,
                                output_dim=1,
                                output_activation="sigmoid",
                                loss=config["loss"],
                                metrics=["accuracy"],
                            )

                            history = model.fit(
                                X_train_proc,
                                y_train,
                                validation_split=0.2,
                                epochs=config["epochs"],
                                batch_size=config["batch_size"],
                                verbose=0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        patience=5,
                                        restore_best_weights=True,
                                    )
                                ],
                            )

                            proba = model.predict(X_test_proc, verbose=0).reshape(-1)
                            pred = (proba >= 0.5).astype(int)
                            metricas = _evaluar_clasificacion_binaria(y_test, pred, proba)

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
                                cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                                with open(cfg_path, "w", encoding="utf-8") as f:
                                    json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                                mlflow.log_artifact(hist_path, artifact_path="history")
                                mlflow.log_artifact(cfg_path, artifact_path="config")

                                model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                                model.save(model_path)
                                mlflow.log_artifact(model_path, artifact_path="model")

                                preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                                joblib.dump(preprocessor, preproc_path)
                                mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                            for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                                history_rows.append(
                                    {
                                        "task": "clasificacion_binaria",
                                        "feature_set": feature_set_name,
                                        "config_id": config_id,
                                        "run_id": run_id,
                                        "run_label": f"bin-{config_id}",
                                        "epoch": epoch_idx,
                                        "loss": loss,
                                        "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                    }
                                )

                            results.append(
                                {
                                    "task": "clasificacion_binaria",
                                    "model": config["name"],
                                    "feature_set": feature_set_name,
                                    "config_id": config_id,
                                    "layers": "-".join([str(x) for x in config["layers"]]),
                                    "dropout": config["dropout"],
                                    "activation": config["activation"],
                                    "optimizer": config["optimizer"],
                                    "learning_rate": config["learning_rate"],
                                    "epochs": config["epochs"],
                                    "batch_size": config["batch_size"],
                                    "loss": config["loss"],
                                    "rmse": None,
                                    "mae": None,
                                    "r2": None,
                                    "accuracy": metricas["accuracy"],
                                    "f1": metricas["f1"],
                                    "roc_auc": metricas.get("roc_auc"),
                                    "f1_macro": None,
                                    "log_loss": None,
                                    "run_id": run_id,
                                }
                            )

                            if metricas["f1"] > best_tracker["clasificacion_binaria"]["metric"]:
                                best_tracker["clasificacion_binaria"] = {"metric": metricas["f1"], "metrics": metricas}
                                _guardar_mejor_modelo_p3(
                                    "clasificacion_binaria",
                                    model,
                                    preprocessor,
                                    config,
                                    metricas,
                                    features,
                                    base_dir=custom_models_dir,
                                )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

            # --------------------- CLASIFICACION MULTICLASE ---------------------
            data_multi = data_clf[features + ["desemp_ingles"]].dropna(subset=["desemp_ingles"])
            if not data_multi.empty and data_multi["desemp_ingles"].nunique() > 1:
                X = data_multi[features]
                y_raw = data_multi["desemp_ingles"].astype(str)
                le = LabelEncoder()
                y = le.fit_transform(y_raw)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=random_state, stratify=y
                )

                X_train = _coerce_numeric_columns(X_train, numeric_vars)
                X_test = _coerce_numeric_columns(X_test, numeric_vars)
                X_train = _sanitize_missing_columns(X_train, categorical_vars)
                X_test = _sanitize_missing_columns(X_test, categorical_vars)

                preprocessor = _build_preprocess(numeric_vars, categorical_vars)
                X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
                X_test_proc = _to_dense(preprocessor.transform(X_test))

                y_train_cat = tf.keras.utils.to_categorical(y_train)
                y_test_cat = tf.keras.utils.to_categorical(y_test)

                for idx, config in enumerate(configs["clasificacion_multiclase"], start=1):
                    run_name = f"p3_custom_multi_{idx}"
                    config_id = f"custom_multi_{idx}"
                    try:
                        with mlflow.start_run(run_name=run_name):
                            model = _crear_mlp(
                                tf,
                                X_train_proc.shape[1],
                                config,
                                output_dim=len(le.classes_),
                                output_activation="softmax",
                                loss=config["loss"],
                                metrics=["accuracy"],
                            )

                            history = model.fit(
                                X_train_proc,
                                y_train_cat,
                                validation_split=0.2,
                                epochs=config["epochs"],
                                batch_size=config["batch_size"],
                                verbose=0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        patience=5,
                                        restore_best_weights=True,
                                    )
                                ],
                            )

                            proba = model.predict(X_test_proc, verbose=0)
                            pred = np.argmax(proba, axis=1)
                            metricas = _evaluar_clasificacion_multi(y_test, pred, proba, labels=list(range(len(le.classes_))))

                            mlflow.set_tag("task", "clasificacion_multiclase")
                            mlflow.set_tag("feature_set", feature_set_name)
                            mlflow.set_tag("model_name", config["name"])
                            mlflow.set_tag("selected_vars", selected_vars_tag)
                            mlflow.log_params(_serializable_params(config))
                            mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None})

                            run_id = mlflow.active_run().info.run_id

                            with tempfile.TemporaryDirectory() as tmpdir:
                                hist_path = os.path.join(tmpdir, f"history_{run_id}.csv")
                                pd.DataFrame(history.history).to_csv(hist_path, index=False)
                                cfg_path = os.path.join(tmpdir, f"config_{run_id}.json")
                                with open(cfg_path, "w", encoding="utf-8") as f:
                                    json.dump(_serializable_params(config), f, ensure_ascii=True, indent=2)
                                mlflow.log_artifact(hist_path, artifact_path="history")
                                mlflow.log_artifact(cfg_path, artifact_path="config")

                                model_path = os.path.join(tmpdir, f"model_{run_id}.keras")
                                model.save(model_path)
                                mlflow.log_artifact(model_path, artifact_path="model")

                                preproc_path = os.path.join(tmpdir, f"preproc_{run_id}.pkl")
                                joblib.dump(preprocessor, preproc_path)
                                mlflow.log_artifact(preproc_path, artifact_path="preprocessor")

                            for epoch_idx, loss in enumerate(history.history.get("loss", []), start=1):
                                history_rows.append(
                                    {
                                        "task": "clasificacion_multiclase",
                                        "feature_set": feature_set_name,
                                        "config_id": config_id,
                                        "run_id": run_id,
                                        "run_label": f"multi-{config_id}",
                                        "epoch": epoch_idx,
                                        "loss": loss,
                                        "val_loss": history.history.get("val_loss", [None] * len(history.history.get("loss", [])))[epoch_idx - 1],
                                    }
                                )

                            results.append(
                                {
                                    "task": "clasificacion_multiclase",
                                    "model": config["name"],
                                    "feature_set": feature_set_name,
                                    "config_id": config_id,
                                    "layers": "-".join([str(x) for x in config["layers"]]),
                                    "dropout": config["dropout"],
                                    "activation": config["activation"],
                                    "optimizer": config["optimizer"],
                                    "learning_rate": config["learning_rate"],
                                    "epochs": config["epochs"],
                                    "batch_size": config["batch_size"],
                                    "loss": config["loss"],
                                    "rmse": None,
                                    "mae": None,
                                    "r2": None,
                                    "accuracy": metricas["accuracy"],
                                    "f1": None,
                                    "roc_auc": None,
                                    "f1_macro": metricas["f1_macro"],
                                    "log_loss": metricas.get("log_loss"),
                                    "run_id": run_id,
                                }
                            )

                            if metricas["f1_macro"] > best_tracker["clasificacion_multiclase"]["metric"]:
                                best_tracker["clasificacion_multiclase"] = {"metric": metricas["f1_macro"], "metrics": metricas}
                                _guardar_mejor_modelo_p3(
                                    "clasificacion_multiclase",
                                    model,
                                    preprocessor,
                                    config,
                                    metricas,
                                    features,
                                    label_mapping=list(le.classes_),
                                    base_dir=custom_models_dir,
                                )
                    except Exception as exc:
                        errores.append(f"{run_name}: {exc}")

    resumen_df = pd.DataFrame(results)
    history_df = pd.DataFrame(history_rows)
    mejores_df = seleccionar_mejores_modelos_resultados(resumen_df)
    interpretacion = generar_interpretacion_p3(resumen_df, mejores_df, dff)

    if resumen_df.empty:
        return resumen_df, history_df, mejores_df, interpretacion, "No se generaron corridas personalizadas."

    mensaje = f"Corridas personalizadas: {len(resumen_df)}"
    if errores:
        mensaje = f"{mensaje} | Corridas con error: {len(errores)}"
    return resumen_df, history_df, mejores_df, interpretacion, mensaje


def predecir_escenarios_p3(valores_a, valores_b):
    artefactos = cargar_modelos_p3()
    if not artefactos.get("disponible"):
        return None, None, None, None, artefactos.get("error")

    def _pred_uno(valores):
        df_input = construir_input_p3(valores)

        reg = artefactos["regresion"]
        X_reg = _to_dense(reg["preprocessor"].transform(df_input[reg["metadata"]["feature_columns"]]))
        puntaje = float(reg["model"].predict(X_reg, verbose=0).reshape(-1)[0])

        bin_m = artefactos["clasificacion_binaria"]
        X_bin = _to_dense(bin_m["preprocessor"].transform(df_input[bin_m["metadata"]["feature_columns"]]))
        proba_b1 = float(bin_m["model"].predict(X_bin, verbose=0).reshape(-1)[0])

        multi = artefactos["clasificacion_multiclase"]
        X_multi = _to_dense(multi["preprocessor"].transform(df_input[multi["metadata"]["feature_columns"]]))
        proba_multi = multi["model"].predict(X_multi, verbose=0).reshape(-1)
        labels = multi["metadata"].get("label_mapping", [])
        nivel = labels[int(np.argmax(proba_multi))] if labels else "N/D"

        return {
            "puntaje": puntaje,
            "proba_b1": proba_b1,
            "nivel": nivel,
        }

    pred_a = _pred_uno(valores_a)
    pred_b = _pred_uno(valores_b)
    fig_reg, fig_clf = grafica_comparacion_escenarios(pred_a, pred_b)
    return pred_a, pred_b, fig_reg, fig_clf, None


def grafica_comparacion_escenarios(pred_a, pred_b):
    if not pred_a or not pred_b:
        empty = go.Figure().update_layout(title="Modelos no disponibles")
        return empty, empty

    df_reg = pd.DataFrame(
        {
            "Escenario": ["A", "B"],
            "Puntaje Ingles": [pred_a["puntaje"], pred_b["puntaje"]],
        }
    )
    fig_reg = px.bar(
        df_reg,
        x="Escenario",
        y="Puntaje Ingles",
        title="Comparacion de escenarios - Regresion (puntaje ingles)",
        labels={"Puntaje Ingles": "Puntaje Ingles"},
    )

    df_clf = pd.DataFrame(
        {
            "Escenario": ["A", "B"],
            "Probabilidad B1+ (%)": [pred_a["proba_b1"] * 100, pred_b["proba_b1"] * 100],
        }
    )
    fig_clf = px.bar(
        df_clf,
        x="Escenario",
        y="Probabilidad B1+ (%)",
        title="Comparacion de escenarios - Clasificacion (B1+)",
        labels={"Probabilidad B1+ (%)": "Probabilidad B1+ (%)"},
    )

    for fig in [fig_reg, fig_clf]:
        fig.update_layout(margin={"r": 10, "t": 40, "l": 10, "b": 10})

    return fig_reg, fig_clf


def analizar_significancia_ols(df, variables_seleccionadas, sample_rows=20000):
    try:
        import statsmodels.api as sm
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), f"Statsmodels no disponible ({exc})."

    try:
        dff = _build_modeling_frame(df)
        if dff.empty or "punt_ingles" not in dff.columns:
            return {}, pd.DataFrame(), pd.DataFrame(), "No hay datos para regresion."

        y = pd.to_numeric(dff["punt_ingles"], errors="coerce")
        dff = dff.loc[y.notna()].copy()
        y = y.loc[y.notna()]

        if sample_rows and len(dff) > sample_rows:
            dff = dff.sample(n=sample_rows, random_state=42)
            y = y.loc[dff.index]

        numeric_vars = [
            v for v in ["internet_flag", "computador_flag", "tic_score", "tic_interaccion", "bilingue_flag"]
            if v in dff.columns
        ]
        categorical_vars = [
            v for v in ["estrato_cat", "zona", "tipo_colegio", "jornada", "genero", "edu_padre", "edu_madre"]
            if v in dff.columns
        ]

        dff = _coerce_numeric_columns(dff, numeric_vars)
        for col in numeric_vars:
            dff[col] = dff[col].fillna(dff[col].median())

        for col in categorical_vars:
            dff[col] = dff[col].astype("string").fillna("SIN_INFO")

        X_num = dff[numeric_vars]
        X_cat = pd.get_dummies(
            dff[categorical_vars],
            prefix=categorical_vars,
            prefix_sep="__",
            drop_first=True,
        ) if categorical_vars else pd.DataFrame(index=dff.index)

        X_full = pd.concat([X_num, X_cat], axis=1)
        X_full = X_full.loc[y.index]
        X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_full = X_full.astype(float)
        y = pd.to_numeric(y, errors="coerce").astype(float)

        if X_full.empty or y.isna().all():
            return {}, pd.DataFrame(), pd.DataFrame(), "No hay datos numericos suficientes para OLS."

        model_full = sm.OLS(y, sm.add_constant(X_full)).fit()

        # Map variable -> columns in design matrix
        group_map = {}
        for var in numeric_vars:
            if var in X_full.columns:
                group_map[var] = [var]
        for var in categorical_vars:
            prefix = f"{var}__"
            group_cols = [c for c in X_full.columns if c.startswith(prefix)]
            if group_cols:
                group_map[var] = group_cols

        # Selected model
        selected_vars = variables_seleccionadas or []
        selected_cols = []
        for var in selected_vars:
            selected_cols.extend(group_map.get(var, []))
        selected_cols = list(dict.fromkeys(selected_cols))
        if not selected_cols:
            return {}, pd.DataFrame(), pd.DataFrame(), "Selecciona al menos una variable independiente."

        X_sel = X_full[selected_cols]
        X_sel = X_sel.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_sel = X_sel.astype(float)
        model_sel = sm.OLS(y, sm.add_constant(X_sel)).fit()

        resumen = {
            "n_obs": int(model_sel.nobs),
            "r2": float(model_sel.rsquared),
            "adj_r2": float(model_sel.rsquared_adj),
        }

        # T-test for numeric variables included
        ttest_rows = []
        for var in numeric_vars:
            if var not in selected_vars:
                continue
            if var not in model_sel.params:
                continue
            ttest_rows.append(
                {
                    "variable": var,
                    "coef": float(model_sel.params[var]),
                    "t_stat": float(model_sel.tvalues[var]),
                    "p_value": float(model_sel.pvalues[var]),
                    "significativa": "Si" if model_sel.pvalues[var] < 0.05 else "No",
                }
            )
        ttest_df = pd.DataFrame(ttest_rows)

        # F-test for each variable group and combinations (full model)
        ftest_rows = []
        exog_names = model_full.model.exog_names
        group_items = list(group_map.items())

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

        # Individual variables
        for var, cols in group_items:
            if not cols:
                continue
            p_val = _f_test_for_columns(cols)
            ftest_rows.append(
                {
                    "variable": var,
                    "p_value": p_val,
                    "incluida": "Si" if var in selected_vars else "No",
                    "significativa": "Si" if p_val is not None and p_val < 0.05 else "No",
                }
            )

        # All combinations from selected variables (2..N)
        comb_vars = [var for var in selected_vars if var in group_map]
        if len(comb_vars) >= 2:
            for k in range(2, len(comb_vars) + 1):
                for combo in itertools.combinations(comb_vars, k):
                    combo_cols = []
                    for var in combo:
                        combo_cols.extend(group_map.get(var, []))
                    p_val = _f_test_for_columns(combo_cols)
                    ftest_rows.append(
                        {
                            "variable": " + ".join(combo),
                            "p_value": p_val,
                            "incluida": "Si",
                            "significativa": "Si" if p_val is not None and p_val < 0.05 else "No",
                        }
                    )

        ftest_df = pd.DataFrame(ftest_rows)
        mensaje = "Analisis OLS listo."
        if ttest_df.empty:
            mensaje = "No hay variables numericas seleccionadas para pruebas t."
        return resumen, ttest_df, ftest_df, mensaje
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), f"No se pudo calcular significancia ({exc})."