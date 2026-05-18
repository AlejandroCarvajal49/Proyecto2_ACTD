import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

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

    dff["estrato_num"] = pd.to_numeric(dff[col_estrato], errors="coerce") if col_estrato else np.nan

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
    contexto_numeric = base_numeric + ["estrato_num", "periodo_year", "bilingue_flag"]
    contexto_categoric = ["zona", "tipo_colegio", "jornada", "genero"]

    feature_sets = {
        "tic_basico": {
            "numeric": base_numeric,
            "categorical": [],
        },
        "tic_contexto": {
            "numeric": contexto_numeric,
            "categorical": contexto_categoric,
        },
        "tic_contexto_municipio": {
            "numeric": contexto_numeric,
            "categorical": contexto_categoric + ["municipio"],
        },
    }

    # Filtrar solo columnas existentes
    for key, cfg in feature_sets.items():
        cfg["numeric"] = [c for c in cfg["numeric"] if c in df.columns]
        cfg["categorical"] = [c for c in cfg["categorical"] if c in df.columns]

    return feature_sets


def _resolve_mlflow_tracking_uri(base_dir):
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return (Path(base_dir) / "mlruns").as_uri()


def obtener_mlflow_info():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_path = Path(base_dir) / "mlruns"
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    ui_url = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")
    return {
        "tracking_uri": tracking_uri,
        "mlruns_path": str(mlruns_path),
        "ui_url": ui_url,
    }


def obtener_resumen_mlflow_p3(max_runs=200):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        return pd.DataFrame(), f"MLflow no disponible ({exc})."

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracking_uri = _resolve_mlflow_tracking_uri(base_dir)
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    exp_names = [
        "P3_ingles_regresion",
        "P3_ingles_clasificacion_binaria",
        "P3_ingles_clasificacion_multiclase",
    ]

    rows = []
    for exp_name in exp_names:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            continue

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
                    "experiment": exp_name,
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


def ejecutar_experimentos_mlflow_p3(df, max_rows=60000, random_state=42):
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