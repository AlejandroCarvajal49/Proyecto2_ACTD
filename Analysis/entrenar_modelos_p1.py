"""
Entrenamiento de los modelos de la Pregunta 1 (Proyecto 2) - Pipeline P3-style.

Para cada feature set y cada configuracion de arquitectura del catalogo en
`logica_p1._configuraciones_modelos_p1()`:
    1) Entrena la red densa correspondiente.
    2) Registra parametros, metricas, history (CSV) y modelo en MLflow,
       bajo el experimento "pregunta_1" en mlruns/pregunta_1/.
    3) Si esa corrida supera al mejor visto para esa task, guarda el modelo
       en models/pregunta_1/<task>/best/ (model.keras + preprocessor.pkl + metadata.json).

Tasks:
    - regresion              (target = punt_global)
    - clasificacion_binaria  (target = prioridad alta, punt_global < P25)

Feature sets (definidos en logica_p1._build_feature_sets_p1):
    - basico, socioeconomico, completo

Ejecutar desde la raiz del proyecto:
    python -m Analysis.entrenar_modelos_p1
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

# Reducir ruido de TF antes de importar.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.model_selection import train_test_split

import mlflow

# Permitir ejecutar tanto "python Analysis/entrenar_modelos_p1.py" como
# "python -m Analysis.entrenar_modelos_p1" sin romper el import relativo.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Analysis.logica_p1 import (
    cargar_datos_p1,
    P1_EXPERIMENT_NAME,
    _p1_base_dir,
    _p1_models_dir,
    _resolve_mlflow_tracking_uri_p1,
    _build_feature_sets_p1,
    _build_preprocessor_p1,
    _configuraciones_modelos_p1,
    _crear_mlp_p1,
    _loss_from_config,
    _evaluar_regresion,
    _evaluar_clasificacion_binaria,
    _guardar_mejor_modelo_p1,
    _serializable_params,
    _coerce_numeric_columns,
    _sanitize_missing_columns,
    _to_dense,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_REG = "punt_global"


def _setup_mlflow():
    base_dir = _p1_base_dir()
    tracking_uri = _resolve_mlflow_tracking_uri_p1(base_dir)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs(os.path.join(base_dir, "mlruns", "pregunta_1"), exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(P1_EXPERIMENT_NAME)
    print(f"      MLflow tracking URI: {tracking_uri}")
    print(f"      MLflow experiment  : {P1_EXPERIMENT_NAME}")


def _cargar_dataset_entrenamiento():
    print("[1/4] Cargando dataset...")
    df = cargar_datos_p1()
    df = df.copy()

    # Forzar tipos consistentes para el preprocesador.
    from Analysis.logica_p1 import COLUMNAS_CATEGORICAS_P2, COLUMNAS_NUMERICAS_P2
    for col in COLUMNAS_CATEGORICAS_P2:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in COLUMNAS_NUMERICAS_P2:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_REG] = pd.to_numeric(df[TARGET_REG], errors="coerce")

    # Eliminar filas sin target.
    antes = len(df)
    df = df.dropna(subset=[TARGET_REG])
    print(f"      Filas usables: {len(df):,} (descartadas {antes - len(df):,} por target nulo).")
    return df


def _log_mlflow_artifacts(model, preprocessor, history, config, run_id):
    """Sube history.csv, config.json, model.keras y preprocessor.pkl como artefactos."""
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


def _entrenar_regresion_runs(df, feature_set_name, num_features, cat_features, umbral_p25,
                              best_tracker, models_dir, run_summary):
    if not (num_features or cat_features):
        return
    features = num_features + cat_features
    data = df[features + [TARGET_REG]].dropna(subset=[TARGET_REG])
    if data.empty:
        return

    X, y = data[features], data[TARGET_REG].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr = _coerce_numeric_columns(X_tr, num_features)
    X_te = _coerce_numeric_columns(X_te, num_features)
    X_tr = _sanitize_missing_columns(X_tr, cat_features)
    X_te = _sanitize_missing_columns(X_te, cat_features)

    preprocessor = _build_preprocessor_p1(num_features, cat_features)
    X_tr_proc = _to_dense(preprocessor.fit_transform(X_tr))
    X_te_proc = _to_dense(preprocessor.transform(X_te))

    for idx, config in enumerate(_configuraciones_modelos_p1()["regresion"], start=1):
        run_name = f"p1_reg_{feature_set_name}_{idx}"
        try:
            with mlflow.start_run(run_name=run_name):
                loss_fn = _loss_from_config(tf, config["loss"])
                model = _crear_mlp_p1(
                    tf, X_tr_proc.shape[1], config,
                    output_dim=1, output_activation="linear", loss=loss_fn, metrics=["mae"],
                )
                history = model.fit(
                    X_tr_proc, y_tr,
                    validation_split=0.15,
                    epochs=config["epochs"], batch_size=config["batch_size"], verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
                )
                preds = model.predict(X_te_proc, verbose=0).reshape(-1)
                metricas = _evaluar_regresion(y_te, preds)

                mlflow.set_tag("task", "regresion")
                mlflow.set_tag("feature_set", feature_set_name)
                mlflow.set_tag("model_name", config["name"])
                mlflow.set_tag("selected_vars", ",".join(features))
                mlflow.log_params(_serializable_params(config))
                mlflow.log_metrics(metricas)
                run_id = mlflow.active_run().info.run_id
                _log_mlflow_artifacts(model, preprocessor, history, config, run_id)

                run_summary.append({
                    "task": "regresion", "feature_set": feature_set_name, "model": config["name"],
                    "rmse": metricas["rmse"], "mae": metricas["mae"], "r2": metricas["r2"],
                    "run_id": run_id,
                })
                print(f"      [reg/{feature_set_name}/{config['name']}] RMSE={metricas['rmse']:.2f}  R2={metricas['r2']:.3f}")

                if metricas["rmse"] < best_tracker["regresion"]["metric"]:
                    best_tracker["regresion"] = {"metric": metricas["rmse"], "metrics": metricas}
                    _guardar_mejor_modelo_p1(
                        "regresion", model, preprocessor, config, metricas,
                        features, umbral_p25=umbral_p25, base_dir=models_dir,
                    )
        except Exception as exc:
            print(f"      [reg/{feature_set_name}/{config['name']}] ERROR: {exc}")


def _entrenar_clasificacion_runs(df, feature_set_name, num_features, cat_features, umbral_p25,
                                  best_tracker, models_dir, run_summary):
    if not (num_features or cat_features):
        return
    features = num_features + cat_features

    df_clf = df.copy()
    df_clf["__prioridad_alta__"] = (
        pd.to_numeric(df_clf[TARGET_REG], errors="coerce") < umbral_p25
    ).astype(int)
    data = df_clf[features + ["__prioridad_alta__"]].dropna(subset=["__prioridad_alta__"])
    if data.empty or data["__prioridad_alta__"].nunique() < 2:
        return

    X, y = data[features], data["__prioridad_alta__"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    X_tr = _coerce_numeric_columns(X_tr, num_features)
    X_te = _coerce_numeric_columns(X_te, num_features)
    X_tr = _sanitize_missing_columns(X_tr, cat_features)
    X_te = _sanitize_missing_columns(X_te, cat_features)

    preprocessor = _build_preprocessor_p1(num_features, cat_features)
    X_tr_proc = _to_dense(preprocessor.fit_transform(X_tr))
    X_te_proc = _to_dense(preprocessor.transform(X_te))

    pos = float(y_tr.mean())
    class_weight = None
    if 0.0 < pos < 1.0:
        class_weight = {0: 1.0 / (2.0 * (1.0 - pos)), 1: 1.0 / (2.0 * pos)}

    for idx, config in enumerate(_configuraciones_modelos_p1()["clasificacion_binaria"], start=1):
        run_name = f"p1_bin_{feature_set_name}_{idx}"
        try:
            with mlflow.start_run(run_name=run_name):
                model = _crear_mlp_p1(
                    tf, X_tr_proc.shape[1], config,
                    output_dim=1, output_activation="sigmoid",
                    loss=config["loss"], metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
                )
                history = model.fit(
                    X_tr_proc, y_tr,
                    validation_split=0.15,
                    epochs=config["epochs"], batch_size=config["batch_size"], verbose=2,
                    class_weight=class_weight,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
                )
                proba = model.predict(X_te_proc, verbose=0).reshape(-1)
                pred = (proba >= 0.5).astype(int)
                metricas = _evaluar_clasificacion_binaria(y_te, pred, proba)

                mlflow.set_tag("task", "clasificacion_binaria")
                mlflow.set_tag("feature_set", feature_set_name)
                mlflow.set_tag("model_name", config["name"])
                mlflow.set_tag("selected_vars", ",".join(features))
                mlflow.log_params(_serializable_params(config))
                mlflow.log_metrics({k: v for k, v in metricas.items() if v is not None})
                run_id = mlflow.active_run().info.run_id
                _log_mlflow_artifacts(model, preprocessor, history, config, run_id)

                run_summary.append({
                    "task": "clasificacion_binaria", "feature_set": feature_set_name,
                    "model": config["name"],
                    "accuracy": metricas["accuracy"], "f1": metricas["f1"],
                    "roc_auc": metricas.get("roc_auc"),
                    "run_id": run_id,
                })
                print(f"      [bin/{feature_set_name}/{config['name']}] Acc={metricas['accuracy']:.3f} F1={metricas['f1']:.3f} AUC={metricas.get('roc_auc', 0):.3f}")

                if metricas["f1"] > best_tracker["clasificacion_binaria"]["metric"]:
                    best_tracker["clasificacion_binaria"] = {"metric": metricas["f1"], "metrics": metricas}
                    _guardar_mejor_modelo_p1(
                        "clasificacion_binaria", model, preprocessor, config, metricas,
                        features, umbral_p25=umbral_p25, base_dir=models_dir,
                    )
        except Exception as exc:
            print(f"      [bin/{feature_set_name}/{config['name']}] ERROR: {exc}")


def _eliminar_artefactos_legacy():
    """Borra los archivos del esquema anterior (weights.h5 + pkls planos) para evitar confusion."""
    legacy_paths = [
        os.path.join("models", "regresion_puntaje.weights.h5"),
        os.path.join("models", "clasificacion_prioridad.weights.h5"),
        os.path.join("models", "preprocesador_reg.pkl"),
        os.path.join("models", "preprocesador_clf.pkl"),
        os.path.join("models", "metadata_modelos.pkl"),
    ]
    for p in legacy_paths:
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"      Eliminado legacy: {p}")
            except OSError:
                pass


def main():
    t0 = time.time()
    _setup_mlflow()
    df = _cargar_dataset_entrenamiento()

    umbral_p25 = float(df[TARGET_REG].quantile(0.25))
    print(f"      Umbral P25 puntaje global = {umbral_p25:.2f}")

    feature_sets = _build_feature_sets_p1(df)
    models_dir = _p1_models_dir()
    best_tracker = {
        "regresion": {"metric": np.inf},
        "clasificacion_binaria": {"metric": -np.inf},
    }
    run_summary = []

    n_combinaciones = sum(
        len(_configuraciones_modelos_p1()[t]) for t in ("regresion", "clasificacion_binaria")
    ) * len(feature_sets)
    print(f"[2/4] Entrenando {n_combinaciones} corridas ({len(feature_sets)} feature sets x configs)...")

    for fs_name, cfg in feature_sets.items():
        num_features, cat_features = cfg["numeric"], cfg["categorical"]
        print(f"\n   -- feature_set='{fs_name}' (num={len(num_features)} cat={len(cat_features)}) --")
        _entrenar_regresion_runs(
            df, fs_name, num_features, cat_features, umbral_p25,
            best_tracker, models_dir, run_summary,
        )
        _entrenar_clasificacion_runs(
            df, fs_name, num_features, cat_features, umbral_p25,
            best_tracker, models_dir, run_summary,
        )

    print(f"\n[3/4] Limpiando artefactos legacy del esquema anterior...")
    _eliminar_artefactos_legacy()

    print(f"\n[4/4] Listo en {time.time() - t0:.1f}s.")
    print(f"      Mejor regresion - RMSE = {best_tracker['regresion']['metric']:.3f}")
    print(f"      Mejor clf bin   - F1   = {best_tracker['clasificacion_binaria']['metric']:.3f}")
    print(f"      Best por task   : {models_dir}/<task>/best/")
    print(f"      Para ver runs   : mlflow ui --backend-store-uri mlruns/pregunta_1")


if __name__ == "__main__":
    main()
