"""
Entrenamiento de los modelos de la Pregunta 1 (Proyecto 2).

Genera dos redes neuronales (solo pesos) y sus preprocesadores:
    - regresion_puntaje.weights.h5       : pesos de la red de regresion (punt_global).
    - clasificacion_prioridad.weights.h5 : pesos de la red de clasificacion binaria
                                           (prioridad alta = punt_global < P25).

Se guardan SOLO los pesos porque el formato .keras completo es sensible a
cambios menores de version entre el entorno de entrenamiento y el de carga.
La arquitectura vive en Analysis.logica_p1 (build_modelo_regresion / _clasificacion)
y al cargar se reconstruye y se hace load_weights.

Tambien guarda los ColumnTransformer y un diccionario de metadata con:
    - columnas_reg / columnas_clf  : columnas crudas que esperan los preprocesadores.
    - input_dim_reg / input_dim_clf: shape de entrada para reconstruir los modelos.
    - umbral_p25                    : umbral del percentil 25 del puntaje global.
    - metricas                      : MAE, RMSE, R^2 / Accuracy, F1, AUC.

Ejecutar desde la raiz del proyecto:
    python -m Analysis.entrenar_modelos_p1
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

# Reducir ruido y forzar CPU si no hay GPU configurada.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.tensorflow

# Permitir ejecutar tanto "python Analysis/entrenar_modelos_p1.py" como
# "python -m Analysis.entrenar_modelos_p1" sin romper el import relativo.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Analysis.logica_p1 import (
    cargar_datos_p1,
    COLUMNAS_CATEGORICAS_P2,
    COLUMNAS_NUMERICAS_P2,
    build_modelo_regresion,
    build_modelo_clasificacion,
)


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MLRUNS_DIR = "mlruns"
EXPERIMENT_NAME = "pregunta_1_brecha_urbano_rural"
RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS_REG = 30
EPOCHS_CLF = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
TARGET_REG = "punt_global"


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def _construir_dataset():
    """Carga el CSV limpio, agrega PIB/Area y deja un DataFrame listo para entrenar."""
    print("[1/6] Cargando dataset...")
    df = cargar_datos_p1()
    columnas_features = COLUMNAS_CATEGORICAS_P2 + COLUMNAS_NUMERICAS_P2
    columnas_necesarias = columnas_features + [TARGET_REG]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        raise RuntimeError(
            f"Faltan columnas en el dataset para entrenar: {faltantes}. "
            "Revise cargar_datos_p1 y los CSV en Data/."
        )

    df = df[columnas_necesarias].copy()

    # Tipado explicito: las categoricas como string para que OneHotEncoder funcione
    # de manera estable independientemente de los dtypes originales (a veces 'category').
    for col in COLUMNAS_CATEGORICAS_P2:
        df[col] = df[col].astype("string")

    for col in COLUMNAS_NUMERICAS_P2:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_REG] = pd.to_numeric(df[TARGET_REG], errors="coerce")

    # Eliminamos filas con cualquier nulo en features o target para entrenar limpio.
    antes = len(df)
    df = df.dropna(subset=columnas_necesarias)
    print(f"      Filas usables: {len(df):,} (descartadas {antes - len(df):,} por nulos).")

    return df


def _construir_preprocesador():
    """ColumnTransformer con OneHot para categoricas y StandardScaler para numericas."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLUMNAS_CATEGORICAS_P2,
            ),
            (
                "num",
                StandardScaler(),
                COLUMNAS_NUMERICAS_P2,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _entrenar_regresion(X_tr, y_tr, X_te, y_te):
    print("[3/6] Entrenando red de regresion...")
    modelo = build_modelo_regresion(X_tr.shape[1])
    es = callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")
    historia = modelo.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=EPOCHS_REG,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2,
    )
    y_pred = modelo.predict(X_te, verbose=0).flatten()
    metricas = {
        "reg_mae": float(mean_absolute_error(y_te, y_pred)),
        "reg_rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "reg_r2": float(r2_score(y_te, y_pred)),
    }
    print(f"      MAE={metricas['reg_mae']:.2f}  RMSE={metricas['reg_rmse']:.2f}  R2={metricas['reg_r2']:.3f}")
    return modelo, metricas, historia


def _entrenar_clasificacion(X_tr, y_tr, X_te, y_te):
    print("[4/6] Entrenando red de clasificacion...")
    modelo = build_modelo_clasificacion(X_tr.shape[1])
    es = callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")
    pos = float(y_tr.mean())
    if 0.0 < pos < 1.0:
        peso_neg = 1.0 / (2.0 * (1.0 - pos))
        peso_pos = 1.0 / (2.0 * pos)
        class_weight = {0: peso_neg, 1: peso_pos}
    else:
        class_weight = None
    historia = modelo.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=EPOCHS_CLF,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2,
    )
    proba = modelo.predict(X_te, verbose=0).flatten()
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "clf_acc": float(accuracy_score(y_te, pred)),
        "clf_f1": float(f1_score(y_te, pred)),
        "clf_auc": float(roc_auc_score(y_te, proba)),
    }
    print(f"      Acc={metricas['clf_acc']:.3f}  F1={metricas['clf_f1']:.3f}  AUC={metricas['clf_auc']:.3f}")
    return modelo, metricas, historia


def _log_historia(historia, prefijo):
    """Loguea el historial epoch-a-epoch como metricas en MLflow."""
    if historia is None or not getattr(historia, "history", None):
        return
    for nombre, valores in historia.history.items():
        for epoca, valor in enumerate(valores, start=1):
            mlflow.log_metric(f"{prefijo}_{nombre}", float(valor), step=epoca)


def _setup_mlflow():
    """Configura mlruns/ local y devuelve el experiment_id."""
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    # file:./mlruns en formato URI valido en Windows (forward slashes).
    tracking_uri = "file:" + os.path.abspath(MLRUNS_DIR).replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"      MLflow tracking URI: {tracking_uri}")
    print(f"      MLflow experiment  : {EXPERIMENT_NAME}")


def main():
    _ensure_models_dir()
    _setup_mlflow()
    t0 = time.time()

    df = _construir_dataset()

    # Umbral P25 para clasificacion binaria de "prioridad alta".
    umbral_p25 = float(df[TARGET_REG].quantile(0.25))
    df["__prioridad_alta__"] = (df[TARGET_REG] < umbral_p25).astype(int)
    print(f"      Umbral P25 puntaje global = {umbral_p25:.2f}  (positivos = {df['__prioridad_alta__'].mean()*100:.1f}%)")

    X = df[COLUMNAS_CATEGORICAS_P2 + COLUMNAS_NUMERICAS_P2]
    y_reg = df[TARGET_REG].values
    y_clf = df["__prioridad_alta__"].values

    print("[2/6] Particionando train/test y ajustando preprocesadores...")
    X_tr, X_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
        X, y_reg, y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    # Usamos preprocesadores separados (los modelos podrian evolucionar por separado).
    preproc_reg = _construir_preprocesador()
    preproc_clf = _construir_preprocesador()

    X_tr_reg = preproc_reg.fit_transform(X_tr)
    X_te_reg = preproc_reg.transform(X_te)
    X_tr_clf = preproc_clf.fit_transform(X_tr)
    X_te_clf = preproc_clf.transform(X_te)

    # Params compartidos por toda la corrida (van en el run padre y se duplican
    # en los nested para facilitar comparacion en la UI).
    params_comunes = {
        "dataset_filas_totales": int(len(df)),
        "dataset_filas_train": int(len(X_tr)),
        "dataset_filas_test": int(len(X_te)),
        "features_categoricas": ",".join(COLUMNAS_CATEGORICAS_P2),
        "features_numericas": ",".join(COLUMNAS_NUMERICAS_P2),
        "input_dim": int(X_tr_reg.shape[1]),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "umbral_p25_puntaje_global": round(umbral_p25, 2),
    }

    with mlflow.start_run(run_name="entrenamiento_p1") as run_padre:
        mlflow.set_tag("proyecto", "Saber11 Antioquia - Pregunta 1")
        mlflow.set_tag("autor", "Santiago Arias")
        mlflow.set_tag("tipo", "parent")
        mlflow.log_params(params_comunes)

        # ----- Run anidado: Regresion -----
        with mlflow.start_run(run_name="regresion_punt_global", nested=True):
            mlflow.set_tag("modelo", "red_neuronal_regresion")
            mlflow.set_tag("target", TARGET_REG)
            mlflow.log_params(params_comunes)
            mlflow.log_param("epochs_max", EPOCHS_REG)
            mlflow.log_param("arquitectura", "Dense(128)->Dropout(0.2)->Dense(64)->Dropout(0.2)->Dense(32)->Dense(1,linear)")
            mlflow.log_param("loss", "mse")

            modelo_reg, met_reg, hist_reg = _entrenar_regresion(X_tr_reg, y_reg_tr, X_te_reg, y_reg_te)

            _log_historia(hist_reg, "train")
            mlflow.log_metrics(met_reg)
            mlflow.log_metric("epochs_efectivos", len(hist_reg.history["loss"]))
            mlflow.tensorflow.log_model(modelo_reg, name="modelo_regresion")

        # ----- Run anidado: Clasificacion -----
        with mlflow.start_run(run_name="clasificacion_prioridad", nested=True):
            mlflow.set_tag("modelo", "red_neuronal_clasificacion")
            mlflow.set_tag("target", "prioridad_alta_p25")
            mlflow.log_params(params_comunes)
            mlflow.log_param("epochs_max", EPOCHS_CLF)
            mlflow.log_param("arquitectura", "Dense(128)->Dropout(0.3)->Dense(64)->Dropout(0.3)->Dense(32)->Dense(1,sigmoid)")
            mlflow.log_param("loss", "binary_crossentropy")
            mlflow.log_param("clase_positiva_pct", round(float(y_clf_tr.mean()) * 100, 2))

            modelo_clf, met_clf, hist_clf = _entrenar_clasificacion(X_tr_clf, y_clf_tr, X_te_clf, y_clf_te)

            _log_historia(hist_clf, "train")
            mlflow.log_metrics(met_clf)
            mlflow.log_metric("epochs_efectivos", len(hist_clf.history["loss"]))
            mlflow.tensorflow.log_model(modelo_clf, name="modelo_clasificacion")

        # Metricas resumen en el run padre (para listado y comparacion rapida).
        mlflow.log_metrics({**met_reg, **met_clf})

        metadata = {
            "columnas_reg": list(X.columns),
            "columnas_clf": list(X.columns),
            "input_dim_reg": int(X_tr_reg.shape[1]),
            "input_dim_clf": int(X_tr_clf.shape[1]),
            "umbral_p25": umbral_p25,
            "metricas": {**met_reg, **met_clf},
            "entrenado_con_filas": int(len(X_tr)),
            "evaluado_con_filas": int(len(X_te)),
            "mlflow_run_id": run_padre.info.run_id,
            "mlflow_experiment": EXPERIMENT_NAME,
        }

        print("[5/6] Guardando artefactos en Models/...")
        modelo_reg.save_weights(os.path.join(MODELS_DIR, "regresion_puntaje.weights.h5"))
        modelo_clf.save_weights(os.path.join(MODELS_DIR, "clasificacion_prioridad.weights.h5"))
        joblib.dump(preproc_reg, os.path.join(MODELS_DIR, "preprocesador_reg.pkl"))
        joblib.dump(preproc_clf, os.path.join(MODELS_DIR, "preprocesador_clf.pkl"))
        joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata_modelos.pkl"))

        # Tambien subimos los preprocesadores y metadata al run padre como artefactos.
        for archivo in ("preprocesador_reg.pkl", "preprocesador_clf.pkl", "metadata_modelos.pkl"):
            mlflow.log_artifact(os.path.join(MODELS_DIR, archivo), artifact_path="preprocesadores")

        # Limpieza: el formato .keras antiguo es fragil entre versiones de Keras.
        for legacy in ("regresion_puntaje.keras", "clasificacion_prioridad.keras"):
            ruta = os.path.join(MODELS_DIR, legacy)
            if os.path.exists(ruta):
                os.remove(ruta)

    print(f"[6/6] Listo en {time.time() - t0:.1f}s. Archivos en {MODELS_DIR}/ y runs en {MLRUNS_DIR}/.")
    print(f"      Para inspeccionar:  mlflow ui --backend-store-uri {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
