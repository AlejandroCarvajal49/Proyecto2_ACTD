"""
Entrena los modelos de regresion y clasificacion binaria para P2:
"Brecha educativa publica vs privada en Antioquia, controlando por estrato".

Dos modelos:
  - Regresion: predice punt_global (0-500)
  - Clasificacion binaria: predice bajo_rendimiento (1 si punt_global < 250)

Ejecutar desde la raiz del proyecto:
    python Analysis/entrenar_modelos_p2.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# PATHS Y CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "saber11_Antioquia_clean.csv"
MODELS_DIR = BASE_DIR / "models" / "pregunta_2"
MLRUNS_DIR = BASE_DIR / "mlruns" / "pregunta_2"
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

FEATURES = [
    "cole_naturaleza",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "cole_area_ubicacion",
    "cole_jornada",
]

# Features adicionales para el experimento de mejora
# fami_tieneinternet y fami_tienecomputador capturan acceso TIC del hogar
# cole_bilingue refleja oferta educativa diferenciada
# fami_personashogar es proxy de condiciones del hogar
FEATURES_AMPLIADO = FEATURES + [
    "fami_tieneinternet",
    "fami_tienecomputador",
    "cole_bilingue",
    "fami_personashogar",
]

TARGET_REG = "punt_global"
UMBRAL_CLASIF = 250
RANDOM_STATE = 42

# Tres configuraciones por tarea — variando arquitectura, dropout y lr
CONFIGS_REG = [
    {
        "name": "mlp_64_32",
        "layers": [64, 32],
        "activation": "relu",
        "dropout": 0.1,
        "l2": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "mse",
        "epochs": 60,
        "batch_size": 256,
    },
    {
        "name": "mlp_128_64_32",
        "layers": [128, 64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "l2": 0.0001,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "mse",
        "epochs": 60,
        "batch_size": 256,
    },
    {
        "name": "mlp_32_16",
        "layers": [32, 16],
        "activation": "relu",
        "dropout": 0.0,
        "l2": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.005,
        "loss": "mse",
        "epochs": 50,
        "batch_size": 512,
    },
]

CONFIGS_CLASIF = [
    {
        "name": "clf_64_32",
        "layers": [64, 32],
        "activation": "relu",
        "dropout": 0.1,
        "l2": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 60,
        "batch_size": 256,
    },
    {
        "name": "clf_128_64_32",
        "layers": [128, 64, 32],
        "activation": "relu",
        "dropout": 0.2,
        "l2": 0.0001,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 60,
        "batch_size": 256,
    },
    {
        "name": "clf_32_16",
        "layers": [32, 16],
        "activation": "relu",
        "dropout": 0.0,
        "l2": 0.0,
        "optimizer": "adam",
        "learning_rate": 0.005,
        "epochs": 50,
        "batch_size": 512,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(m):
    return m.toarray() if hasattr(m, "toarray") else m


def _build_preprocessor(features=None):
    if features is None:
        features = FEATURES
    return ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), features),
    ])


def _crear_optimizer(config):
    lr = float(config.get("learning_rate", 0.001))
    opt = str(config.get("optimizer", "adam")).lower()
    if opt == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if opt == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _crear_mlp(input_dim, config, output_dim, output_activation, loss, metrics=None):
    regularizer = None
    if float(config.get("l2", 0.0)) > 0:
        regularizer = tf.keras.regularizers.l2(float(config["l2"]))

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for units in config["layers"]:
        model.add(
            tf.keras.layers.Dense(
                int(units),
                activation=config.get("activation", "relu"),
                kernel_regularizer=regularizer,
            )
        )
        if float(config.get("dropout", 0.0)) > 0:
            model.add(tf.keras.layers.Dropout(float(config["dropout"])))
    model.add(tf.keras.layers.Dense(output_dim, activation=output_activation))
    model.compile(
        optimizer=_crear_optimizer(config),
        loss=loss,
        metrics=metrics or [],
    )
    return model


def _serializable_params(config):
    params = {}
    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            params[k] = "-".join(str(x) for x in v)
        else:
            params[k] = v
    return params


def _guardar_modelo(task, model, preprocessor, config, metrics, feature_columns):
    task_dir = MODELS_DIR / task / "best"
    task_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(task_dir / "model.keras"))
    joblib.dump(preprocessor, str(task_dir / "preprocessor.pkl"))

    metadata = {
        "task": task,
        "config": _serializable_params(config),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "label_mapping": None,
        "trained_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(task_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)

    print(f"  Artefactos guardados en {task_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar():
    print(f"Cargando datos desde {DATA_PATH} ...")
    df = pd.read_csv(str(DATA_PATH), dtype={"cole_cod_mcpio_ubicacion": str})

    df_model = df[FEATURES + [TARGET_REG]].dropna(subset=[TARGET_REG]).copy()
    df_model["bajo_rendimiento"] = (df_model[TARGET_REG] < UMBRAL_CLASIF).astype(int)

    n = len(df_model)
    balance = df_model["bajo_rendimiento"].value_counts().to_dict()
    print(f"  Filas usables: {n:,}")
    print(f"  bajo_rendimiento={balance}  (umbral={UMBRAL_CLASIF})")
    return df_model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — REGRESIÓN
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_regresion(df_model):
    print("\n" + "=" * 60)
    print("REGRESION: predice punt_global")
    print("=" * 60)

    X = df_model[FEATURES]
    y = df_model[TARGET_REG].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = _build_preprocessor()
    X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
    X_test_proc = _to_dense(preprocessor.transform(X_test))
    input_dim = X_train_proc.shape[1]
    print(f"  input_dim={input_dim} | train={len(X_train):,} | test={len(X_test):,}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("P2_publico_privado_regresion")

    best_rmse = float("inf")
    best_config = None
    best_model_ref = [None]

    for config in CONFIGS_REG:
        run_name = f"p2_reg_{config['name']}"
        print(f"\n  [{run_name}]")
        with mlflow.start_run(run_name=run_name):
            model = _crear_mlp(input_dim, config, 1, "linear", "mse")

            history = model.fit(
                X_train_proc,
                y_train,
                validation_split=0.2,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True
                    )
                ],
            )

            preds = model.predict(X_test_proc, verbose=0).flatten()
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            epochs_run = len(history.history["loss"])

            mlflow.set_tag("task", "regresion")
            mlflow.set_tag("pregunta", "P2_publico_privado")
            for k, v in _serializable_params(config).items():
                mlflow.log_param(k, v)
            mlflow.log_param("input_dim", input_dim)
            mlflow.log_param("epochs_run", epochs_run)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            with tempfile.TemporaryDirectory() as tmp:
                hist_path = os.path.join(tmp, "history.json")
                with open(hist_path, "w") as fh:
                    json.dump(
                        {k: [float(x) for x in v] for k, v in history.history.items()},
                        fh,
                    )
                mlflow.log_artifact(hist_path, "history")

                model_path = os.path.join(tmp, "model.keras")
                model.save(model_path)
                mlflow.log_artifact(model_path, "model")

            print(f"    rmse={rmse:.3f}  mae={mae:.3f}  r2={r2:.4f}  epochs={epochs_run}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config
                best_model_ref[0] = model

    best_model = best_model_ref[0]
    preds_best = best_model.predict(X_test_proc, verbose=0).flatten()
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds_best))),
        "mae": float(mean_absolute_error(y_test, preds_best)),
        "r2": float(r2_score(y_test, preds_best)),
    }

    print(f"\n  >> Mejor: {best_config['name']}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}")
    _guardar_modelo("regresion", best_model, preprocessor, best_config, metrics, FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — CLASIFICACIÓN BINARIA
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_clasificacion(df_model):
    print("\n" + "=" * 60)
    print(f"CLASIFICACION BINARIA: bajo_rendimiento (punt_global < {UMBRAL_CLASIF})")
    print("=" * 60)

    X = df_model[FEATURES]
    y = df_model["bajo_rendimiento"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _build_preprocessor()
    X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
    X_test_proc = _to_dense(preprocessor.transform(X_test))
    input_dim = X_train_proc.shape[1]
    print(f"  input_dim={input_dim} | train={len(X_train):,} | test={len(X_test):,}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("P2_publico_privado_clasificacion")

    best_f1 = -1.0
    best_config = None
    best_model_ref = [None]

    for config in CONFIGS_CLASIF:
        run_name = f"p2_clf_{config['name']}"
        print(f"\n  [{run_name}]")
        with mlflow.start_run(run_name=run_name):
            model = _crear_mlp(
                input_dim,
                config,
                1,
                "sigmoid",
                "binary_crossentropy",
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
                        patience=5, restore_best_weights=True
                    )
                ],
            )

            probas = model.predict(X_test_proc, verbose=0).flatten()
            preds = (probas >= 0.5).astype(int)
            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds, zero_division=0))
            rec = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            epochs_run = len(history.history["loss"])

            mlflow.set_tag("task", "clasificacion_binaria")
            mlflow.set_tag("pregunta", "P2_publico_privado")
            for k, v in _serializable_params(config).items():
                mlflow.log_param(k, v)
            mlflow.log_param("input_dim", input_dim)
            mlflow.log_param("epochs_run", epochs_run)
            mlflow.log_param("umbral_clasif", UMBRAL_CLASIF)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            with tempfile.TemporaryDirectory() as tmp:
                hist_path = os.path.join(tmp, "history.json")
                with open(hist_path, "w") as fh:
                    json.dump(
                        {k: [float(x) for x in v] for k, v in history.history.items()},
                        fh,
                    )
                mlflow.log_artifact(hist_path, "history")

                model_path = os.path.join(tmp, "model.keras")
                model.save(model_path)
                mlflow.log_artifact(model_path, "model")

            print(
                f"    acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  epochs={epochs_run}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_config = config
                best_model_ref[0] = model

    best_model = best_model_ref[0]
    probas_best = best_model.predict(X_test_proc, verbose=0).flatten()
    preds_best = (probas_best >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds_best)),
        "precision": float(precision_score(y_test, preds_best, zero_division=0)),
        "recall": float(recall_score(y_test, preds_best, zero_division=0)),
        "f1": float(f1_score(y_test, preds_best, zero_division=0)),
    }

    print(f"\n  >> Mejor: {best_config['name']}  F1={metrics['f1']:.4f}  acc={metrics['accuracy']:.4f}")
    _guardar_modelo(
        "clasificacion_binaria", best_model, preprocessor, best_config, metrics, FEATURES
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO CON FEATURES AMPLIADO — comparación vs baseline
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_regresion_ampliado(df_model, r2_baseline: float):
    """Entrena regresión con features adicionales. Actualiza best/ solo si mejora R²."""
    print("\n" + "=" * 60)
    print("REGRESION AMPLIADA: + internet, computador, bilingue, personashogar")
    print(f"  R² baseline (features originales) = {r2_baseline:.4f}")
    print("=" * 60)

    # Necesitamos las columnas extra que no están en df_model → recargamos
    df_full = pd.read_csv(str(DATA_PATH), dtype={"cole_cod_mcpio_ubicacion": str})
    df_full = df_full[FEATURES_AMPLIADO + [TARGET_REG]].dropna(subset=[TARGET_REG])

    X = df_full[FEATURES_AMPLIADO]
    y = df_full[TARGET_REG].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = _build_preprocessor(FEATURES_AMPLIADO)
    X_train_proc = _to_dense(preprocessor.fit_transform(X_train))
    X_test_proc  = _to_dense(preprocessor.transform(X_test))
    input_dim = X_train_proc.shape[1]
    print(f"  input_dim={input_dim} | train={len(X_train):,} | test={len(X_test):,}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("P2_publico_privado_regresion")

    configs_amp = [
        {
            "name": "mlp_amp_64_32",
            "layers": [64, 32],
            "activation": "relu",
            "dropout": 0.1,
            "l2": 0.0,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "mse",
            "epochs": 60,
            "batch_size": 256,
        },
        {
            "name": "mlp_amp_128_64_32",
            "layers": [128, 64, 32],
            "activation": "relu",
            "dropout": 0.2,
            "l2": 0.0001,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "mse",
            "epochs": 60,
            "batch_size": 256,
        },
    ]

    best_r2  = -float("inf")
    best_config = None
    best_model_ref = [None]

    for config in configs_amp:
        run_name = f"p2_reg_{config['name']}"
        print(f"\n  [{run_name}]")
        with mlflow.start_run(run_name=run_name):
            model = _crear_mlp(input_dim, config, 1, "linear", "mse")
            history = model.fit(
                X_train_proc, y_train,
                validation_split=0.2,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            )
            preds = model.predict(X_test_proc, verbose=0).flatten()
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae  = float(mean_absolute_error(y_test, preds))
            r2   = float(r2_score(y_test, preds))
            epochs_run = len(history.history["loss"])

            mlflow.set_tag("task",     "regresion")
            mlflow.set_tag("pregunta", "P2_publico_privado")
            mlflow.set_tag("feature_set", "ampliado")
            for k, v in _serializable_params(config).items():
                mlflow.log_param(k, v)
            mlflow.log_param("input_dim",  input_dim)
            mlflow.log_param("epochs_run", epochs_run)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("r2",   r2)

            with tempfile.TemporaryDirectory() as tmp:
                hist_path = os.path.join(tmp, "history.json")
                with open(hist_path, "w") as fh:
                    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, fh)
                mlflow.log_artifact(hist_path, "history")
                model_path = os.path.join(tmp, "model.keras")
                model.save(model_path)
                mlflow.log_artifact(model_path, "model")

            print(f"    rmse={rmse:.3f}  mae={mae:.3f}  r2={r2:.4f}  epochs={epochs_run}")

            if r2 > best_r2:
                best_r2 = r2
                best_config = config
                best_model_ref[0] = model

    mejora = best_r2 - r2_baseline
    print(f"\n  >> Mejor ampliado: {best_config['name']}  R²={best_r2:.4f}")
    print(f"  >> Mejora vs baseline: {mejora:+.4f}")

    if mejora > 0.005:
        preds_best = best_model_ref[0].predict(X_test_proc, verbose=0).flatten()
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds_best))),
            "mae":  float(mean_absolute_error(y_test, preds_best)),
            "r2":   float(r2_score(y_test, preds_best)),
        }
        _guardar_modelo("regresion", best_model_ref[0], preprocessor,
                        best_config, metrics, FEATURES_AMPLIADO)
        print(f"  >> Modelo best/ ACTUALIZADO (mejora > 0.005)")
    else:
        print(f"  >> Modelo best/ NO actualizado — mejora ({mejora:+.4f}) insuficiente.")
        print(f"     Se conserva el modelo original (R²={r2_baseline:.4f}).")

    return best_r2, mejora


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    import json as _json
    with open(MODELS_DIR / "regresion" / "best" / "metadata.json") as f:
        r2_baseline = _json.load(f)["metrics"]["r2"]

    df_model = cargar_y_preparar()
    entrenar_regresion_ampliado(df_model, r2_baseline)
    print("\n" + "=" * 60)
    print("Entrenamiento completo.")
    print(f"Modelos en: {MODELS_DIR}")
    print(f"MLflow runs en: {MLRUNS_DIR}")
    print("=" * 60)
