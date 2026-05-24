import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# CARGA DE DATOS
def cargar_datos(path="data/saber11_Antioquia_clean.csv"):
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, "..", path)
    full_path = os.path.abspath(full_path)

    df = pd.read_csv(full_path)

    # Renombrar naturaleza del colegio para claridad
    df["cole_naturaleza"] = df["cole_naturaleza"].replace({
        "OFICIAL": "Público",
        "NO OFICIAL": "Privado"
    })

    # Merge con coordenadas geograficas para el mapa
    coord_path = os.path.join(base_path, "..", "data", "municipios_unicos.csv")
    coord_path = os.path.abspath(coord_path)
    df_coord = pd.read_csv(coord_path)
    df = pd.merge(df, df_coord, on="cole_mcpio_ubicacion", how="left")

    return df


# MATERIAS DISPONIBLES
MATERIAS = {
    "Inglés": "punt_ingles",
    "Matemáticas": "punt_matematicas",
    "Sociales y Ciudadanas": "punt_sociales_ciudadanas",
    "Ciencias Naturales": "punt_c_naturales",
    "Lectura Crítica": "punt_lectura_critica",
    "Puntaje Global": "punt_global"
}

ETIQUETAS_P2 = {
    "punt_global":              "Puntaje Global",
    "punt_ingles":              "Puntaje Inglés",
    "punt_matematicas":         "Puntaje Matemáticas",
    "punt_lectura_critica":     "Puntaje Lectura Crítica",
    "punt_c_naturales":         "Puntaje Ciencias Naturales",
    "punt_sociales_ciudadanas": "Puntaje Sociales y Ciudadanas",
    "cole_naturaleza":          "Tipo de colegio",
    "fami_estratovivienda":     "Estrato",
    "cole_area_ubicacion":      "Zona",
    "fami_educacionmadre":      "Educación madre",
    "fami_educacionpadre":      "Educación padre",
    "cole_jornada":             "Jornada",
    "fami_tieneinternet":       "Acceso a Internet",
    "fami_tienecomputador":     "Acceso a Computador",
    "cole_bilingue":            "Colegio bilingüe",
    "fami_personashogar":       "Personas en el hogar",
    "OFICIAL":   "Público",
    "NO OFICIAL": "Privado",
    "URBANO":    "Urbano",
    "RURAL":     "Rural",
    "MANANA":    "Mañana",
    "TARDE":     "Tarde",
    "NOCHE":     "Noche",
    "COMPLETA":  "Completa",
    "SABATINA":  "Sabatina",
    "UNICA":     "Única",
}


# CONSTANTES DE FORMATO
FONT_FAMILY = "Segoe UI, Arial, sans-serif"
COLOR_PUBLICO = "#1f77b4"
COLOR_PRIVADO = "#c0392b"
COLOR_GRID = "rgba(200,200,200,0.3)"
PLOT_BG = "white"


# LAYOUT BASE
# Aplica formato consistente a cualquier figura de plotly
def _layout_base(fig, title, subtitle=None, height=500,
                 xaxis_title=None, yaxis_title=None, showlegend=True):

    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=16, family=FONT_FAMILY, color="#222")
        ),
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor="white",
        font=dict(family=FONT_FAMILY, size=12, color="#333"),
        showlegend=showlegend,
        margin=dict(l=60, r=40, t=80, b=80),
    )

    if xaxis_title:
        fig.update_xaxes(
            title=dict(text=xaxis_title, font=dict(size=13)),
            showgrid=True, gridcolor=COLOR_GRID,
            tickfont=dict(size=11)
        )
    if yaxis_title:
        fig.update_yaxes(
            title=dict(text=yaxis_title, font=dict(size=13)),
            showgrid=True, gridcolor=COLOR_GRID,
            tickfont=dict(size=11)
        )

    # Leyenda horizontal debajo de la grafica
    if showlegend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.12,
                xanchor="center", x=0.5,
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1
            )
        )

    return fig


# FILTRAR DATOS
# Filtra por municipio y lista de periodos
def filtrar_datos(df, municipio="Todos", periodo=None):
    df = df.copy()
    if municipio != "Todos":
        df = df[df["cole_mcpio_ubicacion"] == municipio]
    if periodo:
        df = df[df["periodo"].isin(periodo)]
    return df


# CALCULAR BRECHAS
# Calcula la diferencia de medias entre privado y publico para cada materia
def calcular_brechas(df):
    brechas = {}
    for nombre, col in MATERIAS.items():
        df_temp = df[[col, "cole_naturaleza"]].dropna()
        public = df_temp[df_temp["cole_naturaleza"] == "Público"][col]
        private = df_temp[df_temp["cole_naturaleza"] == "Privado"][col]
        if len(public) > 0 and len(private) > 0:
            brecha = private.mean() - public.mean()
            media_pub = public.mean()
            media_priv = private.mean()
            brechas[nombre] = {
                "brecha": round(brecha, 1),
                "media_publico": round(media_pub, 1),
                "media_privado": round(media_priv, 1)
            }
        else:
            brechas[nombre] = {
                "brecha": None,
                "media_publico": None,
                "media_privado": None
            }
    return brechas


# BOXPLOTS POR MATERIA
# Grid de 2x3 boxplots comparando publico vs privado en cada materia
def generar_boxplots_materias(df):

    if df.empty:
        return go.Figure().update_layout(title="No hay datos disponibles")

    nombres = list(MATERIAS.keys())
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=nombres,
        vertical_spacing=0.14,
        horizontal_spacing=0.06
    )

    for i, (nombre, col) in enumerate(MATERIAS.items(), 1):
        row = 1 if i <= 3 else 2
        col_idx = i if i <= 3 else i - 3

        df_temp = df[[col, "cole_naturaleza"]].dropna()
        public = df_temp[df_temp["cole_naturaleza"] == "Público"][col]
        private = df_temp[df_temp["cole_naturaleza"] == "Privado"][col]

        # Solo mostrar leyenda en el primer subplot para no repetir
        fig.add_trace(
            go.Box(
                y=public,
                name="Público",
                marker_color=COLOR_PUBLICO,
                showlegend=(i == 1),
                legendgroup="Público"
            ),
            row=row, col=col_idx
        )

        fig.add_trace(
            go.Box(
                y=private,
                name="Privado",
                marker_color=COLOR_PRIVADO,
                showlegend=(i == 1),
                legendgroup="Privado"
            ),
            row=row, col=col_idx
        )

        fig.update_yaxes(title_text="Puntaje", row=row, col=col_idx,
                         showgrid=True, gridcolor=COLOR_GRID)

    _layout_base(
        fig,
        title="Distribución de puntajes por materia",
        subtitle="Comparación entre colegios públicos y privados",
        height=700,
        showlegend=True
    )

    # Formato de los subtitulos de cada subplot
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, family=FONT_FAMILY, color="#444")

    return fig


# FORMATO DE PERIODO
# Convierte 20151 a "2015-1" para mostrar en ejes y sliders
def formato_periodo(p):
    s = str(p)
    if len(s) >= 5:
        return f"{s[:4]}-{s[4:]}"
    return s


# MAPA DE BRECHA POR MUNICIPIO
# Scatter map donde cada punto es un municipio coloreado segun la brecha
def generar_mapa_brecha(df, columna_materia, municipio_seleccionado="Todos"):

    if df.empty or columna_materia not in df.columns:
        return go.Figure().update_layout(title="No hay datos disponibles")

    df_temp = df[["cole_mcpio_ubicacion", "lat", "lon", columna_materia, "cole_naturaleza"]].dropna()

    if df_temp.empty:
        return go.Figure().update_layout(title="No hay datos disponibles")

    # Promedios por municipio y tipo de colegio
    medias = df_temp.groupby(
        ["cole_mcpio_ubicacion", "lat", "lon", "cole_naturaleza"]
    )[columna_materia].mean().reset_index()

    # Pivotar para tener publico y privado en columnas separadas
    pivot = medias.pivot_table(
        index=["cole_mcpio_ubicacion", "lat", "lon"],
        columns="cole_naturaleza",
        values=columna_materia
    ).reset_index()

    if "Privado" in pivot.columns and "Público" in pivot.columns:
        pivot["brecha"] = (pivot["Privado"] - pivot["Público"]).round(1)
        pivot["media_pub"] = pivot["Público"].round(1)
        pivot["media_priv"] = pivot["Privado"].round(1)
        pivot = pivot.dropna(subset=["brecha"])
        color_col = "brecha"
        color_label = "Brecha (Priv - Púb)"
    else:
        # Si solo hay un tipo de colegio, mostrar promedio general
        agg = df_temp.groupby(
            ["cole_mcpio_ubicacion", "lat", "lon"]
        )[columna_materia].mean().reset_index()
        pivot = pivot.merge(agg, on=["cole_mcpio_ubicacion", "lat", "lon"], how="left")
        pivot.rename(columns={columna_materia: "promedio"}, inplace=True)
        color_col = "promedio"
        color_label = "Puntaje promedio"

    if pivot.empty:
        return go.Figure().update_layout(title="No hay datos suficientes")

    # Resaltar municipio seleccionado, atenuar el resto
    if municipio_seleccionado != "Todos":
        pivot["tamano"] = pivot["cole_mcpio_ubicacion"].apply(
            lambda x: 18 if x == municipio_seleccionado else 6
        )
        pivot["opacidad"] = pivot["cole_mcpio_ubicacion"].apply(
            lambda x: 1.0 if x == municipio_seleccionado else 0.15
        )
    else:
        pivot["tamano"] = 10
        pivot["opacidad"] = 0.85

    nombre_materia = [k for k, v in MATERIAS.items() if v == columna_materia]
    nombre_materia = nombre_materia[0] if nombre_materia else columna_materia

    # Escala simetrica centrada en 0
    max_abs = max(abs(pivot[color_col].min()), abs(pivot[color_col].max()))
    if max_abs == 0:
        max_abs = 1

    # Configurar datos del hover
    hover_data_dict = {
        "lat": False,
        "lon": False,
        "tamano": False,
        "opacidad": False,
        color_col: ":.1f",
    }
    if "media_pub" in pivot.columns:
        hover_data_dict["media_pub"] = ":.1f"
    if "media_priv" in pivot.columns:
        hover_data_dict["media_priv"] = ":.1f"

    fig = px.scatter_mapbox(
        pivot,
        lat="lat",
        lon="lon",
        color=color_col,
        hover_name="cole_mcpio_ubicacion",
        hover_data=hover_data_dict,
        color_continuous_scale=[
            [0, "#2166ac"],
            [0.5, "#f7f7f7"],
            [1, "#b2182b"]
        ],
        range_color=[-max_abs, max_abs],
        size="tamano",
        size_max=18,
        mapbox_style="carto-positron",
        zoom=6.0,
        center={"lat": 6.85, "lon": -75.56},
    )

    fig.update_traces(marker=dict(opacity=pivot["opacidad"].tolist()))

    filtro_label = municipio_seleccionado if municipio_seleccionado != "Todos" else "Todos los municipios"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Brecha educativa por municipio — {nombre_materia}</b>"
                f"<br><span style='font-size:12px;color:#666'>"
                f"Filtro: {filtro_label} | Azul: público supera | Rojo: privado supera</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=16, family=FONT_FAMILY, color="#222")
        ),
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        height=600,
        font=dict(family=FONT_FAMILY, size=12, color="#333"),
        coloraxis_colorbar=dict(
            title=dict(text=color_label, font=dict(size=12)),
            tickfont=dict(size=11)
        )
    )

    return fig


# BRECHA POR ESTRATO SOCIOECONOMICO
# Barras agrupadas publico vs privado por cada estrato
def generar_brecha_por_estrato(df, columna_materia):

    if df.empty or columna_materia not in df.columns:
        return go.Figure().update_layout(title="No hay datos disponibles")

    # Detectar la columna de estrato en el dataset
    col_estrato = None
    for c in ["fami_estratovivienda", "estu_estrato", "estrato"]:
        if c in df.columns:
            col_estrato = c
            break

    if col_estrato is None:
        return go.Figure().update_layout(
            title="No se encontró variable de estrato en los datos"
        )

    df_temp = df[[col_estrato, columna_materia, "cole_naturaleza"]].dropna()

    if df_temp.empty:
        return go.Figure().update_layout(title="No hay datos disponibles")

    df_temp[col_estrato] = df_temp[col_estrato].astype(str).str.strip()

    # Filtrar solo valores de estrato validos
    valores_validos = [
        "Estrato 1", "Estrato 2", "Estrato 3",
        "Estrato 4", "Estrato 5", "Estrato 6",
        "1", "2", "3", "4", "5", "6", "Sin Estrato"
    ]
    df_temp = df_temp[df_temp[col_estrato].isin(valores_validos)]

    if df_temp.empty:
        return go.Figure().update_layout(title="No hay datos de estrato válidos")

    # Normalizar nombres para que todos digan "Estrato X"
    mapeo_estrato = {
        "1": "Estrato 1", "2": "Estrato 2", "3": "Estrato 3",
        "4": "Estrato 4", "5": "Estrato 5", "6": "Estrato 6",
        "Estrato 1": "Estrato 1", "Estrato 2": "Estrato 2",
        "Estrato 3": "Estrato 3", "Estrato 4": "Estrato 4",
        "Estrato 5": "Estrato 5", "Estrato 6": "Estrato 6",
        "Sin Estrato": "Sin Estrato"
    }
    df_temp["estrato_clean"] = df_temp[col_estrato].map(mapeo_estrato)
    df_temp = df_temp.dropna(subset=["estrato_clean"])

    orden_estratos = [
        "Estrato 1", "Estrato 2", "Estrato 3",
        "Estrato 4", "Estrato 5", "Estrato 6", "Sin Estrato"
    ]

    # Promedios agrupados por estrato y tipo de colegio
    medias = df_temp.groupby(
        ["estrato_clean", "cole_naturaleza"]
    )[columna_materia].mean().reset_index()

    nombre_materia = [k for k, v in MATERIAS.items() if v == columna_materia]
    nombre_materia = nombre_materia[0] if nombre_materia else columna_materia

    fig = go.Figure()

    # Barras de colegios publicos
    pub = medias[medias["cole_naturaleza"] == "Público"].copy()
    pub["estrato_clean"] = pd.Categorical(
        pub["estrato_clean"], categories=orden_estratos, ordered=True
    )
    pub = pub.sort_values("estrato_clean")

    fig.add_trace(go.Bar(
        x=pub["estrato_clean"],
        y=pub[columna_materia],
        name="Público",
        marker_color=COLOR_PUBLICO,
        text=[f"{v:.1f}" for v in pub[columna_materia]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x} — Público<br>Puntaje: %{y:.1f}<extra></extra>"
    ))

    # Barras de colegios privados
    priv = medias[medias["cole_naturaleza"] == "Privado"].copy()
    priv["estrato_clean"] = pd.Categorical(
        priv["estrato_clean"], categories=orden_estratos, ordered=True
    )
    priv = priv.sort_values("estrato_clean")

    fig.add_trace(go.Bar(
        x=priv["estrato_clean"],
        y=priv[columna_materia],
        name="Privado",
        marker_color=COLOR_PRIVADO,
        text=[f"{v:.1f}" for v in priv[columna_materia]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x} — Privado<br>Puntaje: %{y:.1f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            categoryorder="array",
            categoryarray=orden_estratos,
        ),
    )

    _layout_base(
        fig,
        title=f"Puntaje promedio por estrato — {nombre_materia}",
        subtitle="Comparación entre colegios públicos y privados por nivel socioeconómico",
        height=450,
        xaxis_title="Estrato socioeconómico",
        yaxis_title="Puntaje promedio",
        showlegend=True
    )

    # Bajar la leyenda un poco mas para que no tape las etiquetas
    fig.update_layout(
        legend=dict(y=-0.2)
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PRUEBAS ESTADÍSTICAS P2
# ─────────────────────────────────────────────────────────────────────────────

_STATS_DIR = Path(__file__).resolve().parent / "resultados_estadisticos_p2"


def cargar_resultados_estadisticos_p2():
    """Carga los CSV de pruebas estadísticas ya calculados por pruebas_estadisticas_p2.py."""
    resultado = {"disponible": False}
    try:
        mat_path = _STATS_DIR / "brecha_por_materia.csv"
        est_path = _STATS_DIR / "brecha_por_estrato.csv"
        if mat_path.exists():
            resultado["brecha_materia"] = pd.read_csv(str(mat_path))
        if est_path.exists():
            resultado["brecha_estrato"] = pd.read_csv(str(est_path))
        resultado["disponible"] = mat_path.exists() or est_path.exists()
    except Exception as exc:
        resultado["error"] = str(exc)
    return resultado


def generar_figura_brecha_materias_stat():
    """Gráfica de barras con la brecha (privado - público) por materia desde las pruebas t."""
    res = cargar_resultados_estadisticos_p2()
    if not res.get("disponible") or "brecha_materia" not in res:
        return go.Figure().update_layout(title="Sin datos estadísticos (ejecutar pruebas_estadisticas_p2.py)")

    df = res["brecha_materia"].sort_values("Brecha", ascending=True)
    colores = ["#c0392b" if b > 0 else "#2166ac" for b in df["Brecha"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Brecha"],
        y=df["Materia"],
        orientation="h",
        marker_color=colores,
        text=[f"{b:+.2f} pts  (d={d:.2f}, {e})"
              for b, d, e in zip(df["Brecha"], df["Cohen_d"].abs(), df["Efecto"])],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_color="#888", line_dash="dash")
    _layout_base(
        fig,
        title="Brecha en puntajes por área",
        subtitle="Diferencia privado − público · positivo = privados superan · negativo = públicos superan",
        height=360,
        xaxis_title="Diferencia de medias (puntos)",
        showlegend=False,
    )
    fig.update_layout(margin=dict(l=190, r=160, t=80, b=40))
    return fig


def generar_figura_brecha_estratos_stat():
    """Gráfica que muestra cómo la brecha público-privado varía por estrato."""
    res = cargar_resultados_estadisticos_p2()
    if not res.get("disponible") or "brecha_estrato" not in res:
        return go.Figure().update_layout(title="Sin datos estadísticos")

    df = res["brecha_estrato"]
    colores = ["#2166ac" if b < 0 else "#c0392b" for b in df["Brecha"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Estrato"],
        y=df["Brecha"],
        marker_color=colores,
        text=[f"{b:+.1f}" for b in df["Brecha"]],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
    ))
    fig.add_hline(y=0, line_color="#888", line_dash="dash")
    _layout_base(
        fig,
        title="Brecha educativa por estrato socioeconómico",
        subtitle="Diferencia privado − público · azul = públicos superan · rojo = privados superan",
        height=400,
        xaxis_title="Estrato socioeconómico",
        yaxis_title="Diferencia de medias (puntos)",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS PREDICTIVOS P2
# ─────────────────────────────────────────────────────────────────────────────

_MODELS_P2_DIR = Path(__file__).resolve().parent.parent / "models" / "pregunta_2"
_MLRUNS_P2_DIR = Path(__file__).resolve().parent.parent / "mlruns" / "pregunta_2"

FEATURES_P2 = [
    "cole_naturaleza",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "cole_area_ubicacion",
    "cole_jornada",
]

# Targets de regresión disponibles: slug → etiqueta de display
TARGETS_LABELS_P2 = {
    "global":              "Puntaje Global",
    "matematicas":         "Matemáticas",
    "ingles":              "Inglés",
    "lectura_critica":     "Lectura Crítica",
    "c_naturales":         "Ciencias Naturales",
    "sociales_ciudadanas": "Sociales y Ciudadanas",
}

OPCIONES_P2 = {
    "cole_naturaleza": ["Público", "Privado"],
    "fami_estratovivienda": [
        "Estrato 1", "Estrato 2", "Estrato 3",
        "Estrato 4", "Estrato 5", "Estrato 6", "Sin Estrato",
    ],
    "fami_educacionmadre": [
        "Ninguno", "Primaria incompleta", "Primaria completa",
        "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
        "Tecnica o tecnologica incompleta", "Tecnica o tecnologica completa",
        "Educacion profesional incompleta", "Educacion profesional completa",
        "Postgrado", "No sabe", "No Aplica",
    ],
    "fami_educacionpadre": [
        "Ninguno", "Primaria incompleta", "Primaria completa",
        "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
        "Tecnica o tecnologica incompleta", "Tecnica o tecnologica completa",
        "Educacion profesional incompleta", "Educacion profesional completa",
        "Postgrado", "No sabe", "No Aplica",
    ],
    "cole_area_ubicacion": ["RURAL", "URBANO"],
    "cole_jornada": ["MANANA", "TARDE", "NOCHE", "COMPLETA", "SABATINA", "UNICA"],
}


def obtener_mlflow_info_p2():
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    tracking_uri = env_uri if env_uri else _MLRUNS_P2_DIR.as_uri()
    return {
        "tracking_uri": tracking_uri,
        "ui_url": os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000"),
    }


def cargar_resultados_mlflow_p2(max_runs=50):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        return pd.DataFrame(), f"MLflow no disponible: {exc}"

    if not _MLRUNS_P2_DIR.exists():
        return pd.DataFrame(), "No hay experimentos de P2. Ejecuta Analysis/entrenar_modelos_p2.py primero."

    mlflow.set_tracking_uri(_MLRUNS_P2_DIR.as_uri())
    client = MlflowClient()

    rows = []
    for exp_name in ["P2_publico_privado_regresion", "P2_publico_privado_clasificacion"]:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            continue
        for run in client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=max_runs,
        ):
            m = run.data.metrics
            p = run.data.params
            rows.append({
                "task": run.data.tags.get("task", ""),
                "run_name": run.info.run_name,
                "layers": p.get("layers", ""),
                "dropout": p.get("dropout", ""),
                "learning_rate": p.get("learning_rate", ""),
                "epochs_run": p.get("epochs_run", ""),
                "batch_size": p.get("batch_size", ""),
                "rmse": round(m["rmse"], 3) if "rmse" in m else None,
                "mae": round(m["mae"], 3) if "mae" in m else None,
                "r2": round(m["r2"], 4) if "r2" in m else None,
                "accuracy": round(m["accuracy"], 4) if "accuracy" in m else None,
                "precision": round(m["precision"], 4) if "precision" in m else None,
                "recall": round(m["recall"], 4) if "recall" in m else None,
                "f1": round(m["f1"], 4) if "f1" in m else None,
                "run_id": run.info.run_id,
            })

    if not rows:
        return pd.DataFrame(), "No se encontraron corridas en mlruns/pregunta_2."

    return pd.DataFrame(rows), f"{len(rows)} corridas cargadas."


def construir_figuras_mlflow_p2(df_runs):
    if df_runs.empty:
        return go.Figure(), go.Figure()

    reg = df_runs[df_runs["task"] == "regresion"].copy()
    clf = df_runs[df_runs["task"] == "clasificacion_binaria"].copy()

    fig_reg = go.Figure()
    if not reg.empty:
        for col, color, name in [("rmse", "#e74c3c", "RMSE"), ("mae", "#3498db", "MAE")]:
            fig_reg.add_trace(go.Bar(
                x=reg["run_name"], y=reg[col].fillna(0),
                name=name, marker_color=color,
                text=[f"{v:.3f}" if v else "" for v in reg[col]],
                textposition="outside",
                textfont=dict(size=10),
                cliponaxis=False,
            ))
        _layout_base(
            fig_reg,
            title="Regresión — error por configuración",
            subtitle="RMSE y MAE de los modelos entrenados (menor es mejor)",
            height=380,
            yaxis_title="Error (puntos)",
            showlegend=True,
        )
        fig_reg.update_layout(
            barmode="group",
            margin=dict(l=70, r=40, t=90, b=120),
        )
        fig_reg.update_xaxes(tickangle=-40, tickfont=dict(size=10))

    fig_clf = go.Figure()
    if not clf.empty:
        for col, color, name in [
            ("accuracy", "#2ecc71", "Accuracy"),
            ("precision", "#3498db", "Precisión"),
            ("recall", "#e67e22", "Recall"),
            ("f1", "#9b59b6", "F1-Score"),
        ]:
            fig_clf.add_trace(go.Bar(
                x=clf["run_name"], y=clf[col].fillna(0),
                name=name, marker_color=color,
                text=[f"{v:.3f}" if v else "" for v in clf[col]],
                textposition="outside",
                textfont=dict(size=10),
                cliponaxis=False,
            ))
        _layout_base(
            fig_clf,
            title="Clasificación binaria — métricas por configuración",
            subtitle="Accuracy, Precisión, Recall y F1-Score (mayor es mejor)",
            height=380,
            showlegend=True,
        )
        fig_clf.update_layout(
            barmode="group",
            yaxis=dict(title="Puntuación", range=[0, 1.25]),
            margin=dict(l=70, r=40, t=90, b=120),
        )
        fig_clf.update_xaxes(tickangle=-40, tickfont=dict(size=10))

    return fig_reg, fig_clf


_MODELOS_CACHE = {}


def _cargar_bundle(model_dir: Path):
    """Carga model.keras + preprocessor.pkl + metadata.json desde model_dir."""
    try:
        import tensorflow as tf
        import joblib
    except Exception as exc:
        return None, str(exc)
    try:
        bundle = {
            "model":       tf.keras.models.load_model(str(model_dir / "model.keras")),
            "preprocessor": joblib.load(str(model_dir / "preprocessor.pkl")),
            "metadata":    json.load(open(str(model_dir / "metadata.json"), encoding="utf-8")),
        }
        return bundle, None
    except Exception as exc:
        return None, str(exc)


def cargar_modelos_p2():
    """Carga (con caché de proceso) los 6 modelos de regresión y el de clasificación."""
    global _MODELOS_CACHE
    if _MODELOS_CACHE:
        return _MODELOS_CACHE

    artefactos = {"disponible": True, "error": None, "regresion": {}}

    # 6 modelos de regresión
    for slug in TARGETS_LABELS_P2:
        model_dir = _MODELS_P2_DIR / "regresion" / slug / "best"
        if not (model_dir / "model.keras").exists():
            continue
        bundle, err = _cargar_bundle(model_dir)
        if bundle:
            artefactos["regresion"][slug] = bundle
        elif not artefactos["error"]:
            artefactos["error"] = err

    # Modelo de clasificación
    clf_dir = _MODELS_P2_DIR / "clasificacion_binaria" / "best"
    if (clf_dir / "model.keras").exists():
        bundle, err = _cargar_bundle(clf_dir)
        artefactos["clasificacion_binaria"] = bundle
        if err and not artefactos["error"]:
            artefactos["error"] = err
    else:
        artefactos["clasificacion_binaria"] = None

    if not artefactos["regresion"] and artefactos["clasificacion_binaria"] is None:
        artefactos["disponible"] = False

    _MODELOS_CACHE = artefactos
    return artefactos


def predecir_escenarios_p2(valores_a, valores_b, target_slug: str = "global"):
    artefactos = cargar_modelos_p2()
    if not artefactos.get("disponible"):
        return None, None, go.Figure(), go.Figure(), artefactos.get("error", "Modelos no disponibles")

    reg_dict = artefactos.get("regresion", {})
    clf     = artefactos.get("clasificacion_binaria")

    reg = reg_dict.get(target_slug)
    if reg is None and reg_dict:
        reg = next(iter(reg_dict.values()))  # fallback al primer disponible

    if reg is None or clf is None:
        return None, None, go.Figure(), go.Figure(), (
            "No se encontraron los modelos en models/pregunta_2/. "
            "Ejecuta Analysis/entrenar_modelos_p2.py primero."
        )

    target_label = TARGETS_LABELS_P2.get(target_slug, target_slug)

    def _to_dense(m):
        return m.toarray() if hasattr(m, "toarray") else m

    def _make_row(bundle, valores):
        # Builds DataFrame with the exact columns the model was trained on.
        # Features not in `valores` get NaN → SimpleImputer fills with training mode.
        cols = bundle["metadata"]["feature_columns"]
        return pd.DataFrame([{col: valores.get(col, np.nan) for col in cols}], columns=cols)

    def _predecir_una(valores):
        X_reg = _to_dense(reg["preprocessor"].transform(_make_row(reg, valores)))
        puntaje = float(reg["model"].predict(X_reg, verbose=0).flatten()[0])
        puntaje = max(0.0, min(500.0, puntaje))

        X_clf = _to_dense(clf["preprocessor"].transform(_make_row(clf, valores)))
        proba_bajo = float(clf["model"].predict(X_clf, verbose=0).flatten()[0])
        proba_bajo = max(0.0, min(1.0, proba_bajo))

        return {"puntaje": round(puntaje, 1), "proba_bajo": round(proba_bajo, 4)}

    try:
        pred_a = _predecir_una(valores_a)
        pred_b = _predecir_una(valores_b)
    except Exception as exc:
        return None, None, go.Figure(), go.Figure(), f"Error al predecir: {exc}"

    labels = ["Escenario A", "Escenario B"]
    puntajes = [pred_a["puntaje"], pred_b["puntaje"]]
    colores_reg = ["#1f77b4" if p >= 250 else "#c0392b" for p in puntajes]

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Bar(
        x=labels, y=puntajes, marker_color=colores_reg,
        text=[f"{p:.1f} pts" for p in puntajes], textposition="outside",
        textfont=dict(size=13, color="#222"),
        width=0.4,
        cliponaxis=False,
    ))
    fig_reg.add_hline(y=250, line_dash="dash", line_color="#888",
                      annotation_text="Umbral mínimo (250 pts)",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#666"))
    _layout_base(
        fig_reg,
        title=f"Puntaje predicho — {target_label}",
        subtitle="Comparación entre escenarios A y B",
        height=380,
        yaxis_title=target_label,
        showlegend=False,
    )
    fig_reg.update_layout(
        yaxis=dict(range=[0, 560 if target_slug == "global" else 120]),
        margin=dict(l=70, r=40, t=90, b=60),
    )

    probas_pct = [pred_a["proba_bajo"] * 100, pred_b["proba_bajo"] * 100]
    colores_clf = ["#c0392b" if p > 50 else "#27ae60" for p in probas_pct]

    fig_clf = go.Figure()
    fig_clf.add_trace(go.Bar(
        x=labels, y=probas_pct, marker_color=colores_clf,
        text=[f"{p:.1f}%" for p in probas_pct], textposition="outside",
        textfont=dict(size=13, color="#222"),
        width=0.4,
        cliponaxis=False,
    ))
    fig_clf.add_hline(y=50, line_dash="dash", line_color="#888",
                      annotation_text="Umbral 50%",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#666"))
    _layout_base(
        fig_clf,
        title="Riesgo de bajo rendimiento (Puntaje Global < 250)",
        subtitle="Probabilidad estimada de quedar bajo el umbral departamental",
        height=380,
        yaxis_title="Probabilidad (%)",
        showlegend=False,
    )
    fig_clf.update_layout(
        yaxis=dict(range=[0, 120]),
        margin=dict(l=70, r=40, t=90, b=60),
    )

    return pred_a, pred_b, fig_reg, fig_clf, ""


# ─────────────────────────────────────────────────────────────────────────────
# BRECHA AJUSTADA POR MATERIA (desde CSV precalculado)
# ─────────────────────────────────────────────────────────────────────────────

_BRECHA_AJUSTADA_PATH = _STATS_DIR / "brecha_ajustada_por_materia.csv"


def cargar_brecha_ajustada_p2():
    if not _BRECHA_AJUSTADA_PATH.exists():
        return None
    try:
        return pd.read_csv(str(_BRECHA_AJUSTADA_PATH))
    except Exception:
        return None


def generar_figura_brecha_ajustada_p2():
    """Heatmap-style bar: brecha ajustada promedio por target (promedio sobre estratos)."""
    df = cargar_brecha_ajustada_p2()
    if df is None:
        return go.Figure().update_layout(title="Sin datos (ejecutar entrenar_modelos_p2.py)")

    resumen = (df.groupby("slug")["brecha_ajustada"]
               .mean()
               .reset_index()
               .rename(columns={"brecha_ajustada": "brecha_media"}))

    # Orden por magnitud absoluta descendente
    resumen["label"] = resumen["slug"].map(TARGETS_LABELS_P2)
    resumen = resumen.sort_values("brecha_media", ascending=True)
    colores = ["#c0392b" if b > 0 else "#2166ac" for b in resumen["brecha_media"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resumen["brecha_media"],
        y=resumen["label"],
        orientation="h",
        marker_color=colores,
        text=[f"{b:+.2f} pts" for b in resumen["brecha_media"]],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_color="#888", line_dash="dash")
    _layout_base(
        fig,
        title="Brecha ajustada por materia — efecto neto del tipo de colegio",
        subtitle="Predicción con perfil base fijo · controla por estrato y educación familiar",
        height=360,
        xaxis_title="Diferencia predicha en puntos (privado − público)",
        showlegend=False,
    )
    fig.update_layout(margin=dict(l=185, r=140, t=80, b=40))
    return fig