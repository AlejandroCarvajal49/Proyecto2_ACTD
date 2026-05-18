"""
Lógica de la Pregunta 1 - Equidad regional urbano/rural.

Responsable: Santiago Arias.

Este archivo contiene:
    - PARTE 1 (Proyecto 1): carga de datos, gráficas descriptivas y prueba t.
    - PARTE 2 (Proyecto 2): carga de modelos pre-entrenados de redes neuronales
      (regresión + clasificación) e inferencia para el simulador del tablero.
"""

import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
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
# PARTE 2 - PROYECTO 2: MODELOS PREDICTIVOS (NUEVO)
# ===========================================================================
# ===========================================================================
#
# Esta sección agrega los modelos de redes neuronales para responder
# operativamente la pregunta de negocio:
#   "¿Existe una brecha significativa entre rural y urbano que justifique
#    intervención diferenciada en municipios con menor PIB?"
#
# Modelos esperados en la carpeta `Models/`:
#   - regresion_puntaje.keras       (red de regresión)
#   - clasificacion_prioridad.keras (red de clasificación binaria)
#   - preprocesador_reg.pkl         (ColumnTransformer de regresión)
#   - preprocesador_clf.pkl         (ColumnTransformer de clasificación)
#   - metadata_modelos.pkl          (umbral P25, columnas, métricas)
# ===========================================================================

MODELS_DIR = "Models"

# Variables que el usuario podrá ingresar en el simulador del tablero.
COLUMNAS_CATEGORICAS_P2 = [
    "Area",                    # Urbano / Rural
    "fami_estratovivienda",    # Estrato 1..6
    "fami_educacionpadre",     # Nivel educativo del padre
    "fami_educacionmadre",     # Nivel educativo de la madre
    "cole_naturaleza",         # Oficial / No oficial
    "cole_jornada",            # Mañana, tarde, completa, sabatina, noche
    "cole_genero",             # Masculino, femenino, mixto
]

COLUMNAS_NUMERICAS_P2 = [
    "PIB miles de millones",
    "PIB per capita",
]


# ---------------------------------------------------------------------------
# 2.1 CARGA DE MODELOS PRE-ENTRENADOS
# ---------------------------------------------------------------------------

def cargar_artefactos_p2():
    """
    Carga modelos pre-entrenados y preprocesadores desde disco.
    Devuelve un dict con flag 'disponible' para que el tablero muestre
    un mensaje informativo si los archivos no existen todavía.
    """
    artefactos = {
        "modelo_reg": None,
        "modelo_clf": None,
        "preproc_reg": None,
        "preproc_clf": None,
        "metadata": None,
        "disponible": False,
        "error": None,
    }

    rutas = {
        "modelo_reg": os.path.join(MODELS_DIR, "regresion_puntaje.keras"),
        "modelo_clf": os.path.join(MODELS_DIR, "clasificacion_prioridad.keras"),
        "preproc_reg": os.path.join(MODELS_DIR, "preprocesador_reg.pkl"),
        "preproc_clf": os.path.join(MODELS_DIR, "preprocesador_clf.pkl"),
        "metadata": os.path.join(MODELS_DIR, "metadata_modelos.pkl"),
    }

    try:
        artefactos["modelo_reg"] = tf.keras.models.load_model(rutas["modelo_reg"])
        artefactos["modelo_clf"] = tf.keras.models.load_model(rutas["modelo_clf"])
        artefactos["preproc_reg"] = joblib.load(rutas["preproc_reg"])
        artefactos["preproc_clf"] = joblib.load(rutas["preproc_clf"])
        artefactos["metadata"] = joblib.load(rutas["metadata"])
        artefactos["disponible"] = True
    except FileNotFoundError as e:
        artefactos["error"] = (
            f"No se encontraron los modelos pre-entrenados en '{MODELS_DIR}/'. "
            f"Ejecute primero el notebook de entrenamiento. Detalle: {e}"
        )
    except Exception as e:
        artefactos["error"] = f"Error al cargar modelos: {e}"

    return artefactos


# ---------------------------------------------------------------------------
# 2.2 UTILIDADES PARA POBLAR EL FORMULARIO DEL TABLERO
# ---------------------------------------------------------------------------

def obtener_opciones_categoricas_p2(df):
    """Extrae las categorías únicas de cada variable categórica del dataset."""
    opciones = {}
    for col in COLUMNAS_CATEGORICAS_P2:
        if col in df.columns:
            valores = (
                df[col].dropna().astype(str).str.strip().unique().tolist()
            )
            valores = [v for v in valores if v.upper() not in
                       ("NAN", "SIN INFORMACION", "SIN INFORMACIÓN", "")]
            valores.sort()
            opciones[col] = valores
        else:
            opciones[col] = []
    return opciones


def obtener_pib_municipio(df, municipio):
    """
    Devuelve (PIB miles de millones, PIB per capita) del municipio.
    Si no se encuentra, devuelve la mediana departamental como respaldo.
    """
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


# ---------------------------------------------------------------------------
# 2.3 INFERENCIA: REGRESIÓN Y CLASIFICACIÓN
# ---------------------------------------------------------------------------

def _construir_dataframe_input(valores_form, columnas_esperadas):
    """DataFrame de una fila con las columnas exactas que espera el preprocesador."""
    fila = {col: valores_form.get(col, np.nan) for col in columnas_esperadas}
    return pd.DataFrame([fila])


def predecir_puntaje(artefactos, valores_form):
    """Inferencia del modelo de regresión: devuelve el puntaje global predicho."""
    if not artefactos["disponible"]:
        return None
    cols = artefactos["metadata"]["columnas_reg"]
    df_input = _construir_dataframe_input(valores_form, cols)
    X = artefactos["preproc_reg"].transform(df_input)
    y_pred = artefactos["modelo_reg"].predict(X, verbose=0).flatten()[0]
    return float(y_pred)


def predecir_prioridad(artefactos, valores_form):
    """Inferencia del clasificador: probabilidad de prioridad alta + etiqueta."""
    if not artefactos["disponible"]:
        return None, None
    cols = artefactos["metadata"]["columnas_clf"]
    df_input = _construir_dataframe_input(valores_form, cols)
    X = artefactos["preproc_clf"].transform(df_input)
    proba = artefactos["modelo_clf"].predict(X, verbose=0).flatten()[0]
    etiqueta = "Prioridad ALTA" if proba >= 0.5 else "Prioridad estándar"
    return float(proba), etiqueta


def simular_contrafactual_zona(artefactos, valores_form):
    """
    Corre dos veces el modelo de regresión: una con Area='Urbano' y otra con
    Area='Rural', manteniendo el resto del perfil fijo. Cuantifica el efecto
    puro del entorno geográfico para ese perfil socioeconómico específico.
    """
    if not artefactos["disponible"]:
        return None

    form_urb = dict(valores_form); form_urb["Area"] = "Urbano"
    form_rur = dict(valores_form); form_rur["Area"] = "Rural"

    p_urb = predecir_puntaje(artefactos, form_urb)
    p_rur = predecir_puntaje(artefactos, form_rur)
    if p_urb is None or p_rur is None:
        return None
    return {
        "puntaje_urbano": p_urb,
        "puntaje_rural": p_rur,
        "brecha_estimada": p_urb - p_rur,
    }


# ---------------------------------------------------------------------------
# 2.4 VISUALIZACIONES DE LOS MODELOS PREDICTIVOS
# ---------------------------------------------------------------------------

def grafica_velocimetro_riesgo(probabilidad, umbral=0.5):
    """Gauge con la probabilidad de prioridad alta."""
    if probabilidad is None:
        return go.Figure().update_layout(title="Modelo no disponible")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Probabilidad de Prioridad Alta", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2c3e50"},
            "steps": [
                {"range": [0, 33], "color": "#a8e6a3"},
                {"range": [33, 66], "color": "#ffe082"},
                {"range": [66, 100], "color": "#ef9a9a"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.85,
                "value": umbral * 100,
            },
        },
    ))
    fig.update_layout(margin={"r": 20, "t": 60, "l": 20, "b": 20}, height=320)
    return fig


def grafica_comparacion_contrafactual(resultado_cf, umbral_meta=250):
    """Barra horizontal Urbano vs Rural para el mismo perfil."""
    if resultado_cf is None:
        return go.Figure().update_layout(title="Modelo no disponible")

    df_plot = pd.DataFrame({
        "Zona": ["Urbano", "Rural"],
        "Puntaje Predicho": [resultado_cf["puntaje_urbano"], resultado_cf["puntaje_rural"]],
    })

    fig = px.bar(
        df_plot,
        x="Puntaje Predicho",
        y="Zona",
        orientation="h",
        color="Zona",
        text="Puntaje Predicho",
        color_discrete_map={"Urbano": "#1f77b4", "Rural": "#2ca02c"},
        title=f"Contrafactual: mismo perfil en Urbano vs Rural (brecha = {resultado_cf['brecha_estimada']:.1f} pts)",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.add_vline(x=umbral_meta, line_dash="dash", line_color="red",
                  annotation_text="Meta mínima", annotation_position="top")
    fig.update_layout(margin={"r": 20, "t": 60, "l": 20, "b": 20},
                      showlegend=False, height=320)
    return fig


def grafica_importancia_features(artefactos):
    """
    Importancia aproximada de variables vía pesos absolutos promedio de la
    primera capa densa del modelo de regresión. Proxy interpretable y barato.
    """
    if not artefactos["disponible"]:
        return go.Figure().update_layout(title="Modelo no disponible")

    try:
        preproc = artefactos["preproc_reg"]
        modelo = artefactos["modelo_reg"]

        primera_capa = modelo.layers[0]
        W = primera_capa.get_weights()[0]
        importancia = np.mean(np.abs(W), axis=1)

        nombres = preproc.get_feature_names_out()

        agregado = {}
        for nom, imp in zip(nombres, importancia):
            base = nom.split("__")[-1].split("_")[0:2]
            clave = "_".join(base) if base else nom
            agregado[clave] = agregado.get(clave, 0) + imp

        df_imp = (
            pd.DataFrame({"Feature": list(agregado.keys()),
                          "Importancia": list(agregado.values())})
            .sort_values("Importancia", ascending=True)
            .tail(12)
        )

        fig = px.bar(
            df_imp, x="Importancia", y="Feature", orientation="h",
            title="Importancia relativa de variables (regresión)",
            color="Importancia", color_continuous_scale="Blues",
        )
        fig.update_layout(margin={"r": 20, "t": 60, "l": 20, "b": 20},
                          coloraxis_showscale=False, height=400)
        return fig
    except Exception as e:
        return go.Figure().update_layout(title=f"No se pudo calcular importancia: {e}")


def grafica_metricas_modelos(artefactos):
    """Tabla con las métricas de evaluación de ambos modelos."""
    if not artefactos["disponible"]:
        return go.Figure().update_layout(title="Modelo no disponible")

    meta = artefactos["metadata"]
    metricas = meta.get("metricas", {})

    def _fmt(valor, decimales=2):
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return "N/D"
        return f"{valor:.{decimales}f}"

    filas = [
        ["Regresión - MAE (test)", _fmt(metricas.get("reg_mae"))],
        ["Regresión - RMSE (test)", _fmt(metricas.get("reg_rmse"))],
        ["Regresión - R² (test)", _fmt(metricas.get("reg_r2"), 3)],
        ["Clasificación - Accuracy", _fmt(metricas.get("clf_acc"), 3)],
        ["Clasificación - F1 (clase positiva)", _fmt(metricas.get("clf_f1"), 3)],
        ["Clasificación - AUC ROC", _fmt(metricas.get("clf_auc"), 3)],
        ["Umbral P25 (clasificación)", _fmt(meta.get("umbral_p25"), 1)],
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Métrica", "Valor"],
                    fill_color="#1f77b4", font=dict(color="white", size=13),
                    align="left"),
        cells=dict(values=list(zip(*filas)),
                   fill_color=[["#f5f5f5", "white"] * len(filas)],
                   align="left", font=dict(size=12), height=28),
    )])
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, height=300,
                      title="Evaluación de los modelos (conjunto de prueba)")
    return fig


# ---------------------------------------------------------------------------
# 2.5 INTERPRETACIÓN EN LENGUAJE NATURAL
# ---------------------------------------------------------------------------

def generar_interpretacion(puntaje, proba_prioridad, resultado_cf, umbral_meta=250):
    """Lectura del resultado para el usuario del Ministerio."""
    if puntaje is None or proba_prioridad is None:
        return ("Los modelos predictivos aún no están disponibles. "
                "Verifique que los archivos en la carpeta 'Models/' estén presentes.")

    nivel_riesgo = ("ALTO" if proba_prioridad >= 0.66
                    else "MEDIO" if proba_prioridad >= 0.33
                    else "BAJO")

    brecha_txt = ""
    if resultado_cf is not None:
        brecha = resultado_cf["brecha_estimada"]
        signo = "favorable a zonas urbanas" if brecha > 0 else "favorable a zonas rurales"
        brecha_txt = (
            f" Manteniendo idéntico el perfil socioeconómico, el modelo estima una "
            f"brecha de {abs(brecha):.1f} puntos {signo}, lo que sugiere que el efecto "
            f"contextual de la zona "
            f"{'es relevante' if abs(brecha) > 10 else 'es marginal'} para este perfil."
        )

    bajo_meta = puntaje < umbral_meta
    accion = ("Se recomienda intervención focalizada (nivelación, refuerzo, "
              "acompañamiento docente)." if bajo_meta or proba_prioridad >= 0.5
              else "El perfil no requiere intervención prioritaria, pero conviene "
                   "mantener monitoreo.")

    return (
        f"Puntaje global esperado: {puntaje:.1f} puntos. "
        f"Probabilidad de prioridad alta: {proba_prioridad * 100:.1f}% (riesgo {nivel_riesgo}). "
        f"{brecha_txt} {accion}"
    )