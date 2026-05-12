import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


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