import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from Analysis.logica_p2 import (
    cargar_datos, filtrar_datos, calcular_brechas,
    generar_boxplots_materias, generar_mapa_brecha,
    generar_brecha_por_estrato, formato_periodo, MATERIAS,
    obtener_mlflow_info_p2, cargar_resultados_mlflow_p2,
    construir_figuras_mlflow_p2, predecir_escenarios_p2,
    OPCIONES_P2,
    generar_figura_brecha_materias_stat, generar_figura_brecha_estratos_stat,
)

dash.register_page(__name__, path="/pregunta_2")

df = cargar_datos()
municipios = ["Todos"] + sorted(df["cole_mcpio_ubicacion"].dropna().unique())
periodos = sorted(df["periodo"].dropna().unique())
mlflow_info = obtener_mlflow_info_p2()
OPT = OPCIONES_P2


def _dd(component_id, label, options, value):
    return dbc.Col([
        html.Label(label, className="fw-bold small"),
        dcc.Dropdown(
            id=component_id,
            options=[{"label": o, "value": o} for o in options],
            value=value,
            clearable=False,
            className="mb-2 shadow-sm",
        ),
    ], md=6)


def crear_tarjeta_brecha(nombre_materia):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H6(nombre_materia, className="text-center mb-2",
                        style={"fontWeight": "600", "fontSize": "13px", "color": "#444"}),
                html.Div(id=f"brecha-{nombre_materia}", children=[
                    html.P("Cargando...", className="text-center text-muted",
                           style={"fontSize": "12px"})
                ])
            ], style={"padding": "12px 8px"})
        ], className="shadow-sm", style={"borderRadius": "8px", "border": "1px solid #e0e0e0"}),
        width=4, className="mb-3"
    )


COLUMNAS_MODELOS_P2 = [
    {"name": "task", "id": "task"},
    {"name": "run_name", "id": "run_name"},
    {"name": "layers", "id": "layers"},
    {"name": "dropout", "id": "dropout"},
    {"name": "learning_rate", "id": "learning_rate"},
    {"name": "epochs_run", "id": "epochs_run"},
    {"name": "batch_size", "id": "batch_size"},
    {"name": "rmse", "id": "rmse"},
    {"name": "mae", "id": "mae"},
    {"name": "r2", "id": "r2"},
    {"name": "accuracy", "id": "accuracy"},
    {"name": "precision", "id": "precision"},
    {"name": "recall", "id": "recall"},
    {"name": "f1", "id": "f1"},
    {"name": "run_id", "id": "run_id"},
]

layout = html.Div([

    html.H2("Calidad Educativa: Colegios Públicos vs Privados",
            className="text-center mt-4 mb-1",
            style={"fontWeight": "bold", "color": "#222"}),
    html.P("Análisis de brechas en puntajes Saber 11 en Antioquia",
           className="text-center mb-4",
           style={"fontSize": "14px", "color": "#666"}),

    # ── PREGUNTA DE NEGOCIO ────────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            dcc.Markdown("""
### Pregunta de negocio
¿En qué áreas del conocimiento presentan los colegios públicos el mayor rezago frente a los
privados en Antioquia, y en qué medida estas diferencias se mantienen al controlar por el nivel
socioeconómico de los estudiantes?

### Modelos propuestos
**Regresión**: predice el puntaje global (0-500) en función del tipo de colegio (público/privado),
estrato socioeconómico y educación de los padres.

**Clasificación binaria**: predice si un estudiante tiene bajo rendimiento (punt_global < 250,
umbral igual a la media departamental). Al incluir cole_naturaleza y fami_estratovivienda como
entradas, el modelo permite cuantificar el efecto neto del tipo de colegio controlando por estrato.
""")
        ])
    ], className="mb-4 shadow-sm"),

    # ── PRUEBAS ESTADÍSTICAS ───────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            html.H4("Pruebas estadísticas", className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("t-Welch punt_global (público vs privado)", className="text-muted mb-1"),
                        html.H4("15.36 pts", className="text-danger fw-bold mb-0"),
                        html.Small("Brecha media (privado − público) · p<0.001 *** · Cohen's d=0.31 (small)",
                                   className="text-muted"),
                    ])
                ], className="shadow-sm h-100"), md=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Mayor brecha por materia: Inglés", className="text-muted mb-1"),
                        html.H4("d = 0.56 (medium)", className="text-danger fw-bold mb-0"),
                        html.Small("Único efecto práctico medio; resto de materias → efecto small",
                                   className="text-muted"),
                    ])
                ], className="shadow-sm h-100"), md=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("ANOVA punt_global ~ estrato", className="text-muted mb-1"),
                        html.H4("F = 13 350 · p<0.001", className="text-danger fw-bold mb-0"),
                        html.Small("Efecto escalera socioeconómico altamente significativo",
                                   className="text-muted"),
                    ])
                ], className="shadow-sm h-100"), md=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=generar_figura_brecha_materias_stat()), md=6),
                dbc.Col(dcc.Graph(figure=generar_figura_brecha_estratos_stat()), md=6),
            ]),

            dbc.Alert([
                html.Strong("Hallazgo clave: "),
                "En estratos 1 y 2 los colegios públicos superan a los privados (brechas de −13.5 y −11.4 pts). "
                "La ventaja privada aparece solo desde estrato 3 y crece fuertemente hasta estrato 6 "
                "(+114.8 pts, Cohen's d = 2.31 large). "
                "Esto sugiere que el efecto 'privado = mejor' es un artefacto del nivel socioeconómico, "
                "no de la naturaleza del colegio per se.",
            ], color="warning", className="mt-3"),
        ])
    ], className="mb-4 shadow-sm"),

    # ── LABORATORIO DE MODELOS ─────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            html.H4("Laboratorio de modelos predictivos", className="mb-2"),
            html.P(
                "Carga y compara los resultados registrados en MLflow para los dos modelos de P2. "
                "El entrenamiento se ejecutó con Analysis/entrenar_modelos_p2.py (3 configuraciones por tarea).",
                className="text-muted",
            ),
            dbc.Button(
                "Cargar resultados MLflow",
                id="p2-btn-cargar-mlflow",
                color="primary",
                className="mb-3",
            ),
            html.Small(
                f"Tracking URI: {mlflow_info['tracking_uri']}",
                className="text-muted d-block mb-2",
            ),
            html.Div(id="p2-texto-mlflow", className="text-muted mb-3"),

            dcc.Loading(
                dash_table.DataTable(
                    id="p2-tabla-modelos",
                    columns=COLUMNAS_MODELOS_P2,
                    data=[],
                    page_size=6,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                ),
                type="default",
            ),

            dbc.Row([
                dbc.Col(dcc.Graph(id="p2-grafica-reg-mlflow", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="p2-grafica-clf-mlflow", figure=go.Figure()), md=6),
            ], className="mt-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Interpretación", className="text-success fw-bold"),
                    html.P(id="p2-texto-interpretacion", className="mb-0 text-muted"),
                ])
            ], className="mt-3 border-success"),
        ])
    ], className="mb-4 shadow-sm"),

    # ── SIMULADOR DE ESCENARIOS ───────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            html.H4("Simulador de escenarios", className="mb-3"),
            dbc.Alert(
                [
                    html.Strong("Regresión: "), "predice el puntaje global esperado (0-500). ",
                    html.Strong("Clasificación: "),
                    "estima la probabilidad de bajo rendimiento (punt_global < 250). ",
                    "Compara dos perfiles para aislar el efecto de cada variable.",
                ],
                color="info", className="shadow-sm",
            ),
            dbc.Alert(id="p2-alerta-sim", color="danger", is_open=False, className="mb-3"),

            dbc.Row([
                # Escenario A
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario A — Público / Estrato bajo",
                                       className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dd("p2-a-naturaleza", "Tipo de colegio",
                                    OPT["cole_naturaleza"], "OFICIAL"),
                                _dd("p2-a-estrato", "Estrato vivienda",
                                    OPT["fami_estratovivienda"], "Estrato 1"),
                            ]),
                            dbc.Row([
                                _dd("p2-a-edu-madre", "Educacion madre",
                                    OPT["fami_educacionmadre"], "Primaria incompleta"),
                                _dd("p2-a-edu-padre", "Educacion padre",
                                    OPT["fami_educacionpadre"], "Primaria incompleta"),
                            ]),
                            dbc.Row([
                                _dd("p2-a-area", "Zona del colegio",
                                    OPT["cole_area_ubicacion"], "RURAL"),
                                _dd("p2-a-jornada", "Jornada",
                                    OPT["cole_jornada"], "TARDE"),
                            ]),
                        ]),
                    ], className="shadow-sm"),
                ], md=6),

                # Escenario B
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario B — Privado / Estrato alto",
                                       className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dd("p2-b-naturaleza", "Tipo de colegio",
                                    OPT["cole_naturaleza"], "NO OFICIAL"),
                                _dd("p2-b-estrato", "Estrato vivienda",
                                    OPT["fami_estratovivienda"], "Estrato 5"),
                            ]),
                            dbc.Row([
                                _dd("p2-b-edu-madre", "Educacion madre",
                                    OPT["fami_educacionmadre"], "Educacion profesional completa"),
                                _dd("p2-b-edu-padre", "Educacion padre",
                                    OPT["fami_educacionpadre"], "Educacion profesional completa"),
                            ]),
                            dbc.Row([
                                _dd("p2-b-area", "Zona del colegio",
                                    OPT["cole_area_ubicacion"], "URBANO"),
                                _dd("p2-b-jornada", "Jornada",
                                    OPT["cole_jornada"], "COMPLETA"),
                            ]),
                        ]),
                    ], className="shadow-sm"),
                ], md=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Switch(
                        id="p2-switch-controlada",
                        label="Comparación controlada (solo tipo de colegio)",
                        value=False,
                        className="mb-1",
                    ),
                    html.Small(
                        "Cuando está activo, el Escenario B es idéntico al A "
                        "excepto por el tipo de colegio (público ↔ privado). "
                        "El delta resultante aísla el efecto neto de la naturaleza del colegio, "
                        "equivalente a la prueba t estratificada dentro del mismo estrato.",
                        className="text-muted d-block mb-3",
                    ),
                ]),
            ]),

            dbc.Button("Comparar escenarios", id="p2-btn-simular",
                       color="success", className="mb-4"),

            dbc.Row([
                dbc.Col([
                    html.Small("Puntaje global (A)", className="text-muted d-block"),
                    html.H4(id="p2-out-puntaje-a", children="—"),
                    html.Small("Riesgo bajo rendimiento (A)", className="text-muted d-block mt-2"),
                    html.H5(id="p2-out-riesgo-a", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Puntaje global (B)", className="text-muted d-block"),
                    html.H4(id="p2-out-puntaje-b", children="—"),
                    html.Small("Riesgo bajo rendimiento (B)", className="text-muted d-block mt-2"),
                    html.H5(id="p2-out-riesgo-b", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Diferencia puntaje (B − A)", className="text-muted d-block"),
                    html.H4(id="p2-out-delta-puntaje", children="—"),
                    html.Small("Diferencia riesgo (B − A)", className="text-muted d-block mt-2"),
                    html.H5(id="p2-out-delta-riesgo", children="—"),
                ], md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="p2-grafica-sim-reg", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="p2-grafica-sim-clf", figure=go.Figure()), md=6),
            ]),
        ])
    ], className="mb-4 shadow-sm"),

    html.Hr(style={"borderColor": "#ddd"}),
    html.H4("Análisis descriptivo", className="mt-3 mb-3",
            style={"color": "#333", "fontWeight": "600"}),

    # ── FILTROS GLOBALES ───────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Municipio", className="fw-bold", style={"fontSize": "13px"}),
            dcc.Dropdown(
                id="filtro-municipio",
                options=[{"label": m, "value": m} for m in municipios],
                value="Todos",
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            html.Label("Periodo", className="fw-bold", style={"fontSize": "13px"}),
            html.Div([
                dcc.RangeSlider(
                    id="filtro-periodo-timeline",
                    min=0,
                    max=len(periodos) - 1,
                    step=1,
                    marks={i: {"label": formato_periodo(p),
                               "style": {"fontSize": "11px", "transform": "rotate(-45deg)"}}
                           for i, p in enumerate(periodos)},
                    value=[0, len(periodos) - 1],
                    tooltip={"placement": "top", "always_visible": False},
                    allowCross=False,
                )
            ], style={"padding": "5px 10px 25px 10px"})
        ], width=7),
    ], justify="center", className="mb-4"),

    html.Hr(style={"borderColor": "#ddd"}),

    html.H5("Brecha por materia (Privado − Público)",
            className="text-center mt-3 mb-3",
            style={"fontWeight": "600", "color": "#333"}),
    dbc.Row(
        [crear_tarjeta_brecha(nombre) for nombre in list(MATERIAS.keys())[:3]],
        justify="center",
    ),
    dbc.Row(
        [crear_tarjeta_brecha(nombre) for nombre in list(MATERIAS.keys())[3:]],
        justify="center",
        className="mb-3",
    ),

    html.Hr(style={"borderColor": "#ddd"}),
    dcc.Graph(id="grafica-boxplot-brecha"),
    html.Hr(style={"borderColor": "#ddd"}),

    dbc.Row([
        dbc.Col([
            html.Label("Materia", className="fw-bold", style={"fontSize": "13px"}),
            dcc.Dropdown(
                id="filtro-materia-estrato",
                options=[{"label": nombre, "value": col} for nombre, col in MATERIAS.items()],
                value="punt_global",
                clearable=False,
            ),
        ], width=3)
    ], justify="center", className="mb-3"),
    dcc.Graph(id="grafica-brecha-estrato"),

    html.Hr(style={"borderColor": "#ddd"}),

    dbc.Row([
        dbc.Col([
            html.Label("Materia", className="fw-bold", style={"fontSize": "13px"}),
            dcc.Dropdown(
                id="filtro-materia-mapa",
                options=[{"label": nombre, "value": col} for nombre, col in MATERIAS.items()],
                value="punt_global",
                clearable=False,
            ),
        ], width=3)
    ], justify="center", className="mb-3"),
    dcc.Graph(id="grafica-mapa-brecha"),
    html.Br(),
])


# ── CALLBACKS LABORATORIO MLflow ───────────────────────────────────────────

@callback(
    Output("p2-tabla-modelos", "data"),
    Output("p2-grafica-reg-mlflow", "figure"),
    Output("p2-grafica-clf-mlflow", "figure"),
    Output("p2-texto-mlflow", "children"),
    Output("p2-texto-interpretacion", "children"),
    Input("p2-btn-cargar-mlflow", "n_clicks"),
    prevent_initial_call=True,
)
def cargar_lab_mlflow(n_clicks):
    if not n_clicks:
        return [], go.Figure(), go.Figure(), "", ""

    df_runs, mensaje = cargar_resultados_mlflow_p2()

    if df_runs.empty:
        return [], go.Figure(), go.Figure(), mensaje, ""

    fig_reg, fig_clf = construir_figuras_mlflow_p2(df_runs)

    reg = df_runs[df_runs["task"] == "regresion"].dropna(subset=["rmse"])
    clf = df_runs[df_runs["task"] == "clasificacion_binaria"].dropna(subset=["f1"])

    partes = []
    if not reg.empty:
        best = reg.loc[reg["rmse"].idxmin()]
        partes.append(
            f"Mejor regresión: config '{best['run_name']}' con RMSE={best['rmse']:.3f}, "
            f"MAE={best['mae']:.3f}, R²={best['r2']:.4f}."
        )
    if not clf.empty:
        best = clf.loc[clf["f1"].idxmax()]
        partes.append(
            f"Mejor clasificación: config '{best['run_name']}' con F1={best['f1']:.4f}, "
            f"Accuracy={best['accuracy']:.4f}, Precision={best['precision']:.4f}, "
            f"Recall={best['recall']:.4f}."
        )
    partes.append(
        "El modelo incluye cole_naturaleza y fami_estratovivienda como entradas, "
        "lo que permite cuantificar la brecha público-privado controlando por nivel socioeconómico."
    )
    interpretacion = " ".join(partes)

    return df_runs.to_dict("records"), fig_reg, fig_clf, mensaje, interpretacion


# ── CALLBACK SIMULADOR ─────────────────────────────────────────────────────

@callback(
    Output("p2-out-puntaje-a", "children"),
    Output("p2-out-riesgo-a", "children"),
    Output("p2-out-puntaje-b", "children"),
    Output("p2-out-riesgo-b", "children"),
    Output("p2-out-delta-puntaje", "children"),
    Output("p2-out-delta-riesgo", "children"),
    Output("p2-grafica-sim-reg", "figure"),
    Output("p2-grafica-sim-clf", "figure"),
    Output("p2-alerta-sim", "children"),
    Output("p2-alerta-sim", "is_open"),
    Input("p2-btn-simular", "n_clicks"),
    State("p2-a-naturaleza", "value"),
    State("p2-a-estrato", "value"),
    State("p2-a-edu-madre", "value"),
    State("p2-a-edu-padre", "value"),
    State("p2-a-area", "value"),
    State("p2-a-jornada", "value"),
    State("p2-b-naturaleza", "value"),
    State("p2-b-estrato", "value"),
    State("p2-b-edu-madre", "value"),
    State("p2-b-edu-padre", "value"),
    State("p2-b-area", "value"),
    State("p2-b-jornada", "value"),
    State("p2-switch-controlada", "value"),
    prevent_initial_call=True,
)
def simular_p2(
    n_clicks,
    a_nat, a_est, a_edu_madre, a_edu_padre, a_area, a_jornada,
    b_nat, b_est, b_edu_madre, b_edu_padre, b_area, b_jornada,
    controlada,
):
    if not n_clicks:
        return "—", "—", "—", "—", "—", "—", go.Figure(), go.Figure(), "", False

    valores_a = {
        "cole_naturaleza": a_nat,
        "fami_estratovivienda": a_est,
        "fami_educacionmadre": a_edu_madre,
        "fami_educacionpadre": a_edu_padre,
        "cole_area_ubicacion": a_area,
        "cole_jornada": a_jornada,
    }

    if controlada:
        opts_nat = OPT["cole_naturaleza"]
        b_nat_ctrl = opts_nat[1] if a_nat == opts_nat[0] else opts_nat[0]
        valores_b = {**valores_a, "cole_naturaleza": b_nat_ctrl}
    else:
        valores_b = {
            "cole_naturaleza": b_nat,
            "fami_estratovivienda": b_est,
            "fami_educacionmadre": b_edu_madre,
            "fami_educacionpadre": b_edu_padre,
            "cole_area_ubicacion": b_area,
            "cole_jornada": b_jornada,
        }

    pred_a, pred_b, fig_reg, fig_clf, error = predecir_escenarios_p2(valores_a, valores_b)

    if error:
        return "—", "—", "—", "—", "—", "—", go.Figure(), go.Figure(), error, True

    delta_puntaje = pred_b["puntaje"] - pred_a["puntaje"]
    delta_riesgo = (pred_b["proba_bajo"] - pred_a["proba_bajo"]) * 100

    signo_p = "+" if delta_puntaje >= 0 else ""
    signo_r = "+" if delta_riesgo >= 0 else ""

    return (
        f"{pred_a['puntaje']:.1f} pts",
        f"{pred_a['proba_bajo'] * 100:.1f}%",
        f"{pred_b['puntaje']:.1f} pts",
        f"{pred_b['proba_bajo'] * 100:.1f}%",
        f"{signo_p}{delta_puntaje:.1f} pts",
        f"{signo_r}{delta_riesgo:.1f}%",
        fig_reg,
        fig_clf,
        "",
        False,
    )


# ── CALLBACKS DESCRIPTIVOS (sin cambios) ──────────────────────────────────

@dash.callback(
    Output("grafica-boxplot-brecha", "figure"),
    *[Output(f"brecha-{nombre}", "children") for nombre in MATERIAS.keys()],
    Input("filtro-municipio", "value"),
    Input("filtro-periodo-timeline", "value"),
)
def actualizar_principales(municipio, rango_periodo):
    idx_min, idx_max = rango_periodo
    periodos_sel = periodos[idx_min:idx_max + 1]
    df_f = filtrar_datos(df, municipio=municipio, periodo=periodos_sel)
    fig_boxplot = generar_boxplots_materias(df_f)
    brechas = calcular_brechas(df_f)

    tarjetas = []
    for nombre in MATERIAS.keys():
        info = brechas.get(nombre, {})
        brecha_val = info.get("brecha")
        media_pub = info.get("media_publico")
        media_priv = info.get("media_privado")
        if brecha_val is not None:
            color_brecha = "#c0392b" if brecha_val > 0 else ("#2166ac" if brecha_val < 0 else "#888")
            signo = "+" if brecha_val > 0 else ""
            contenido = html.Div([
                html.H4(f"{signo}{brecha_val:.1f} pts",
                        className="text-center mb-1",
                        style={"color": color_brecha, "fontWeight": "bold",
                               "fontSize": "18px", "marginBottom": "4px"}),
                html.P([
                    html.Span(f"Púb: {media_pub}",
                              style={"color": "#1f77b4", "fontSize": "11px"}),
                    html.Span(" | ", style={"color": "#999", "fontSize": "11px"}),
                    html.Span(f"Priv: {media_priv}",
                              style={"color": "#c0392b", "fontSize": "11px"}),
                ], className="text-center", style={"marginBottom": "0"}),
            ])
        else:
            contenido = html.P("Sin datos", className="text-center text-muted",
                               style={"fontSize": "12px"})
        tarjetas.append(contenido)

    return fig_boxplot, *tarjetas


@dash.callback(
    Output("grafica-brecha-estrato", "figure"),
    Input("filtro-municipio", "value"),
    Input("filtro-periodo-timeline", "value"),
    Input("filtro-materia-estrato", "value"),
)
def actualizar_estrato(municipio, rango_periodo, columna_materia):
    idx_min, idx_max = rango_periodo
    periodos_sel = periodos[idx_min:idx_max + 1]
    df_f = filtrar_datos(df, municipio=municipio, periodo=periodos_sel)
    return generar_brecha_por_estrato(df_f, columna_materia)


@dash.callback(
    Output("grafica-mapa-brecha", "figure"),
    Input("filtro-municipio", "value"),
    Input("filtro-periodo-timeline", "value"),
    Input("filtro-materia-mapa", "value"),
)
def actualizar_mapa(municipio, rango_periodo, columna_materia):
    idx_min, idx_max = rango_periodo
    periodos_sel = periodos[idx_min:idx_max + 1]
    df_f = filtrar_datos(df, municipio="Todos", periodo=periodos_sel)
    return generar_mapa_brecha(df_f, columna_materia, municipio_seleccionado=municipio)
