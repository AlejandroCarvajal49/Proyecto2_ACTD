"""
Pagina del tablero - Pregunta 1.

Layout (orden de la pagina):
    A) Laboratorio de modelos predictivos (MLflow + entrenamiento custom)
    B) Simulador de escenarios A/B (comparador de perfiles)
    C) Significancia de variables (OLS - pruebas t y F)
    D) Analisis descriptivo (boxplots, mapa PIB, brecha Urbano/Rural)
"""

import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from Analysis.logica_p1 import (
    # Datos descriptivos
    cargar_datos_p1,
    obtener_lista_municipios_p1,
    generar_boxplot_brecha,
    generar_dispersion_pib_brecha,
    calcular_estadisticas_brecha,
    generar_barras_brecha_error,
    generar_mapa_pib_puntaje,
    # Pipeline predictivo
    obtener_mlflow_info_p1,
    verificar_mlflow_ui_p1,
    iniciar_mlflow_ui_p1,
    cargar_resultados_mlflow_p1,
    cargar_historial_mlflow_p1,
    seleccionar_mejores_modelos_resultados_p1,
    seleccionar_top_modelos_resultados_p1,
    construir_figuras_comparativas_p1,
    entrenar_modelos_p1_personalizados,
    predecir_escenarios_p1,
    analizar_significancia_ols_p1,
    generar_interpretacion_p1,
    obtener_opciones_simulador_p1,
    obtener_pib_municipio,
    ETIQUETAS_VARIABLES_P1,
    COLUMNAS_CATEGORICAS_P2,
    COLUMNAS_NUMERICAS_P2,
)


dash.register_page(__name__, path='/pregunta_1', name="Brecha Urbano/Rural")


# ============================================================================
# CARGAS PRE-PAGINA
# ============================================================================
df_p1 = cargar_datos_p1()
lista_municipios = obtener_lista_municipios_p1(df_p1)
lista_municipios_simulador = [m for m in lista_municipios if m != 'TODOS']
grafica_pib_estatica = generar_dispersion_pib_brecha(df_p1)
mlflow_info = obtener_mlflow_info_p1()
opciones_sim = obtener_opciones_simulador_p1(df_p1)


# Catalogo de variables disponibles para el lab + OLS.
VARIABLES_P1 = [
    {"label": ETIQUETAS_VARIABLES_P1.get(v, v), "value": v}
    for v in COLUMNAS_CATEGORICAS_P2 + COLUMNAS_NUMERICAS_P2
]


COLUMNAS_MODELOS = [
    {"name": c, "id": c}
    for c in (
        "task", "config_id", "model", "feature_set", "layers", "dropout",
        "activation", "optimizer", "learning_rate", "epochs", "batch_size",
        "loss", "rmse", "mae", "r2", "accuracy", "f1", "roc_auc", "run_id",
    )
]

COLUMNAS_TOP = [
    {"name": c, "id": c}
    for c in (
        "rank", "model", "feature_set", "metric", "metric_value",
        "variables_included", "variables_excluded", "run_id",
    )
]

COLUMNAS_TTEST = [
    {"name": c, "id": c}
    for c in ("variable", "coef", "t_stat", "p_value", "significativa")
]

COLUMNAS_FTEST = [
    {"name": c, "id": c}
    for c in ("variable", "p_value", "incluida", "significativa")
]


def _default(options, fallback):
    return options[0] if options else fallback


def _dropdown(component_id, label, options, value, md=6):
    return dbc.Col(
        [
            html.Label(label, className="fw-bold small"),
            dcc.Dropdown(
                id=component_id,
                options=[{"label": o, "value": o} for o in options],
                value=value,
                clearable=False,
                className="mb-2 shadow-sm",
            ),
        ],
        md=md,
    )


def _format_vars(values, label_map):
    if not values:
        return "N/D"
    return ", ".join([label_map.get(v, v) for v in values])


def _enriquecer_top_df(top_df):
    if top_df.empty:
        return top_df
    label_map = {v["value"]: v["label"] for v in VARIABLES_P1}
    all_vars = [v["value"] for v in VARIABLES_P1]
    incluidos, excluidos = [], []
    for _, row in top_df.iterrows():
        selected_raw = row.get("selected_vars")
        if isinstance(selected_raw, str) and selected_raw.strip():
            selected = [v.strip() for v in selected_raw.split(",") if v.strip()]
        else:
            selected = []
        if not selected:
            incluidos.append("N/D")
            excluidos.append("N/D")
            continue
        incluidos.append(_format_vars(selected, label_map))
        excluidos.append(_format_vars([v for v in all_vars if v not in selected], label_map))
    enriched = top_df.copy()
    enriched["variables_included"] = incluidos
    enriched["variables_excluded"] = excluidos
    return enriched


# ============================================================================
# LAYOUT
# ============================================================================

layout = dbc.Container([

    html.H2("Pregunta 1: Brecha de desempeno urbano vs. rural",
            className="my-4 fw-bold"),
    dbc.Alert(
        [
            html.Strong("Pregunta de negocio: "),
            "Existe una brecha rural/urbana en el puntaje global del Saber 11 que "
            "justifique intervencion diferenciada en municipios de menor PIB? Se entrenan "
            "redes neuronales de regresion (puntaje global) y clasificacion binaria "
            "(prioridad alta = punt_global < P25 departamental).",
        ],
        color="info", className="shadow-sm",
    ),
    html.Hr(),

    # ========================================================================
    # A) LABORATORIO DE MODELOS PREDICTIVOS (MLflow)
    # ========================================================================

    dbc.Card([
        dbc.CardBody([
            html.H4("Laboratorio de modelos predictivos", className="mb-2"),
            html.P(
                "Carga las corridas registradas en MLflow para comparar arquitecturas "
                "y feature sets. Tambien puedes entrenar configuraciones personalizadas "
                "seleccionando las variables independientes desde aqui."
            ),
            html.Div([
                dbc.Button("CARGAR RESULTADOS MLflow", id="btn-cargar-resultados-p1",
                           color="primary", className="me-2"),
                dbc.Button("Iniciar MLflow UI", id="btn-iniciar-mlflow-p1",
                           color="secondary", outline=True, className="me-2"),
                dbc.Button("Abrir panel MLflow", href=mlflow_info["ui_url"],
                           target="_blank", color="secondary", outline=True),
            ], className="mb-3"),
            html.Small(f"Tracking URI: {mlflow_info['tracking_uri']}",
                       className="text-muted d-block mb-2"),
            html.Div(id="estado-mlflow-p1", className="text-muted mb-2"),
            html.Div(id="texto-modelos-p1", className="text-muted mb-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Entrenar con variables seleccionadas", className="mb-2"),
                    html.P(
                        "Selecciona variables independientes y ejecuta multiples "
                        "configuraciones. Las corridas quedan en MLflow y el mejor "
                        "modelo por task se guarda en models/pregunta_1/custom/."
                    ),
                    dcc.Dropdown(
                        id="variables-entrenar-p1",
                        options=VARIABLES_P1,
                        value=[v["value"] for v in VARIABLES_P1],
                        multi=True,
                        className="mb-2",
                    ),
                    dbc.Button("Entrenar modelos con variables",
                               id="btn-entrenar-variables-p1", color="success",
                               className="mb-2"),
                    html.Div(id="texto-entrenamiento-p1", className="text-muted"),
                ])
            ], className="mb-3"),

            dcc.Loading(
                dash_table.DataTable(
                    id="tabla-modelos-p1",
                    columns=COLUMNAS_MODELOS,
                    data=[],
                    page_size=8,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                ),
                type="default",
            ),

            html.H5("Top 2 modelos por task (MLflow - Pregunta 1)", className="mt-3"),
            dcc.Loading(
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Regresion (menor RMSE)", className="mb-2"),
                        dash_table.DataTable(
                            id="tabla-top-reg-p1",
                            columns=COLUMNAS_TOP, data=[], page_size=2,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ])), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Clasificacion binaria (mayor F1)", className="mb-2"),
                        dash_table.DataTable(
                            id="tabla-top-bin-p1",
                            columns=COLUMNAS_TOP, data=[], page_size=2,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                            style_header={"fontWeight": "bold"},
                        ),
                    ])), md=6),
                ], className="mb-3"),
                type="default",
            ),

            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-metricas-reg-p1", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="grafica-metricas-bin-p1", figure=go.Figure()), md=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-loss-p1", figure=go.Figure()), md=12),
            ]),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Interpretacion automatica", className="text-success fw-bold"),
                    html.P(id="texto-interpretacion-p1", className="mb-0"),
                ])
            ], className="mt-3 border-success"),
        ])
    ], className="mb-4 shadow-sm"),

    # ========================================================================
    # B) SIMULADOR DE ESCENARIOS A/B
    # ========================================================================

    dbc.Card([
        dbc.CardBody([
            html.H4("Simulador de escenarios A vs B", className="mb-3"),
            dbc.Alert(
                [
                    html.Strong("Regresion: "), "predice el puntaje global esperado. ",
                    html.Strong("Clasificacion binaria: "),
                    "estima la probabilidad de prioridad alta (puntaje < P25 departamental). "
                    "Configura dos perfiles y comparalos lado a lado.",
                ],
                color="info", className="shadow-sm",
            ),
            dbc.Alert(
                id="alerta-modelos-p1", color="danger", is_open=False,
                className="mb-3",
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario A", className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dropdown("a-municipio-p1", "Municipio (auto-pobla PIB)",
                                          lista_municipios_simulador,
                                          _default(lista_municipios_simulador, "MEDELLIN"), md=12),
                            ]),
                            dbc.Row([
                                _dropdown("a-area-p1", "Zona",
                                          opciones_sim.get("Area", ["Urbano", "Rural"]),
                                          _default(opciones_sim.get("Area"), "Urbano")),
                                _dropdown("a-estrato-p1", "Estrato vivienda",
                                          opciones_sim.get("fami_estratovivienda", []),
                                          _default(opciones_sim.get("fami_estratovivienda"), "Estrato 2")),
                            ]),
                            dbc.Row([
                                _dropdown("a-edupadre-p1", "Educacion padre",
                                          opciones_sim.get("fami_educacionpadre", []),
                                          _default(opciones_sim.get("fami_educacionpadre"), "")),
                                _dropdown("a-edumadre-p1", "Educacion madre",
                                          opciones_sim.get("fami_educacionmadre", []),
                                          _default(opciones_sim.get("fami_educacionmadre"), "")),
                            ]),
                            dbc.Row([
                                _dropdown("a-naturaleza-p1", "Naturaleza colegio",
                                          opciones_sim.get("cole_naturaleza", []),
                                          _default(opciones_sim.get("cole_naturaleza"), "")),
                                _dropdown("a-jornada-p1", "Jornada",
                                          opciones_sim.get("cole_jornada", []),
                                          _default(opciones_sim.get("cole_jornada"), "")),
                            ]),
                            dbc.Row([
                                _dropdown("a-genero-p1", "Genero colegio",
                                          opciones_sim.get("cole_genero", []),
                                          _default(opciones_sim.get("cole_genero"), ""), md=12),
                            ]),
                        ])
                    ], className="shadow-sm"),
                ], md=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario B", className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dropdown("b-municipio-p1", "Municipio (auto-pobla PIB)",
                                          lista_municipios_simulador,
                                          _default(lista_municipios_simulador[1:],
                                                   _default(lista_municipios_simulador, "MEDELLIN")),
                                          md=12),
                            ]),
                            dbc.Row([
                                _dropdown("b-area-p1", "Zona",
                                          opciones_sim.get("Area", ["Urbano", "Rural"]),
                                          "Rural" if "Rural" in opciones_sim.get("Area", []) else _default(opciones_sim.get("Area"), "Rural")),
                                _dropdown("b-estrato-p1", "Estrato vivienda",
                                          opciones_sim.get("fami_estratovivienda", []),
                                          _default(opciones_sim.get("fami_estratovivienda"), "Estrato 1")),
                            ]),
                            dbc.Row([
                                _dropdown("b-edupadre-p1", "Educacion padre",
                                          opciones_sim.get("fami_educacionpadre", []),
                                          _default(opciones_sim.get("fami_educacionpadre"), "")),
                                _dropdown("b-edumadre-p1", "Educacion madre",
                                          opciones_sim.get("fami_educacionmadre", []),
                                          _default(opciones_sim.get("fami_educacionmadre"), "")),
                            ]),
                            dbc.Row([
                                _dropdown("b-naturaleza-p1", "Naturaleza colegio",
                                          opciones_sim.get("cole_naturaleza", []),
                                          _default(opciones_sim.get("cole_naturaleza"), "")),
                                _dropdown("b-jornada-p1", "Jornada",
                                          opciones_sim.get("cole_jornada", []),
                                          _default(opciones_sim.get("cole_jornada"), "")),
                            ]),
                            dbc.Row([
                                _dropdown("b-genero-p1", "Genero colegio",
                                          opciones_sim.get("cole_genero", []),
                                          _default(opciones_sim.get("cole_genero"), ""), md=12),
                            ]),
                        ])
                    ], className="shadow-sm"),
                ], md=6),
            ], className="mb-3"),

            html.Div([
                dbc.Button("Comparar escenarios", id="btn-simular-p1",
                           color="success", className="me-2 mb-3"),
                dbc.Button("Contrafactual urbano↔rural (perfil A)",
                           id="btn-contrafactual-p1",
                           color="info", outline=True, className="mb-3"),
            ]),
            html.Small(
                "El botón \"contrafactual\" usa el perfil A y cambia solo la zona en B "
                "para aislar el efecto puro de urbano vs. rural.",
                className="text-muted d-block mb-2",
            ),

            dbc.Row([
                dbc.Col([
                    html.Small("Puntaje global (A)", className="text-muted"),
                    html.H4(id="out-puntaje-a-p1", children="—"),
                    html.Small("Prob. prioridad alta (A)", className="text-muted"),
                    html.H5(id="out-proba-a-p1", children="—"),
                    html.Small("Etiqueta (A)", className="text-muted"),
                    html.H6(id="out-etiqueta-a-p1", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Puntaje global (B)", className="text-muted"),
                    html.H4(id="out-puntaje-b-p1", children="—"),
                    html.Small("Prob. prioridad alta (B)", className="text-muted"),
                    html.H5(id="out-proba-b-p1", children="—"),
                    html.Small("Etiqueta (B)", className="text-muted"),
                    html.H6(id="out-etiqueta-b-p1", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Delta puntaje (B - A)", className="text-muted"),
                    html.H4(id="out-delta-puntaje-p1", children="—"),
                    html.Small("Delta probabilidad (B - A)", className="text-muted"),
                    html.H5(id="out-delta-proba-p1", children="—"),
                ], md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-escenarios-reg-p1", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="grafica-escenarios-clf-p1", figure=go.Figure()), md=6),
            ]),
        ])
    ], className="mb-4 shadow-sm"),

    # ========================================================================
    # C) ANALISIS DE SIGNIFICANCIA OLS (t-test y F-test)
    # ========================================================================

    dbc.Card([
        dbc.CardBody([
            html.H4("Significancia de variables (OLS)", className="mb-2"),
            html.P(
                "Selecciona variables independientes para la regresion lineal sobre "
                "punt_global. Se reportan pruebas t (para variables numericas) y "
                "pruebas F (por variable agrupada + combinaciones), todas con p-value."
            ),
            dcc.Dropdown(
                id="variables-ols-p1",
                options=VARIABLES_P1,
                value=[v["value"] for v in VARIABLES_P1],
                multi=True,
                className="mb-2",
            ),
            dbc.Button("Calcular significancia", id="btn-significancia-p1",
                       color="primary", className="mb-3"),
            html.Div(id="texto-significancia-p1", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    html.H6("Pruebas t (variables numericas)", className="mb-2"),
                    dash_table.DataTable(
                        id="tabla-ttest-p1",
                        columns=COLUMNAS_TTEST, data=[], page_size=6,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ], md=6),
                dbc.Col([
                    html.H6("Pruebas F (significancia por variable y combinaciones)", className="mb-2"),
                    dash_table.DataTable(
                        id="tabla-ftest-p1",
                        columns=COLUMNAS_FTEST, data=[], page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ], md=6),
            ]),
        ])
    ], className="mb-4 shadow-sm"),

    # ========================================================================
    # D) ANALISIS DESCRIPTIVO (Proyecto 1)
    # ========================================================================

    html.Hr(className="my-5"),
    html.H3("Analisis descriptivo (Proyecto 1)",
            className="mt-4 mb-3 text-primary fw-bold"),

    dbc.Row([
        dbc.Col([
            html.Label("Focalizar analisis por municipio:", className="fw-bold"),
            dcc.Dropdown(
                id='filtro-municipio-p1',
                options=[{'label': m, 'value': m} for m in lista_municipios],
                value='TODOS',
                clearable=False,
                className="mb-3 shadow-sm"
            )
        ], md=4)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Hallazgos e Insights (Prueba T-Student)",
                            className="card-title text-success fw-bold"),
                    html.P(id='texto-insight-p1', className="card-text fs-5"),
                ])
            ], className="mb-4 shadow-sm border-success")
        ], md=12),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='grafica-mapa-p1')]),
                          className="mb-4 shadow-sm"), md=12),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Comparacion de distribucion (Urbano vs Rural)",
                           className="fw-bold bg-light"),
            dbc.CardBody([dcc.Graph(id='grafica-boxplot-p1')]),
        ], className="mb-4 shadow-sm"), md=6),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Promedios con desviacion estandar",
                           className="fw-bold bg-light"),
            dbc.CardBody([
                dcc.Graph(id='grafica-barras-error-p1'),
                html.Small(
                    "Las lineas sobre las barras indican variabilidad (desviacion estandar).",
                    className="text-muted text-center d-block mt-2"
                ),
            ]),
        ], className="mb-4 shadow-sm"), md=6),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Impacto del PIB en la brecha educativa (departamental)",
                           className="fw-bold bg-light"),
            dbc.CardBody([
                dcc.Graph(figure=grafica_pib_estatica),
                html.Small(
                    "Y positivo = ventaja urbana. Muestra si municipios mas pobres "
                    "sufren brechas mas grandes.",
                    className="text-muted text-center d-block mt-2"
                ),
            ]),
        ], className="mb-4 shadow-sm"), md=12),
    ]),

], fluid=True)


# ============================================================================
# CALLBACKS - ANALISIS DESCRIPTIVO
# ============================================================================

@callback(
    [Output('grafica-boxplot-p1', 'figure'),
     Output('grafica-barras-error-p1', 'figure'),
     Output('grafica-mapa-p1', 'figure'),
     Output('texto-insight-p1', 'children')],
    [Input('filtro-municipio-p1', 'value')]
)
def actualizar_tablero_descriptivo(municipio):
    return (
        generar_boxplot_brecha(df_p1, municipio),
        generar_barras_brecha_error(df_p1, municipio),
        generar_mapa_pib_puntaje(df_p1, municipio),
        calcular_estadisticas_brecha(df_p1, municipio),
    )


# ============================================================================
# CALLBACKS - LABORATORIO MLflow
# ============================================================================

@callback(
    [
        Output("tabla-modelos-p1", "data"),
        Output("tabla-top-reg-p1", "data"),
        Output("tabla-top-bin-p1", "data"),
        Output("grafica-metricas-reg-p1", "figure"),
        Output("grafica-metricas-bin-p1", "figure"),
        Output("grafica-loss-p1", "figure"),
        Output("texto-modelos-p1", "children"),
        Output("texto-interpretacion-p1", "children"),
        Output("texto-entrenamiento-p1", "children"),
    ],
    [Input("btn-cargar-resultados-p1", "n_clicks"),
     Input("btn-entrenar-variables-p1", "n_clicks")],
    [State("variables-entrenar-p1", "value")],
    prevent_initial_call=True,
)
def cargar_resultados_p1(n_cargar, n_entrenar, variables_entrenar):
    if not (n_cargar or n_entrenar):
        empty = go.Figure()
        return [], [], [], empty, empty, empty, "", "", ""

    trigger = dash.ctx.triggered_id
    mensaje_entrenamiento = ""

    if trigger == "btn-entrenar-variables-p1":
        resumen, history, mejores, interpretacion, mensaje = entrenar_modelos_p1_personalizados(
            df_p1, variables_entrenar,
        )
        if not resumen.empty:
            resumen = resumen.copy()
            resumen["selected_vars"] = ",".join(variables_entrenar or [])
        mensaje_entrenamiento = mensaje
    else:
        resumen, mensaje = cargar_resultados_mlflow_p1()
        history = cargar_historial_mlflow_p1(resumen)
        mejores = seleccionar_mejores_modelos_resultados_p1(resumen)
        interpretacion = generar_interpretacion_p1(resumen, mejores, df_p1) if not resumen.empty else ""

    fig_reg, fig_bin, fig_loss = construir_figuras_comparativas_p1(resumen, history)

    top_df = seleccionar_top_modelos_resultados_p1(resumen, top_n=2)
    top_df = _enriquecer_top_df(top_df)
    top_reg = top_df[top_df["task"] == "regresion"] if not top_df.empty else pd.DataFrame()
    top_bin = top_df[top_df["task"] == "clasificacion_binaria"] if not top_df.empty else pd.DataFrame()

    if resumen.empty:
        return [], [], [], fig_reg, fig_bin, fig_loss, mensaje, interpretacion, mensaje_entrenamiento

    return (
        resumen.to_dict("records"),
        top_reg.to_dict("records"),
        top_bin.to_dict("records"),
        fig_reg, fig_bin, fig_loss,
        mensaje, interpretacion, mensaje_entrenamiento,
    )


@callback(
    Output("estado-mlflow-p1", "children"),
    [Input("btn-iniciar-mlflow-p1", "n_clicks"),
     Input("btn-cargar-resultados-p1", "n_clicks")],
    prevent_initial_call=True,
)
def actualizar_estado_mlflow_p1(n_iniciar, n_cargar):
    trigger = dash.ctx.triggered_id
    if trigger == "btn-iniciar-mlflow-p1":
        _, mensaje = iniciar_mlflow_ui_p1()
    else:
        _, mensaje = verificar_mlflow_ui_p1()
    return mensaje


# ============================================================================
# CALLBACKS - SIMULADOR A/B
# ============================================================================

@callback(
    [
        Output("out-puntaje-a-p1", "children"),
        Output("out-proba-a-p1", "children"),
        Output("out-etiqueta-a-p1", "children"),
        Output("out-puntaje-b-p1", "children"),
        Output("out-proba-b-p1", "children"),
        Output("out-etiqueta-b-p1", "children"),
        Output("out-delta-puntaje-p1", "children"),
        Output("out-delta-proba-p1", "children"),
        Output("grafica-escenarios-reg-p1", "figure"),
        Output("grafica-escenarios-clf-p1", "figure"),
        Output("alerta-modelos-p1", "children"),
        Output("alerta-modelos-p1", "is_open"),
    ],
    [Input("btn-simular-p1", "n_clicks"),
     Input("btn-contrafactual-p1", "n_clicks")],
    [
        State("a-municipio-p1", "value"),
        State("a-area-p1", "value"),
        State("a-estrato-p1", "value"),
        State("a-edupadre-p1", "value"),
        State("a-edumadre-p1", "value"),
        State("a-naturaleza-p1", "value"),
        State("a-jornada-p1", "value"),
        State("a-genero-p1", "value"),
        State("b-municipio-p1", "value"),
        State("b-area-p1", "value"),
        State("b-estrato-p1", "value"),
        State("b-edupadre-p1", "value"),
        State("b-edumadre-p1", "value"),
        State("b-naturaleza-p1", "value"),
        State("b-jornada-p1", "value"),
        State("b-genero-p1", "value"),
    ],
    prevent_initial_call=True,
)
def simular_escenarios_p1(
    n_simular, n_contrafactual,
    a_muni, a_area, a_estrato, a_edupadre, a_edumadre, a_natur, a_jorn, a_gen,
    b_muni, b_area, b_estrato, b_edupadre, b_edumadre, b_natur, b_jorn, b_gen,
):
    if not (n_simular or n_contrafactual):
        return tuple([no_update] * 12)

    # En modo contrafactual, B = perfil A pero con Area opuesta. Aisla el efecto
    # puro de la zona geografica manteniendo todo lo demas idéntico.
    if dash.ctx.triggered_id == "btn-contrafactual-p1":
        area_opciones = opciones_sim.get("Area", ["Urbano", "Rural"])
        opuesta = next((a for a in area_opciones if a != a_area), a_area)
        b_muni = a_muni
        b_area = opuesta
        b_estrato = a_estrato
        b_edupadre = a_edupadre
        b_edumadre = a_edumadre
        b_natur = a_natur
        b_jorn = a_jorn
        b_gen = a_gen

    pib_a_m, pib_a_pc = obtener_pib_municipio(df_p1, a_muni)
    pib_b_m, pib_b_pc = obtener_pib_municipio(df_p1, b_muni)

    valores_a = {
        "Area": a_area, "fami_estratovivienda": a_estrato,
        "fami_educacionpadre": a_edupadre, "fami_educacionmadre": a_edumadre,
        "cole_naturaleza": a_natur, "cole_jornada": a_jorn, "cole_genero": a_gen,
        "PIB miles de millones": pib_a_m, "PIB per capita": pib_a_pc,
        "municipio": a_muni,
    }
    valores_b = {
        "Area": b_area, "fami_estratovivienda": b_estrato,
        "fami_educacionpadre": b_edupadre, "fami_educacionmadre": b_edumadre,
        "cole_naturaleza": b_natur, "cole_jornada": b_jorn, "cole_genero": b_gen,
        "PIB miles de millones": pib_b_m, "PIB per capita": pib_b_pc,
        "municipio": b_muni,
    }

    pred_a, pred_b, fig_reg, fig_clf, error = predecir_escenarios_p1(valores_a, valores_b, df_base=df_p1)

    if error or pred_a is None or pred_b is None:
        empty = go.Figure()
        return (
            "—", "—", "—", "—", "—", "—", "—", "—",
            empty, empty,
            error or "Modelos no disponibles. Ejecuta el laboratorio o el script de entrenamiento.",
            True,
        )

    delta_p = pred_b["puntaje"] - pred_a["puntaje"]
    delta_pr = pred_b["proba_prioridad_alta"] - pred_a["proba_prioridad_alta"]

    return (
        f"{pred_a['puntaje']:.1f}",
        f"{pred_a['proba_prioridad_alta'] * 100:.1f}%",
        pred_a["etiqueta"],
        f"{pred_b['puntaje']:.1f}",
        f"{pred_b['proba_prioridad_alta'] * 100:.1f}%",
        pred_b["etiqueta"],
        f"{delta_p:+.1f}",
        f"{delta_pr * 100:+.1f}%",
        fig_reg, fig_clf,
        "", False,
    )


# ============================================================================
# CALLBACKS - SIGNIFICANCIA OLS
# ============================================================================

@callback(
    [
        Output("tabla-ttest-p1", "data"),
        Output("tabla-ftest-p1", "data"),
        Output("texto-significancia-p1", "children"),
    ],
    [Input("btn-significancia-p1", "n_clicks")],
    [State("variables-ols-p1", "value")],
    prevent_initial_call=True,
)
def calcular_significancia_p1(n, variables):
    if not n:
        return [], [], ""

    resumen, ttest_df, ftest_df, mensaje = analizar_significancia_ols_p1(df_p1, variables)
    if not resumen:
        return [], [], mensaje

    label_map = {v["value"]: v["label"] for v in VARIABLES_P1}
    if not ttest_df.empty:
        ttest_df["variable"] = ttest_df["variable"].map(label_map).fillna(ttest_df["variable"])
        for col in ("coef", "t_stat", "p_value"):
            ttest_df[col] = ttest_df[col].round(4)
    if not ftest_df.empty:
        ftest_df["variable"] = ftest_df["variable"].map(label_map).fillna(ftest_df["variable"])
        ftest_df["p_value"] = ftest_df["p_value"].apply(
            lambda v: round(v, 4) if pd.notna(v) else None
        )

    texto = (
        f"OLS con {resumen.get('n_obs', 0):,} observaciones | "
        f"R2={resumen.get('r2', 0):.3f} | Adj R2={resumen.get('adj_r2', 0):.3f}"
    )
    return ttest_df.to_dict("records"), ftest_df.to_dict("records"), texto
