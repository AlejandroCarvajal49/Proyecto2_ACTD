import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from Analysis.logica_p3 import (
    cargar_datos_p3,
    generar_mapa_antioquia,
    generar_ranking_municipios_estatico,
    generar_histograma_tic,
    generar_dispersion_clusters,
    calcular_probabilidad_b1,
    generar_serie_tic_ingles_por_periodo,
    obtener_lista_municipios,
    obtener_opciones_simulador_p3,
    cargar_resultados_mlflow_p3,
    cargar_historial_mlflow_p3,
    seleccionar_mejores_modelos_resultados,
    seleccionar_top_modelos_resultados,
    generar_interpretacion_p3,
    verificar_mlflow_ui,
    iniciar_mlflow_ui,
    analizar_significancia_ols,
    entrenar_modelos_p3_personalizados,
    construir_figuras_comparativas,
    predecir_escenarios_p3,
    obtener_mlflow_info,
)

dash.register_page(__name__, path='/pregunta_3', name="Competitividad / Bilingüismo")

df_p3 = cargar_datos_p3()
lista_municipios = obtener_lista_municipios(df_p3)
ranking_estatico = generar_ranking_municipios_estatico(df_p3)
mlflow_info = obtener_mlflow_info()
opciones_sim = obtener_opciones_simulador_p3(df_p3)

COLUMNAS_MODELOS = [
    {"name": "task", "id": "task"},
    {"name": "config_id", "id": "config_id"},
    {"name": "model", "id": "model"},
    {"name": "feature_set", "id": "feature_set"},
    {"name": "layers", "id": "layers"},
    {"name": "dropout", "id": "dropout"},
    {"name": "activation", "id": "activation"},
    {"name": "optimizer", "id": "optimizer"},
    {"name": "learning_rate", "id": "learning_rate"},
    {"name": "epochs", "id": "epochs"},
    {"name": "batch_size", "id": "batch_size"},
    {"name": "loss", "id": "loss"},
    {"name": "rmse", "id": "rmse"},
    {"name": "mae", "id": "mae"},
    {"name": "r2", "id": "r2"},
    {"name": "accuracy", "id": "accuracy"},
    {"name": "f1", "id": "f1"},
    {"name": "roc_auc", "id": "roc_auc"},
    {"name": "f1_macro", "id": "f1_macro"},
    {"name": "log_loss", "id": "log_loss"},
    {"name": "run_id", "id": "run_id"},
]

COLUMNAS_TOP = [
    {"name": "rank", "id": "rank"},
    {"name": "model", "id": "model"},
    {"name": "feature_set", "id": "feature_set"},
    {"name": "metric", "id": "metric"},
    {"name": "metric_value", "id": "metric_value"},
    {"name": "variables_included", "id": "variables_included"},
    {"name": "variables_excluded", "id": "variables_excluded"},
    {"name": "run_id", "id": "run_id"},
]

VARIABLES_P3 = [
    {"label": "Acceso a Internet", "value": "internet_flag"},
    {"label": "Acceso a Computador", "value": "computador_flag"},
    {"label": "Score TIC (0-2)", "value": "tic_score"},
    {"label": "Interaccion Internet*Computador", "value": "tic_interaccion"},
    {"label": "Colegio bilingue", "value": "bilingue_flag"},
    {"label": "Estrato vivienda", "value": "estrato_cat"},
    {"label": "Zona de residencia", "value": "zona"},
    {"label": "Tipo de colegio", "value": "tipo_colegio"},
    {"label": "Jornada", "value": "jornada"},
    {"label": "Genero del colegio", "value": "genero"},
    {"label": "Educacion padre", "value": "edu_padre"},
    {"label": "Educacion madre", "value": "edu_madre"},
]

COLUMNAS_TTEST = [
    {"name": "variable", "id": "variable"},
    {"name": "coef", "id": "coef"},
    {"name": "t_stat", "id": "t_stat"},
    {"name": "p_value", "id": "p_value"},
    {"name": "significativa", "id": "significativa"},
]

COLUMNAS_FTEST = [
    {"name": "variable", "id": "variable"},
    {"name": "p_value", "id": "p_value"},
    {"name": "incluida", "id": "incluida"},
    {"name": "significativa", "id": "significativa"},
]


def _default(options, fallback):
    return options[0] if options else fallback


def _dropdown(component_id, label, options, value):
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
        md=6,
    )


def _parse_selected_vars(selected_vars):
    if isinstance(selected_vars, str) and selected_vars.strip():
        return [v.strip() for v in selected_vars.split(",") if v.strip()]
    return None


def _infer_selected_vars(selected_vars, feature_set):
    parsed = _parse_selected_vars(selected_vars)
    if parsed:
        return list(dict.fromkeys(parsed))
    if feature_set == "tic_basico":
        return ["internet_flag", "computador_flag", "tic_score", "tic_interaccion"]
    if feature_set == "tic_contexto":
        return [
            "internet_flag",
            "computador_flag",
            "tic_score",
            "tic_interaccion",
            "bilingue_flag",
            "estrato_cat",
            "zona",
            "tipo_colegio",
            "jornada",
            "genero",
            "edu_padre",
            "edu_madre",
        ]
    return None


def _format_vars(values, label_map):
    if not values:
        return "N/D"
    return ", ".join([label_map.get(v, v) for v in values])


def _enriquecer_top_df(top_df):
    if top_df.empty:
        return top_df

    label_map = {v["value"]: v["label"] for v in VARIABLES_P3}
    all_vars = [v["value"] for v in VARIABLES_P3]

    incluidos = []
    excluidos = []
    for _, row in top_df.iterrows():
        selected = _infer_selected_vars(row.get("selected_vars"), row.get("feature_set"))
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

layout = dbc.Container([
    html.H2("Competitividad y Bilinguismo: Impacto TIC", className="my-4"),
    html.Hr(),

    dbc.Card([
        dbc.CardBody([
            dcc.Markdown(
                """
### Pregunta de negocio
Como se distribuye el nivel de desempeno en ingles (A-, A1, A2, B1, B+) a lo largo de Antioquia y que relacion tiene con el acceso a herramientas tecnologicas (Internet / Computador) en el hogar?

### Modelos propuestos
**Regresion**: explicar el puntaje de ingles en funcion de acceso a Internet, acceso a computador, variables de contexto del estudiante y variables asociadas al colegio.

**Clasificacion**: predecir la probabilidad de alcanzar un nivel alto de ingles.
- Binaria: B1 o superior vs inferior a B1.
- Multiclase: A-, A1, A2, B1, B+.
"""
            )
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardBody([
            html.H4("Laboratorio de modelos predictivos", className="mb-2"),
            html.P(
                "Este panel no reentrena. Solo carga y compara resultados ya registrados en MLflow. "
                "Si no hay corridas, ejecuta el entrenamiento fuera del dashboard."
            ),
            html.Div([
                dbc.Button(
                    "CARGAR RESULTADOS MLflow",
                    id="btn-cargar-resultados",
                    color="primary",
                    className="me-2"
                ),
                dbc.Button(
                    "Iniciar MLflow UI",
                    id="btn-iniciar-mlflow",
                    color="secondary",
                    outline=True,
                    className="me-2"
                ),
                dbc.Button(
                    "Abrir panel MLflow",
                    href=mlflow_info["ui_url"],
                    target="_blank",
                    color="secondary",
                    outline=True
                ),
            ], className="mb-3"),
            html.Small(f"Tracking URI: {mlflow_info['tracking_uri']}", className="text-muted d-block mb-2"),
            html.Div(id="estado-mlflow", className="text-muted mb-2"),
            html.Div(id="texto-modelos", className="text-muted mb-3"),
            html.Ul([
                html.Li("Regresion: variable dependiente = punt_ingles."),
                html.Li("Clasificacion: objetivo B1+ (binaria) o niveles A-/A1/A2/B1/B+ (multiclase)."),
                html.Li("Variables independientes: TIC (internet/computador) + contexto del estudiante y del colegio."),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Entrenamiento con variables seleccionadas", className="mb-2"),
                    html.P(
                        "Selecciona variables independientes y ejecuta multiples modelos con hiperparametros distintos. "
                        "Las corridas quedan registradas en MLflow y se selecciona el mejor automaticamente."
                    ),
                    dcc.Dropdown(
                        id="variables-entrenar",
                        options=VARIABLES_P3,
                        value=[v["value"] for v in VARIABLES_P3],
                        multi=True,
                        className="mb-2",
                    ),
                    dbc.Button(
                        "Entrenar modelos con variables",
                        id="btn-entrenar-variables",
                        color="success",
                        className="mb-2",
                    ),
                    html.Div(id="texto-entrenamiento", className="text-muted"),
                ])
            ], className="mb-3"),

            dcc.Loading(
                dash_table.DataTable(
                    id="tabla-modelos",
                    columns=COLUMNAS_MODELOS,
                    data=[],
                    page_size=8,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                    style_header={"fontWeight": "bold"}
                ),
                type="default",
            ),

            html.H5("Top 2 modelos por tarea (MLflow - Pregunta 3)", className="mt-3"),
            dcc.Loading(
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6("Regresion", className="mb-2"),
                                dash_table.DataTable(
                                    id="tabla-top-reg",
                                    columns=COLUMNAS_TOP,
                                    data=[],
                                    page_size=2,
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                                    style_header={"fontWeight": "bold"},
                                ),
                            ])
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6("Clasificacion binaria", className="mb-2"),
                                dash_table.DataTable(
                                    id="tabla-top-bin",
                                    columns=COLUMNAS_TOP,
                                    data=[],
                                    page_size=2,
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                                    style_header={"fontWeight": "bold"},
                                ),
                            ])
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6("Clasificacion multiclase", className="mb-2"),
                                dash_table.DataTable(
                                    id="tabla-top-multi",
                                    columns=COLUMNAS_TOP,
                                    data=[],
                                    page_size=2,
                                    sort_action="native",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                                    style_header={"fontWeight": "bold"},
                                ),
                            ])
                        ),
                        md=4,
                    ),
                ], className="mb-3"),
                type="default",
            ),

            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-metricas-reg", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="grafica-metricas-bin", figure=go.Figure()), md=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-metricas-multi", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="grafica-loss", figure=go.Figure()), md=6),
            ]),

            dbc.Card([
                dbc.CardBody([
                    html.H5("Interpretacion automatica", className="text-success fw-bold"),
                    html.P(id="texto-interpretacion", className="mb-0")
                ])
            ], className="mt-3 border-success"),
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardBody([
            html.H4("Simulador de escenarios TIC", className="mb-3"),
            dbc.Alert(
                [
                    html.Strong("Regresion: "),
                    "predice el puntaje de ingles esperado. ",
                    html.Strong("Clasificacion: "),
                    "estima la probabilidad de B1+ (binaria) y el nivel A-/A1/A2/B1/B+ (multiclase).",
                ],
                color="info",
                className="shadow-sm",
            ),
            dbc.Alert(
                id="alerta-modelos-p3",
                color="danger",
                is_open=False,
                className="mb-3",
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario A", className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dropdown("a-internet", "Acceso a Internet", ["Si", "No"], "Si"),
                                _dropdown("a-computador", "Acceso a Computador", ["Si", "No"], "Si"),
                            ]),
                            dbc.Row([
                                _dropdown("a-bilingue", "Colegio bilingue", ["Si", "No"], "No"),
                                _dropdown("a-estrato", "Estrato vivienda", opciones_sim["estrato"], _default(opciones_sim["estrato"], "ESTRATO 1")),
                            ]),
                            dbc.Row([
                                _dropdown("a-zona", "Zona de residencia", opciones_sim["zona"], _default(opciones_sim["zona"], "URBANO")),
                                _dropdown("a-tipo", "Tipo de colegio", opciones_sim["tipo_colegio"], _default(opciones_sim["tipo_colegio"], "PUBLICO")),
                            ]),
                            dbc.Row([
                                _dropdown("a-jornada", "Jornada", opciones_sim["jornada"], _default(opciones_sim["jornada"], "COMPLETA")),
                                _dropdown("a-genero", "Genero colegio", opciones_sim["genero"], _default(opciones_sim["genero"], "MIXTO")),
                            ]),
                            dbc.Row([
                                _dropdown("a-edu-padre", "Educacion padre", opciones_sim["edu_padre"], _default(opciones_sim["edu_padre"], "SECUNDARIA")),
                                _dropdown("a-edu-madre", "Educacion madre", opciones_sim["edu_madre"], _default(opciones_sim["edu_madre"], "SECUNDARIA")),
                            ]),
                        ])
                    ], className="shadow-sm"),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Escenario B", className="fw-bold bg-light"),
                        dbc.CardBody([
                            dbc.Row([
                                _dropdown("b-internet", "Acceso a Internet", ["Si", "No"], "No"),
                                _dropdown("b-computador", "Acceso a Computador", ["Si", "No"], "No"),
                            ]),
                            dbc.Row([
                                _dropdown("b-bilingue", "Colegio bilingue", ["Si", "No"], "No"),
                                _dropdown("b-estrato", "Estrato vivienda", opciones_sim["estrato"], _default(opciones_sim["estrato"], "ESTRATO 1")),
                            ]),
                            dbc.Row([
                                _dropdown("b-zona", "Zona de residencia", opciones_sim["zona"], _default(opciones_sim["zona"], "URBANO")),
                                _dropdown("b-tipo", "Tipo de colegio", opciones_sim["tipo_colegio"], _default(opciones_sim["tipo_colegio"], "PUBLICO")),
                            ]),
                            dbc.Row([
                                _dropdown("b-jornada", "Jornada", opciones_sim["jornada"], _default(opciones_sim["jornada"], "COMPLETA")),
                                _dropdown("b-genero", "Genero colegio", opciones_sim["genero"], _default(opciones_sim["genero"], "MIXTO")),
                            ]),
                            dbc.Row([
                                _dropdown("b-edu-padre", "Educacion padre", opciones_sim["edu_padre"], _default(opciones_sim["edu_padre"], "SECUNDARIA")),
                                _dropdown("b-edu-madre", "Educacion madre", opciones_sim["edu_madre"], _default(opciones_sim["edu_madre"], "SECUNDARIA")),
                            ]),
                        ])
                    ], className="shadow-sm"),
                ], md=6),
            ], className="mb-3"),

            dbc.Button("Comparar escenarios", id="btn-simular", color="success", className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Small("Puntaje ingles (A)", className="text-muted"),
                    html.H4(id="out-puntaje-a", children="—"),
                    html.Small("Probabilidad B1+ (A)", className="text-muted"),
                    html.H5(id="out-proba-a", children="—"),
                    html.Small("Nivel multiclase (A)", className="text-muted"),
                    html.H6(id="out-nivel-a", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Puntaje ingles (B)", className="text-muted"),
                    html.H4(id="out-puntaje-b", children="—"),
                    html.Small("Probabilidad B1+ (B)", className="text-muted"),
                    html.H5(id="out-proba-b", children="—"),
                    html.Small("Nivel multiclase (B)", className="text-muted"),
                    html.H6(id="out-nivel-b", children="—"),
                ], md=4),
                dbc.Col([
                    html.Small("Delta puntaje (B - A)", className="text-muted"),
                    html.H4(id="out-delta-puntaje", children="—"),
                    html.Small("Delta probabilidad (B - A)", className="text-muted"),
                    html.H5(id="out-delta-proba", children="—"),
                ], md=4),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="grafica-escenarios-reg", figure=go.Figure()), md=6),
                dbc.Col(dcc.Graph(id="grafica-escenarios-clf", figure=go.Figure()), md=6),
            ]),
        ])
    ], className="mb-4 shadow-sm"),

    dbc.Card([
        dbc.CardBody([
            html.H4("Significancia de variables (OLS)", className="mb-2"),
            html.P(
                "Selecciona variables independientes para la regresion. El panel usa pruebas t y F para "
                "evaluar significancia estadistica."
            ),
            html.Small(
                "Si una variable queda excluida y el p-value es < 0.05, se considera significativa y su omision reduce el ajuste.",
                className="text-muted d-block mb-2",
            ),
            dcc.Store(id="ols-run-map", data={}),
            html.Label("Corrida de regresion (MLflow)", className="fw-bold small"),
            dcc.Dropdown(
                id="ols-run-selector",
                options=[],
                value=None,
                clearable=True,
                placeholder="Selecciona una corrida para cargar variables",
                className="mb-2",
            ),
            html.Div(id="texto-ols-corrida", className="text-muted mb-2"),
            dcc.Dropdown(
                id="variables-ols",
                options=VARIABLES_P3,
                value=[v["value"] for v in VARIABLES_P3],
                multi=True,
                className="mb-2",
            ),
            dbc.Button(
                "Calcular significancia",
                id="btn-significancia",
                color="primary",
                className="mb-3",
            ),
            html.Div(id="texto-significancia", className="text-muted mb-2"),
            dbc.Row([
                dbc.Col([
                    html.H6("Pruebas t (variables numericas)", className="mb-2"),
                    dash_table.DataTable(
                        id="tabla-ttest",
                        columns=COLUMNAS_TTEST,
                        data=[],
                        page_size=6,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ], md=6),
                dbc.Col([
                    html.H6("Pruebas F (significancia por variable)", className="mb-2"),
                    dash_table.DataTable(
                        id="tabla-ftest",
                        columns=COLUMNAS_FTEST,
                        data=[],
                        page_size=8,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                        style_header={"fontWeight": "bold"},
                    ),
                ], md=6),
            ]),
        ])
    ], className="mb-4 shadow-sm"),

    html.Hr(),
    html.H4("Analisis descriptivo", className="mt-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Filtrar Analisis por Municipio:", className="fw-bold"),
            dcc.Dropdown(
                id="filtro-municipio",
                options=[{"label": m, "value": m} for m in lista_municipios],
                value="TODOS",
                clearable=False,
                className="mb-3 shadow-sm",
            ),
        ], md=4)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id="grafica-mapa")])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id="grafica-histograma")])
            ], className="mb-4 shadow-sm")
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(id="texto-probabilidad", className="text-center text-primary mb-3 fw-bold"),
                    dcc.Graph(id="grafica-dispersion"),
                ])
            ], className="mb-4 shadow-sm")
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=ranking_estatico)])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="grafica-tiempo")])), md=12)
    ], className="mb-4"),
], fluid=True)

@callback(
    [Output('grafica-mapa', 'figure'),
     Output('grafica-histograma', 'figure'),
     Output('grafica-dispersion', 'figure'),
     Output('texto-probabilidad', 'children'),
     Output('grafica-tiempo', 'figure')],
    [Input('filtro-municipio', 'value')]
)
def actualizar_tablero(municipio_seleccionado):
    mapa = generar_mapa_antioquia(df_p3, municipio_seleccionado)
    histograma = generar_histograma_tic(df_p3, municipio_seleccionado)
    dispersion = generar_dispersion_clusters(df_p3, municipio_seleccionado)
    
    probabilidad_z = calcular_probabilidad_b1(df_p3, municipio_seleccionado)
    texto_insight = f"Insight: El acceso a internet altera la probabilidad de alcanzar nivel B1/B+ en un {probabilidad_z}%"
    # Serie temporal (por acceso TIC)
    serie = generar_serie_tic_ingles_por_periodo(df_p3, municipio_seleccionado)

    return mapa, histograma, dispersion, texto_insight, serie


@callback(
    [
        Output("tabla-modelos", "data"),
        Output("tabla-top-reg", "data"),
        Output("tabla-top-bin", "data"),
        Output("tabla-top-multi", "data"),
        Output("grafica-metricas-reg", "figure"),
        Output("grafica-metricas-bin", "figure"),
        Output("grafica-metricas-multi", "figure"),
        Output("grafica-loss", "figure"),
        Output("texto-modelos", "children"),
        Output("texto-interpretacion", "children"),
        Output("texto-entrenamiento", "children"),
        Output("ols-run-selector", "options"),
        Output("ols-run-selector", "value"),
        Output("ols-run-map", "data"),
    ],
    [Input("btn-cargar-resultados", "n_clicks"), Input("btn-entrenar-variables", "n_clicks")],
    [State("variables-entrenar", "value")],
    prevent_initial_call=True
)
def cargar_resultados(n_clicks_cargar, n_clicks_entrenar, variables_entrenar):
    if not (n_clicks_cargar or n_clicks_entrenar):
        return [], [], [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), "", "", "", [], None, {}

    trigger = dash.ctx.triggered_id
    mensaje_entrenamiento = ""

    if trigger == "btn-entrenar-variables":
        resumen, history, mejores, interpretacion, mensaje = entrenar_modelos_p3_personalizados(
            df_p3,
            variables_entrenar,
        )
        if not resumen.empty:
            resumen = resumen.copy()
            resumen["selected_vars"] = ",".join(variables_entrenar or [])
        mensaje_entrenamiento = mensaje
    else:
        resumen, mensaje = cargar_resultados_mlflow_p3()
        history = cargar_historial_mlflow_p3(resumen)
        mejores = seleccionar_mejores_modelos_resultados(resumen)
        interpretacion = generar_interpretacion_p3(resumen, mejores, df_p3) if not resumen.empty else ""

    fig_reg, fig_bin, fig_multi, fig_loss = construir_figuras_comparativas(resumen, history)

    top_df = seleccionar_top_modelos_resultados(resumen, top_n=2)
    top_df = _enriquecer_top_df(top_df)
    top_reg = top_df[top_df["task"] == "regresion"] if not top_df.empty else pd.DataFrame()
    top_bin = top_df[top_df["task"] == "clasificacion_binaria"] if not top_df.empty else pd.DataFrame()
    top_multi = top_df[top_df["task"] == "clasificacion_multiclase"] if not top_df.empty else pd.DataFrame()

    reg_runs = resumen[resumen["task"] == "regresion"] if not resumen.empty else pd.DataFrame()
    ols_options = []
    ols_map = {}
    if not reg_runs.empty:
        for _, row in reg_runs.iterrows():
            config_id = row.get("config_id")
            if not config_id:
                continue
            label = f"{config_id} ({row.get('model')})"
            ols_options.append({"label": label, "value": config_id})
            ols_map[config_id] = {
                "selected_vars": row.get("selected_vars"),
                "feature_set": row.get("feature_set"),
            }

    ols_value = None
    if not top_reg.empty:
        ols_value = top_reg.iloc[0].get("config_id")
    elif ols_options:
        ols_value = ols_options[0]["value"]

    if resumen.empty:
        return [], [], [], [], fig_reg, fig_bin, fig_multi, fig_loss, mensaje, interpretacion, mensaje_entrenamiento, ols_options, ols_value, ols_map

    return (
        resumen.to_dict("records"),
        top_reg.to_dict("records"),
        top_bin.to_dict("records"),
        top_multi.to_dict("records"),
        fig_reg,
        fig_bin,
        fig_multi,
        fig_loss,
        mensaje,
        interpretacion,
        mensaje_entrenamiento,
        ols_options,
        ols_value,
        ols_map,
    )


@callback(
    Output("estado-mlflow", "children"),
    [Input("btn-iniciar-mlflow", "n_clicks"), Input("btn-cargar-resultados", "n_clicks")],
    prevent_initial_call=True,
)
def actualizar_estado_mlflow(n_clicks_iniciar, n_clicks_cargar):
    trigger = dash.ctx.triggered_id
    if trigger == "btn-iniciar-mlflow":
        ok, mensaje = iniciar_mlflow_ui()
    else:
        ok, mensaje = verificar_mlflow_ui()
    return mensaje


@callback(
    [
        Output("variables-ols", "value"),
        Output("texto-ols-corrida", "children"),
    ],
    [Input("ols-run-selector", "value")],
    [State("ols-run-map", "data")],
    prevent_initial_call=True,
)
def cargar_variables_ols_por_corrida(run_id, run_map):
    if not run_id or not run_map:
        return no_update, ""

    info = run_map.get(run_id) or {}
    selected = _infer_selected_vars(info.get("selected_vars"), info.get("feature_set"))
    if not selected:
        return no_update, f"No se encontraron variables guardadas para la corrida {run_id}."

    return selected, f"Variables cargadas desde la corrida {run_id}."


@callback(
    [
        Output("out-puntaje-a", "children"),
        Output("out-proba-a", "children"),
        Output("out-nivel-a", "children"),
        Output("out-puntaje-b", "children"),
        Output("out-proba-b", "children"),
        Output("out-nivel-b", "children"),
        Output("out-delta-puntaje", "children"),
        Output("out-delta-proba", "children"),
        Output("grafica-escenarios-reg", "figure"),
        Output("grafica-escenarios-clf", "figure"),
        Output("alerta-modelos-p3", "children"),
        Output("alerta-modelos-p3", "is_open"),
    ],
    [Input("btn-simular", "n_clicks")],
    [
        State("a-internet", "value"),
        State("a-computador", "value"),
        State("a-bilingue", "value"),
        State("a-estrato", "value"),
        State("a-zona", "value"),
        State("a-tipo", "value"),
        State("a-jornada", "value"),
        State("a-genero", "value"),
        State("a-edu-padre", "value"),
        State("a-edu-madre", "value"),
        State("b-internet", "value"),
        State("b-computador", "value"),
        State("b-bilingue", "value"),
        State("b-estrato", "value"),
        State("b-zona", "value"),
        State("b-tipo", "value"),
        State("b-jornada", "value"),
        State("b-genero", "value"),
        State("b-edu-padre", "value"),
        State("b-edu-madre", "value"),
    ],
    prevent_initial_call=True,
)
def simular_escenarios(
    n_clicks,
    a_internet,
    a_computador,
    a_bilingue,
    a_estrato,
    a_zona,
    a_tipo,
    a_jornada,
    a_genero,
    a_edu_padre,
    a_edu_madre,
    b_internet,
    b_computador,
    b_bilingue,
    b_estrato,
    b_zona,
    b_tipo,
    b_jornada,
    b_genero,
    b_edu_padre,
    b_edu_madre,
):
    if not n_clicks:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    valores_a = {
        "internet": a_internet,
        "computador": a_computador,
        "bilingue": a_bilingue,
        "estrato": a_estrato,
        "zona": a_zona,
        "tipo_colegio": a_tipo,
        "jornada": a_jornada,
        "genero": a_genero,
        "edu_padre": a_edu_padre,
        "edu_madre": a_edu_madre,
    }
    valores_b = {
        "internet": b_internet,
        "computador": b_computador,
        "bilingue": b_bilingue,
        "estrato": b_estrato,
        "zona": b_zona,
        "tipo_colegio": b_tipo,
        "jornada": b_jornada,
        "genero": b_genero,
        "edu_padre": b_edu_padre,
        "edu_madre": b_edu_madre,
    }

    pred_a, pred_b, fig_reg, fig_clf, error = predecir_escenarios_p3(valores_a, valores_b)
    if error:
        return (
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            go.Figure(),
            go.Figure(),
            error,
            True,
        )

    delta_puntaje = pred_b["puntaje"] - pred_a["puntaje"]
    delta_proba = pred_b["proba_b1"] - pred_a["proba_b1"]

    return (
        f"{pred_a['puntaje']:.1f}",
        f"{pred_a['proba_b1'] * 100:.1f}%",
        pred_a["nivel"],
        f"{pred_b['puntaje']:.1f}",
        f"{pred_b['proba_b1'] * 100:.1f}%",
        pred_b["nivel"],
        f"{delta_puntaje:.1f}",
        f"{delta_proba * 100:.1f}%",
        fig_reg,
        fig_clf,
        "",
        False,
    )


@callback(
    [
        Output("tabla-ttest", "data"),
        Output("tabla-ftest", "data"),
        Output("texto-significancia", "children"),
    ],
    [Input("btn-significancia", "n_clicks")],
    [State("variables-ols", "value")],
    prevent_initial_call=True,
)
def calcular_significancia(n_clicks, variables):
    if not n_clicks:
        return [], [], ""

    resumen, ttest_df, ftest_df, mensaje = analizar_significancia_ols(df_p3, variables)
    if not resumen:
        return [], [], mensaje

    label_map = {v["value"]: v["label"] for v in VARIABLES_P3}
    if not ttest_df.empty:
        ttest_df["variable"] = ttest_df["variable"].map(label_map).fillna(ttest_df["variable"])
    if not ftest_df.empty:
        ftest_df["variable"] = ftest_df["variable"].map(label_map).fillna(ftest_df["variable"])

    texto = (
        f"OLS con {resumen.get('n_obs', 0)} observaciones | "
        f"R2={resumen.get('r2', 0):.3f} | Adj R2={resumen.get('adj_r2', 0):.3f}"
    )
    if ttest_df.empty:
        texto = f"{texto} | No hay variables numericas seleccionadas para t-test."
    if ftest_df.empty:
        texto = f"{texto} | Sin combinaciones disponibles para F-test."
    return ttest_df.to_dict("records"), ftest_df.to_dict("records"), texto