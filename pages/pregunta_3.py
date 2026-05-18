import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
from Analysis.logica_p3 import (
    cargar_datos_p3, 
    generar_mapa_antioquia, 
    generar_ranking_municipios_estatico,
    generar_histograma_tic, 
    generar_dispersion_clusters,
    calcular_probabilidad_b1,
    generar_serie_tic_ingles_por_periodo,
    obtener_lista_municipios,
    ejecutar_experimentos_mlflow_p3,
    obtener_mlflow_info,
    obtener_resumen_mlflow_p3,
    obtener_mejores_modelos_mlflow
)

dash.register_page(__name__, path='/pregunta_3', name="Competitividad / Bilingüismo")

df_p3 = cargar_datos_p3()
lista_municipios = obtener_lista_municipios(df_p3)
ranking_estatico = generar_ranking_municipios_estatico(df_p3)
mlflow_info = obtener_mlflow_info()

COLUMNAS_MODELOS = [
    {"name": "task", "id": "task"},
    {"name": "model", "id": "model"},
    {"name": "feature_set", "id": "feature_set"},
    {"name": "metric_name", "id": "metric_name"},
    {"name": "metric_value", "id": "metric_value"},
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

COLUMNAS_MLFLOW = [
    {"name": "experiment", "id": "experiment"},
    {"name": "task", "id": "task"},
    {"name": "model", "id": "model"},
    {"name": "feature_set", "id": "feature_set"},
    {"name": "rmse", "id": "rmse"},
    {"name": "mae", "id": "mae"},
    {"name": "r2", "id": "r2"},
    {"name": "accuracy", "id": "accuracy"},
    {"name": "f1", "id": "f1"},
    {"name": "roc_auc", "id": "roc_auc"},
    {"name": "f1_macro", "id": "f1_macro"},
    {"name": "log_loss", "id": "log_loss"},
    {"name": "status", "id": "status"},
    {"name": "start_time", "id": "start_time"},
    {"name": "run_id", "id": "run_id"},
]

COLUMNAS_MEJORES = [
    {"name": "task", "id": "task"},
    {"name": "model", "id": "model"},
    {"name": "feature_set", "id": "feature_set"},
    {"name": "metric", "id": "metric"},
    {"name": "metric_value", "id": "metric_value"},
    {"name": "run_id", "id": "run_id"},
]

layout = dbc.Container([
    html.H2("Competitividad y Bilingüismo: Impacto TIC", className="my-4"),
    html.Hr(),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Contexto y pregunta de negocio", className="mb-3"),
                    html.Ul([
                        html.Li(
                            "Contexto del usuario: medir el impacto de programas de bilingüismo y entender si la inversion en conectividad y herramientas TIC se relaciona con mejores resultados en ingles."
                        ),
                        html.Li(
                            "Pregunta de negocio: como se distribuye el nivel de desempeno en ingles (A-, A1, A2, B1, B+) a lo largo de Antioquia y que relacion tiene con el acceso a herramientas tecnologicas (Internet/Computador) en el hogar."
                        ),
                    ], className="mb-3"),

                    html.H4("Modelos propuestos", className="mb-3"),
                    html.Ul([
                        html.Li(
                            "Modelo de regresion: explicar puntaje de ingles en funcion de acceso a Internet, computador y variables de contexto del estudiante y del colegio."
                        ),
                        html.Li(
                            "Modelo de clasificacion: predecir la probabilidad de alcanzar un nivel alto de ingles; objetivo binario (B1 o superior) o multiclase (A-, A1, A2, B1, B+)."
                        ),
                    ], className="mb-3"),

                    html.H4("Justificacion y metodologia", className="mb-3"),
                    html.Ul([
                        html.Li(
                            "La regresion mide el efecto esperado de contar con Internet o computador en el hogar sobre el puntaje de ingles."
                        ),
                        html.Li(
                            "La clasificacion estima la probabilidad de llegar a un nivel alto, convirtiendo la relacion TIC-bilinguismo en una decision accionable."
                        ),
                        html.Li(
                            "Se probaran multiples configuraciones (tic_basico, tic_contexto, tic_contexto_municipio) con ingenieria de caracteristicas e imputacion de datos."
                        ),
                        html.Li(
                            "Modelos explorados: regresion lineal/regularizada, random forest y gradient boosting; clasificacion logistica y arboles."
                        ),
                        html.Li(
                            "Metricas: RMSE/MAE/R2 para regresion; Accuracy/F1/ROC-AUC (binaria) y F1 macro/log-loss (multiclase)."
                        ),
                    ]),
                    html.H5("Bibliografia sugerida", className="mt-3"),
                    html.Ul([
                        html.Li("Hastie, Tibshirani, Friedman - The Elements of Statistical Learning."),
                        html.Li("James, Witten, Hastie, Tibshirani - An Introduction to Statistical Learning."),
                        html.Li("Kuhn, Johnson - Applied Predictive Modeling."),
                    ]),
                ])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Experimentacion con MLflow", className="mb-3"),
                    html.P(
                        "Ejecuta las corridas para registrar los modelos. Luego abre MLflow con:"
                    ),
                    html.Pre("mlflow ui --backend-store-uri ./mlruns"),
                    dbc.Button(
                        "Ejecutar experimentos",
                        id="btn-ejecutar-modelos",
                        color="primary",
                        className="mb-3"
                    ),
                    html.Div(id="texto-modelos", className="text-muted mb-2"),
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
                        type="default"
                    ),
                ])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Estado MLflow y resumen", className="mb-3"),
                    html.Div([
                        html.Div([
                            html.Strong("Tracking URI: "),
                            html.Span(mlflow_info["tracking_uri"])
                        ]),
                        html.Div([
                            html.Strong("MLruns path: "),
                            html.Span(mlflow_info["mlruns_path"])
                        ]),
                        html.Div([
                            html.Strong("MLflow UI: "),
                            html.A(
                                mlflow_info["ui_url"],
                                href=mlflow_info["ui_url"],
                                target="_blank"
                            )
                        ]),
                    ], className="mb-2"),
                    dbc.Button(
                        "Actualizar resumen MLflow",
                        id="btn-refrescar-mlflow",
                        color="secondary",
                        className="mb-2"
                    ),
                    html.Div(id="texto-mlflow", className="text-muted mb-2"),
                    html.H6("Mejores modelos por tarea", className="mt-2"),
                    dcc.Loading(
                        dash_table.DataTable(
                            id="tabla-mejores-mlflow",
                            columns=COLUMNAS_MEJORES,
                            data=[],
                            page_size=6,
                            sort_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                            style_header={"fontWeight": "bold"}
                        ),
                        type="default"
                    ),
                    html.H6("Detalle de corridas", className="mt-3"),
                    dcc.Loading(
                        dash_table.DataTable(
                            id="tabla-mlflow",
                            columns=COLUMNAS_MLFLOW,
                            data=[],
                            page_size=8,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "textAlign": "left", "padding": "6px"},
                            style_header={"fontWeight": "bold"}
                        ),
                        type="default"
                    ),
                ])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Filtrar Análisis por Municipio:", className="fw-bold"),
            dcc.Dropdown(
                id='filtro-municipio',
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
                dbc.CardBody([dcc.Graph(id='grafica-mapa')])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='grafica-histograma')])
            ], className="mb-4 shadow-sm")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(id='texto-probabilidad', className="text-center text-primary mb-3 fw-bold"),
                    dcc.Graph(id='grafica-dispersion')
                ])
            ], className="mb-4 shadow-sm")
        ], md=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=ranking_estatico)])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ])

    ,

    # Serie temporal (última en esta página)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='grafica-tiempo')])), md=12)
    ], className="mb-4")
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
    [Output("tabla-modelos", "data"), Output("texto-modelos", "children")],
    [Input("btn-ejecutar-modelos", "n_clicks")],
    prevent_initial_call=True
)
def ejecutar_experimentos(n_clicks):
    if not n_clicks:
        return [], ""

    resumen, texto = ejecutar_experimentos_mlflow_p3(df_p3)
    if resumen.empty:
        return [], texto

    return resumen.to_dict("records"), texto


@callback(
    [Output("tabla-mlflow", "data"), Output("tabla-mejores-mlflow", "data"), Output("texto-mlflow", "children")],
    [Input("btn-refrescar-mlflow", "n_clicks"), Input("btn-ejecutar-modelos", "n_clicks")],
    prevent_initial_call=True
)
def actualizar_resumen_mlflow(n_clicks_refrescar, n_clicks_entrenar):
    resumen, mensaje = obtener_resumen_mlflow_p3()
    if resumen.empty:
        return [], [], mensaje

    mejores = obtener_mejores_modelos_mlflow(resumen)
    return resumen.to_dict("records"), mejores.to_dict("records"), mensaje