import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from Analysis.logica_p3 import (
    cargar_datos_p3, 
    generar_mapa_antioquia, 
    generar_ranking_municipios_estatico,
    generar_histograma_tic, 
    generar_dispersion_clusters,
    calcular_probabilidad_b1,
    generar_serie_tic_ingles_por_periodo,
    obtener_lista_municipios
)

dash.register_page(__name__, path='/pregunta_3', name="Competitividad / Bilingüismo")

df_p3 = cargar_datos_p3()
lista_municipios = obtener_lista_municipios(df_p3)
ranking_estatico = generar_ranking_municipios_estatico(df_p3)

layout = dbc.Container([
    html.H2("Competitividad y Bilingüismo: Impacto TIC", className="my-4"),
    html.Hr(),
    
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