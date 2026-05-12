import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from Analysis.logica_p1 import (
    cargar_datos_p1,
    obtener_lista_municipios_p1,
    generar_boxplot_brecha,
    generar_dispersion_pib_brecha,
    calcular_estadisticas_brecha,
    generar_barras_brecha_error,
    generar_mapa_pib_puntaje # <-- IMPORTAMOS LA NUEVA FUNCIÓN
)

dash.register_page(__name__, path='/pregunta_1', name="Brecha Urbano/Rural")

# Carga de datos y gráficas estáticas
df_p1 = cargar_datos_p1()
lista_municipios = obtener_lista_municipios_p1(df_p1)
grafica_pib_estatica = generar_dispersion_pib_brecha(df_p1)

layout = dbc.Container([
    # Encabezado y Contexto
    html.H2("Pregunta 1: Brecha de Desempeño Urbano vs. Rural", className="my-4 fw-bold"),
    dbc.Alert(
        "Contexto del Ministerio: Identificar brechas críticas de desempeño entre zonas urbanas y rurales para focalizar recursos y programas de nivelación en municipios de menor PIB.",
        color="info",
        className="shadow-sm"
    ),
    html.Hr(),
    
    # Filtro
    dbc.Row([
        dbc.Col([
            html.Label("Focalizar Análisis por Municipio:", className="fw-bold"),
            dcc.Dropdown(
                id='filtro-municipio-p1',
                options=[{'label': m, 'value': m} for m in lista_municipios],
                value='TODOS',
                clearable=False,
                className="mb-3 shadow-sm"
            )
        ], md=4)
    ]),
    
    # Tarjeta de Insights y Estadísticas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Hallazgos e Insights (Prueba T-Student)", className="card-title text-success fw-bold"),
                    html.P(id='texto-insight-p1', className="card-text fs-5")
                ])
            ], className="mb-4 shadow-sm border-success")
        ], md=12)
    ]),

    # NUEVA FILA: Mapa Interactivo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='grafica-mapa-p1')])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),
    
    # Fila: Boxplot y Gráfico de Barras con Error
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparación de Distribución (Urbano vs Rural)", className="fw-bold bg-light"),
                dbc.CardBody([dcc.Graph(id='grafica-boxplot-p1')])
            ], className="mb-4 shadow-sm")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Promedios con Desviación Estándar", className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(id='grafica-barras-error-p1'),
                    html.Small(
                        "Nota: Las líneas sobre las barras indican la variabilidad de los datos (Desviación Estándar).", 
                        className="text-muted text-center d-block mt-2"
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], md=6)
    ]),

    # Fila: Dispersión del PIB
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Impacto del PIB en la Brecha Educativa (Global Departamental)", className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(figure=grafica_pib_estatica),
                    html.Small(
                        "Nota: Valores positivos en Y indican ventaja urbana. Muestra si los municipios más pobres sufren brechas más grandes.", 
                        className="text-muted text-center d-block mt-2"
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ])
], fluid=True)


# Callback actualizado para incluir el mapa
@callback(
    [Output('grafica-boxplot-p1', 'figure'),
     Output('grafica-barras-error-p1', 'figure'),
     Output('grafica-mapa-p1', 'figure'), # <-- NUEVO OUTPUT PARA EL MAPA
     Output('texto-insight-p1', 'children')],
    [Input('filtro-municipio-p1', 'value')]
)
def actualizar_tablero_p1(municipio_seleccionado):
    boxplot = generar_boxplot_brecha(df_p1, municipio_seleccionado)
    barras_error = generar_barras_brecha_error(df_p1, municipio_seleccionado)
    mapa = generar_mapa_pib_puntaje(df_p1, municipio_seleccionado) # <-- GENERAR EL MAPA
    texto_insight = calcular_estadisticas_brecha(df_p1, municipio_seleccionado)
    
    return boxplot, barras_error, mapa, texto_insight