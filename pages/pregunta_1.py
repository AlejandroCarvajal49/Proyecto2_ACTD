"""
Página del tablero - Pregunta 1.

Responsable: Santiago Arias.

Contiene:
    - PARTE 1 (Proyecto 1): análisis descriptivo de la brecha urbano/rural.
    - PARTE 2 (Proyecto 2): simulador interactivo basado en redes neuronales
      (regresión + clasificación) para focalizar intervenciones.
"""

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from Analysis.logica_p1 import (
    # ---- Parte 1 (Proyecto 1) ----
    cargar_datos_p1,
    obtener_lista_municipios_p1,
    generar_boxplot_brecha,
    generar_dispersion_pib_brecha,
    calcular_estadisticas_brecha,
    generar_barras_brecha_error,
    generar_mapa_pib_puntaje,
    # ---- Parte 2 (Proyecto 2) ----
    cargar_artefactos_p2,
    obtener_opciones_categoricas_p2,
    obtener_pib_municipio,
    predecir_puntaje,
    predecir_prioridad,
    simular_contrafactual_zona,
    grafica_velocimetro_riesgo,
    grafica_comparacion_contrafactual,
    grafica_importancia_features,
    grafica_metricas_modelos,
    generar_interpretacion,
)

dash.register_page(__name__, path='/pregunta_1', name="Brecha Urbano/Rural")

# ============================================================================
# Carga única en el arranque
# ============================================================================
df_p1 = cargar_datos_p1()
lista_municipios = obtener_lista_municipios_p1(df_p1)
grafica_pib_estatica = generar_dispersion_pib_brecha(df_p1)

# Artefactos del Proyecto 2 (modelos)
artefactos = cargar_artefactos_p2()
opciones_cat = obtener_opciones_categoricas_p2(df_p1)
lista_municipios_simulador = [m for m in lista_municipios if m != 'TODOS']

# Gráficas estáticas de modelos
fig_metricas = grafica_metricas_modelos(artefactos)
fig_importancia = grafica_importancia_features(artefactos)


def _dropdown(id_, label, opciones, default=None):
    """Helper para construir dropdowns uniformes del simulador."""
    return dbc.Col([
        html.Label(label, className="fw-bold small"),
        dcc.Dropdown(
            id=id_,
            options=[{"label": o, "value": o} for o in opciones],
            value=default or (opciones[0] if opciones else None),
            clearable=False,
            className="mb-2 shadow-sm",
        ),
    ], md=6)


# ============================================================================
# LAYOUT
# ============================================================================

layout = dbc.Container([

    # ------------------------------------------------------------------------
    # ENCABEZADO
    # ------------------------------------------------------------------------
    html.H2("Pregunta 1: Brecha de Desempeño Urbano vs. Rural",
            className="my-4 fw-bold"),
    dbc.Alert(
        "Contexto del Ministerio: Identificar brechas críticas de desempeño entre "
        "zonas urbanas y rurales para focalizar recursos y programas de nivelación "
        "en municipios de menor PIB.",
        color="info",
        className="shadow-sm"
    ),
    html.Hr(),

    # ========================================================================
    # ========================================================================
    # PARTE 1 - PROYECTO 1: ANÁLISIS DESCRIPTIVO
    # ========================================================================
    # ========================================================================

    html.H3("Análisis Descriptivo (Proyecto 1)",
            className="mt-4 mb-3 text-primary fw-bold"),

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
                    html.H5("Hallazgos e Insights (Prueba T-Student)",
                            className="card-title text-success fw-bold"),
                    html.P(id='texto-insight-p1', className="card-text fs-5")
                ])
            ], className="mb-4 shadow-sm border-success")
        ], md=12)
    ]),

    # Mapa interactivo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(id='grafica-mapa-p1')])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    # Boxplot y barras con error
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparación de Distribución (Urbano vs Rural)",
                               className="fw-bold bg-light"),
                dbc.CardBody([dcc.Graph(id='grafica-boxplot-p1')])
            ], className="mb-4 shadow-sm")
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Promedios con Desviación Estándar",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(id='grafica-barras-error-p1'),
                    html.Small(
                        "Nota: Las líneas sobre las barras indican la variabilidad "
                        "de los datos (Desviación Estándar).",
                        className="text-muted text-center d-block mt-2"
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], md=6)
    ]),

    # Dispersión del PIB
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Impacto del PIB en la Brecha Educativa (Global Departamental)",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(figure=grafica_pib_estatica),
                    html.Small(
                        "Nota: Valores positivos en Y indican ventaja urbana. "
                        "Muestra si los municipios más pobres sufren brechas más grandes.",
                        className="text-muted text-center d-block mt-2"
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], md=12)
    ]),

    # ========================================================================
    # ========================================================================
    # PARTE 2 - PROYECTO 2: MODELOS PREDICTIVOS
    # ========================================================================
    # ========================================================================

    html.Hr(className="my-5"),
    html.H3("Modelos Predictivos - Simulador (Proyecto 2)",
            className="mt-4 mb-3 text-primary fw-bold"),
    dbc.Alert(
        [html.Strong("Pregunta de negocio: "),
         "¿Existe una brecha significativa entre estudiantes rurales y urbanos que "
         "justifique intervención diferenciada en municipios con menor PIB? Este "
         "simulador combina dos redes neuronales para responder operativamente: una "
         "predice el puntaje global esperado (regresión) y otra clasifica si el "
         "perfil es de prioridad alta para focalización (clasificación)."],
        color="info", className="shadow-sm",
    ),

    # Banner de error si los modelos no están disponibles
    dbc.Alert(
        artefactos.get("error") or "",
        color="danger",
        is_open=not artefactos["disponible"],
        className="shadow-sm",
    ),

    # ----- Formulario + resultados numéricos -----
    dbc.Row([
        # Columna izquierda: formulario
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Perfil del estudiante / municipio",
                               className="fw-bold bg-light"),
                dbc.CardBody([

                    dbc.Row([
                        dbc.Col([
                            html.Label("Municipio (auto-pobla el PIB):",
                                       className="fw-bold small"),
                            dcc.Dropdown(
                                id="form-municipio",
                                options=[{"label": m, "value": m}
                                         for m in lista_municipios_simulador],
                                value=lista_municipios_simulador[0]
                                if lista_municipios_simulador else None,
                                clearable=False,
                                className="mb-2 shadow-sm",
                            ),
                        ], md=12),
                    ]),

                    dbc.Row([
                        _dropdown("form-area", "Zona de residencia",
                                  opciones_cat.get("Area", ["Urbano", "Rural"])),
                        _dropdown("form-estrato", "Estrato vivienda",
                                  opciones_cat.get("fami_estratovivienda", [])),
                    ]),

                    dbc.Row([
                        _dropdown("form-edu-padre", "Educación del padre",
                                  opciones_cat.get("fami_educacionpadre", [])),
                        _dropdown("form-edu-madre", "Educación de la madre",
                                  opciones_cat.get("fami_educacionmadre", [])),
                    ]),

                    dbc.Row([
                        _dropdown("form-naturaleza", "Naturaleza del colegio",
                                  opciones_cat.get("cole_naturaleza", [])),
                        _dropdown("form-jornada", "Jornada",
                                  opciones_cat.get("cole_jornada", [])),
                    ]),

                    dbc.Row([
                        _dropdown("form-genero-colegio", "Género del colegio",
                                  opciones_cat.get("cole_genero", [])),
                        dbc.Col([
                            html.Label("PIB municipal (miles de millones):",
                                       className="fw-bold small"),
                            dcc.Input(id="form-pib", type="number", step=0.1,
                                      className="form-control mb-2 shadow-sm"),
                        ], md=6),
                    ]),

                    html.Div([
                        dbc.Button("Ejecutar modelos predictivos",
                                   id="btn-predecir",
                                   color="primary", size="lg",
                                   className="w-100 mt-2 shadow-sm",
                                   disabled=not artefactos["disponible"]),
                    ]),
                ]),
            ], className="shadow-sm"),
        ], md=6),

        # Columna derecha: resultados numéricos + velocímetro
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Resultados de la predicción",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Puntaje global esperado",
                                     className="text-muted small"),
                            html.H2(id="out-puntaje", children="—",
                                    className="text-primary fw-bold"),
                        ], md=6),
                        dbc.Col([
                            html.Div("Etiqueta de prioridad",
                                     className="text-muted small"),
                            html.H4(id="out-etiqueta", children="—",
                                    className="fw-bold"),
                        ], md=6),
                    ], className="mb-3"),

                    dcc.Graph(id="grafica-velocimetro",
                              figure=grafica_velocimetro_riesgo(None)),
                ]),
            ], className="shadow-sm mb-3"),
        ], md=6),
    ]),

    # ----- Interpretación en lenguaje natural -----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Lectura para el Ministerio",
                            className="card-title text-success fw-bold"),
                    html.P(id="out-interpretacion",
                           children="Configure un perfil y ejecute los modelos.",
                           className="card-text fs-5"),
                ]),
            ], className="mb-4 shadow-sm border-success"),
        ], md=12),
    ]),

    # ----- Contrafactual y métricas -----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Contrafactual: efecto de la zona geográfica",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(id="grafica-contrafactual",
                              figure=grafica_comparacion_contrafactual(None)),
                    html.Small(
                        "Mantiene fijo el perfil socioeconómico y cambia únicamente "
                        "la zona. Cuantifica el efecto puro del entorno geográfico "
                        "predicho por el modelo.",
                        className="text-muted d-block mt-2",
                    ),
                ]),
            ], className="mb-4 shadow-sm"),
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Desempeño de los modelos",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(figure=fig_metricas),
                ]),
            ], className="mb-4 shadow-sm"),
        ], md=6),
    ]),

    # ----- Importancia de variables -----
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("¿Qué variables empujan más el puntaje?",
                               className="fw-bold bg-light"),
                dbc.CardBody([
                    dcc.Graph(figure=fig_importancia),
                    html.Small(
                        "Magnitud promedio de los pesos de la primera capa de la red "
                        "de regresión, agregados por feature de entrada. Sirve como "
                        "aproximación de importancia relativa.",
                        className="text-muted d-block mt-2",
                    ),
                ]),
            ], className="mb-4 shadow-sm"),
        ], md=12),
    ]),

], fluid=True)


# ============================================================================
# CALLBACKS - PARTE 1 (Proyecto 1)
# ============================================================================

@callback(
    [Output('grafica-boxplot-p1', 'figure'),
     Output('grafica-barras-error-p1', 'figure'),
     Output('grafica-mapa-p1', 'figure'),
     Output('texto-insight-p1', 'children')],
    [Input('filtro-municipio-p1', 'value')]
)
def actualizar_tablero_p1(municipio_seleccionado):
    boxplot = generar_boxplot_brecha(df_p1, municipio_seleccionado)
    barras_error = generar_barras_brecha_error(df_p1, municipio_seleccionado)
    mapa = generar_mapa_pib_puntaje(df_p1, municipio_seleccionado)
    texto_insight = calcular_estadisticas_brecha(df_p1, municipio_seleccionado)
    return boxplot, barras_error, mapa, texto_insight


# ============================================================================
# CALLBACKS - PARTE 2 (Proyecto 2: Modelos Predictivos)
# ============================================================================

@callback(
    Output("form-pib", "value"),
    Input("form-municipio", "value"),
)
def autopoblar_pib(municipio):
    """Cuando el usuario cambia el municipio, auto-rellena el PIB."""
    if not municipio:
        return no_update
    pib, _per_cap = obtener_pib_municipio(df_p1, municipio)
    return round(pib, 2) if pib is not None else no_update


@callback(
    [Output("out-puntaje", "children"),
     Output("out-etiqueta", "children"),
     Output("out-etiqueta", "className"),
     Output("grafica-velocimetro", "figure"),
     Output("grafica-contrafactual", "figure"),
     Output("out-interpretacion", "children")],
    [Input("btn-predecir", "n_clicks")],
    [State("form-municipio", "value"),
     State("form-area", "value"),
     State("form-estrato", "value"),
     State("form-edu-padre", "value"),
     State("form-edu-madre", "value"),
     State("form-naturaleza", "value"),
     State("form-jornada", "value"),
     State("form-genero-colegio", "value"),
     State("form-pib", "value")],
    prevent_initial_call=True,
)
def ejecutar_modelos(n_clicks, municipio, area, estrato, edu_padre, edu_madre,
                     naturaleza, jornada, genero_col, pib):
    """Toma el formulario, ejecuta regresión + clasificación + contrafactual."""
    if not artefactos["disponible"]:
        return ("—", "Modelos no disponibles", "fw-bold text-danger",
                no_update, no_update,
                "Los modelos aún no han sido entrenados. "
                "Ejecute el notebook de entrenamiento.")

    # PIB per cápita derivado del municipio (no lo pedimos al usuario)
    _pib_m, pib_per_cap = obtener_pib_municipio(df_p1, municipio)

    valores_form = {
        "Area": area,
        "fami_estratovivienda": estrato,
        "fami_educacionpadre": edu_padre,
        "fami_educacionmadre": edu_madre,
        "cole_naturaleza": naturaleza,
        "cole_jornada": jornada,
        "cole_genero": genero_col,
        "PIB miles de millones": pib,
        "PIB per capita": pib_per_cap,
    }

    puntaje = predecir_puntaje(artefactos, valores_form)
    proba, etiqueta = predecir_prioridad(artefactos, valores_form)
    contrafactual = simular_contrafactual_zona(artefactos, valores_form)

    color_etiq = ("fw-bold text-danger" if proba is not None and proba >= 0.5
                  else "fw-bold text-success")

    fig_gauge = grafica_velocimetro_riesgo(proba)
    fig_cf = grafica_comparacion_contrafactual(contrafactual)
    texto = generar_interpretacion(puntaje, proba, contrafactual)

    return (
        f"{puntaje:.1f}" if puntaje is not None else "—",
        etiqueta or "—",
        color_etiq,
        fig_gauge,
        fig_cf,
        texto,
    )