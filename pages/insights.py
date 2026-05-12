import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Importar la función desde tu archivo de lógica
from Analysis.logica_insights import obtener_figuras_eda, build_bar_with_comparisons
from dash import Input, Output

dash.register_page(__name__, path='/insights', name="Insights Generales")

# Traer KPIs, figuras y datos auxiliares
kp, figs, aux = obtener_figuras_eda()

# Helper para crear una tarjeta KPI
def _kpi_card(title, value, md=3):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.H6(title, className="card-title text-muted"),
        html.H4(f"{value}", className="card-text")
    ]), className="mb-3 shadow-sm"), md=md)

layout = dbc.Container([
    html.H2("Insights Generales - Exploración y Análisis", className="my-4"),
    html.Hr(),

    # Serie temporal - aparecer primero (puntaje global)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('serie_punt_global_por_periodo'))])), md=12)
    ], className="mb-4"),

    # KPIs en la parte superior
    dbc.Row([
        _kpi_card("Máximo puntaje global", round(kp.get("max_punt_global", 0), 2) if kp.get("max_punt_global") is not None else "N/A"),
        _kpi_card("Mínimo puntaje global", round(kp.get("min_punt_global", 0), 2) if kp.get("min_punt_global") is not None else "N/A"),
        _kpi_card("Media puntaje global", round(kp.get("mean_punt_global", 0), 2) if kp.get("mean_punt_global") is not None else "N/A"),
        _kpi_card("% > 300 (global)", f"{round(kp.get('pct_over_300', 0),2)}%" if kp.get("pct_over_300") is not None else "N/A")
    ], className="mb-4"),

    # Histogramas y pie
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('hist_global'))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('pie_genero'))])), md=6)
    ], className="mb-4"),

    # Histogramas por categoría
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('hist_by_area'))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('hist_by_genero'))])), md=6)
    ], className="mb-4"),

    # Boxplot y barra con selector de métrica (interactive)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get('box_global_by_category'))])), md=6),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                dcc.Dropdown(
                    id='metric-select',
                    options=[{"label": m, "value": m} for m in aux.get('metrics_list', [])],
                    value=(aux.get('metrics_list', [None])[0] if aux.get('metrics_list') else None),
                    clearable=False
                ),
                dcc.Graph(id='bar-estrato-graph', figure=figs.get('bar_by_estrato_metric_select')),
                html.Div(id='bar-estrato-kpis')
            ]))
        ], md=6)
    ], className="mb-4")
], fluid=True)


# Callback para actualizar la barra por estrato y mostrar KPIs comparativos
@dash.callback(
    Output('bar-estrato-graph', 'figure'),
    Output('bar-estrato-kpis', 'children'),
    Input('metric-select', 'value')
)
def _update_bar_and_kpis(selected_metric):
    df = aux.get('df')
    estrato_col = aux.get('detected', {}).get('col_estrato')
    metrics_list = aux.get('metrics_list', [])
    if df is None or selected_metric is None or estrato_col is None:
        return go.Figure(), html.Div("No hay datos o columna de estrato detectada.")

    fig, kpis_bar = build_bar_with_comparisons(df, selected_metric, estrato_col, metrics_list)

    # Construir representación de KPIs
    children = []
    children.append(html.H6(f"KPIs para {selected_metric}"))
    children.append(html.Ul([
        html.Li(f"Rango entre estratos: {round(kpis_bar.get('range', 0),2)}"),
    ]))

    # Comparaciones vs otras métricas
    comps = kpis_bar.get('comparisons_percent_vs_other', {})
    if comps:
        children.append(html.H6("Comparaciones vs otras métricas (porcentaje):"))
        items = []
        for other, val in comps.items():
            if val is None:
                items.append(html.Li(f"{other}: no calculable"))
            else:
                items.append(html.Li(f"{other}: {round(val,2)}% (positivo = {selected_metric} tiene mayor rango)"))
        children.append(html.Ul(items))

    return fig, dbc.Card(dbc.CardBody(children), className="mt-3")