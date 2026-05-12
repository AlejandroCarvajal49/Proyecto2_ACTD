import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

# Importar la funcion desde tu archivo de logica
from Analysis.logica_insights import obtener_figuras_eda

dash.register_page(__name__, path="/insights", name="Insights Accionables")

# Traer KPIs, figuras y datos auxiliares
kp, figs, aux = obtener_figuras_eda()


def _fmt_percent(value):
    return f"{value:.2f}%" if value is not None else "N/A"


def _fmt_points(value):
    return f"{value:.2f}" if value is not None else "N/A"


def _kpi_card(title, value, kpi_id, message, md=3):
    card_id = f"{kpi_id}-card"
    return dbc.Col(
        [
            dbc.Card(
                dbc.CardBody([
                    html.H6(title, className="card-title text-muted"),
                    html.H4(f"{value}", className="card-text"),
                ]),
                className="mb-3 shadow-sm",
                id=card_id,
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader(title),
                    dbc.PopoverBody(message),
                ],
                target=card_id,
                trigger="click",
                placement="bottom",
            ),
        ],
        md=md,
    )


def _segment_table(title, data):
    if not data:
        return dbc.Card(dbc.CardBody([html.H6(title), html.Div("N/A")]), className="mb-3")

    rows = [{"Segmento": k, "Tasa riesgo (%)": round(v, 2)} for k, v in data.items()]
    df = pd.DataFrame(rows)
    table = dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True, size="sm")
    return dbc.Card(dbc.CardBody([html.H6(title), table]), className="mb-3")


impacto_metric = kp.get("impacto_estimado_publico_privado_metric")
impacto_gap = kp.get("impacto_estimado_publico_privado_max")
impacto_text = f"{impacto_metric}: {impacto_gap:.2f}" if impacto_metric and impacto_gap is not None else "N/A"

layout = dbc.Container([
    html.H2("Insights Accionables - Brechas y Priorizacion", className="my-4"),
    html.P("Resultados orientados a focalizacion de recursos y cierre de brechas territoriales."),
    html.Hr(),

    dbc.Row([
        _kpi_card(
            "Tasa de riesgo departamental",
            _fmt_percent(kp.get("tasa_riesgo_departamental")),
            "kpi-riesgo",
            "Porcentaje de estudiantes bajo el umbral P25 del puntaje global. Resume la magnitud del riesgo academico.",
        ),
        _kpi_card(
            "Brecha rural-urbana (P50)",
            _fmt_points(kp.get("brecha_rural_urbana_p50")),
            "kpi-brecha-rural",
            "Diferencia entre la mediana urbana y rural del puntaje global. Valores altos reflejan mayor desigualdad territorial.",
        ),
        _kpi_card(
            "Impacto estimado publico-privado (max)",
            impacto_text,
            "kpi-publico-privado",
            "Mayor brecha ajustada por estrato entre colegios privados y publicos. Se calcula con medias ajustadas.",
        ),
        _kpi_card(
            "Impacto digital en ingles",
            _fmt_points(kp.get("impacto_digital_ingles_gap")),
            "kpi-digital-ingles",
            "Diferencia mediana en ingles entre estudiantes con y sin acceso digital.",
        ),
    ], className="mb-4"),

    dbc.Row([
        _kpi_card(
            "Municipios prioridad alta",
            kp.get("municipios_prioridad_alta", "N/A"),
            "kpi-municipios",
            "Municipios con brecha rural-urbana alta y PIB bajo. Son focos de intervencion prioritaria.",
            md=3,
        ),
        _kpi_card(
            "Umbral de riesgo (P25)",
            _fmt_points(kp.get("umbral_riesgo")),
            "kpi-umbral",
            "Puntaje global que define riesgo alto (percentil 25). Sirve para clasificar prioridad.",
            md=3,
        ),
    ], className="mb-4"),

    html.H4("Q1. Equidad regional - Mapa de brechas", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_rural_urbana", go.Figure()))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_municipios_pib", go.Figure()))])), md=6),
    ], className="mb-4"),

    html.H4("Q2. Calidad publico vs privado - Impacto estimado", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_tipo_colegio", go.Figure()))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("impacto_estimado_publico_privado", go.Figure()))])), md=6),
    ], className="mb-4"),

    html.H4("Q3. Competitividad y bilinguismo - Brecha digital", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_digital_ingles", go.Figure()))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("potencial_mejora_digital_municipio", go.Figure()))])), md=6),
    ], className="mb-4"),

    html.H4("Diagnostico base - contexto general", className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_puntaje_global", go.Figure()))])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("mapa_brechas_genero", go.Figure()))])), md=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(figure=figs.get("serie_punt_global_por_periodo", go.Figure()))])), md=12),
    ], className="mb-4"),

    html.H4("Tasas de riesgo por segmento", className="mt-3"),
    dbc.Row([
        dbc.Col(_segment_table("Zona", aux.get("segment_risk", {}).get("ZONA")), md=4),
        dbc.Col(_segment_table("Tipo de colegio", aux.get("segment_risk", {}).get("TIPO_COLEGIO")), md=4),
        dbc.Col(_segment_table("Acceso digital", aux.get("segment_risk", {}).get("ACCESO_INTERNET")), md=4),
    ], className="mb-4"),
], fluid=True)