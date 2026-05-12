#Inicio del proyecto 1 - Analitica Computacional para la toma de decisiones

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Usamos un tema de Bootstrap (LUX es limpio y profesional)
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.LUX])
server = app.server  # Necesario para despliegue en AWS/Gunicorn

# Navbar simple que siempre se ve arriba
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Inicio", href="/")),
        dbc.NavItem(dbc.NavLink("Pregunta 1", href="/pregunta_1")),
        dbc.NavItem(dbc.NavLink("Pregunta 2", href="/pregunta_2")),
        dbc.NavItem(dbc.NavLink("Pregunta 3", href="/pregunta_3")),
    ],
    brand="Analítica Saber 11 - Antioquia",
    brand_href="/",
    color="primary",
    dark=True,
)

app.layout = html.Div([
    navbar,
    dbc.Container([
        dash.page_container  # Aquí se renderizan las páginas de la carpeta /pages
    ], fluid=True, class_name="py-3")
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)