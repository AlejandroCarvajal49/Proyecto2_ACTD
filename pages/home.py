import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name="Inicio")

def crear_tarjeta_navegacion(titulo, descripcion, ruta, imagen):
    return dbc.Card([
        dbc.CardImg(src=imagen, top=True),  # Imagen de ejemplo, reemplazar con algo relevante
        dbc.CardBody([
            html.H4(titulo, className="card-title"),
            html.P(descripcion, className="card-text"),
            dbc.Button("Ingresar", href=ruta, color="primary", className="w-100"),
        ])
    ], class_name="shadow-sm h-100")

layout = dbc.Container([
    html.H1("Panel Principal - Resultados ICFES", className="text-center my-5"),
    
    dbc.Row([
        dbc.Col([
            crear_tarjeta_navegacion(
                "Insights Generales",
                "Resumen ejecutivo e interactivo de los hallazgos mas relevantes para la toma de decisiones.",
                "/insights",
                "assets/Gemini_Generated_Image_z38mdqz38mdqz38m.png"
            )
        ], md=6, className="mb-4"),
    ], justify="center"),
], fluid=True)