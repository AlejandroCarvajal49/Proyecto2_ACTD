import pandas as pd
import plotly.express as px

def cargar_datos_p3():
    df = pd.read_csv('Data/saber11_Antioquia_clean.csv', dtype={'cole_cod_mcpio_ubicacion': str})
    df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].str.upper().str.strip()
    
    df['Acceso_TIC'] = df.apply(
        lambda x: 'Internet y Computador' if x['fami_tieneinternet'] == 'Si' and x['fami_tienecomputador'] == 'Si'
        else ('Solo Internet' if x['fami_tieneinternet'] == 'Si'
              else ('Solo Computador' if x['fami_tienecomputador'] == 'Si' else 'Sin Acceso TIC')),
        axis=1
    )
    
    df_coord = pd.read_csv('Data/municipios_unicos.csv')
    df = pd.merge(df, df_coord, on='cole_mcpio_ubicacion', how='left')
    
    return df

def obtener_lista_municipios(df):
    municipios = df['cole_mcpio_ubicacion'].dropna().unique().tolist()
    municipios.sort()
    return ['TODOS'] + municipios

def generar_mapa_antioquia(df, municipio):
    dff = df.copy()
    
    df_mapa = dff.groupby(['cole_mcpio_ubicacion', 'lat', 'lon'])['punt_ingles'].mean().reset_index()
    min_ingles = df_mapa['punt_ingles'].min()
    max_ingles = df_mapa['punt_ingles'].max()
    
    if municipio != 'TODOS':
        df_mapa['opacidad'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 1.0 if x == municipio else 0.1)
        df_mapa['tamano'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 15 if x == municipio else 5)
    else:
        df_mapa['opacidad'] = 0.8
        df_mapa['tamano'] = 8

    fig = px.scatter_mapbox(
        df_mapa,
        lat='lat',
        lon='lon',
        color='punt_ingles',
        hover_name='cole_mcpio_ubicacion',
        color_continuous_scale='Viridis',
        range_color=[min_ingles, max_ingles],
        mapbox_style='carto-positron',
        zoom=6.0,
        center={"lat": 6.2518, "lon": -75.5636},
        title=f'Promedio de Puntaje en Inglés por Municipio ({municipio})',
        size='tamano',
        size_max=15
    )
    fig.update_traces(marker=dict(opacity=df_mapa['opacidad']))
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def generar_ranking_municipios_estatico(df):
    dff = df.copy()
    df_rank = dff.groupby('cole_mcpio_ubicacion', as_index=False)['punt_ingles'].mean()
    df_rank = df_rank.sort_values('punt_ingles', ascending=True)

    fig = px.bar(
        df_rank,
        x='punt_ingles',
        y='cole_mcpio_ubicacion',
        orientation='h',
        labels={'punt_ingles': 'Promedio Puntaje Inglés', 'cole_mcpio_ubicacion': 'Municipio'},
        title='Ranking de Municipios por Promedio de Puntaje en Inglés (Antioquia)',
        color='punt_ingles',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=900)
    return fig

def generar_histograma_tic(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.histogram(
        dff,
        x='desemp_ingles',
        color='Acceso_TIC',
        barmode='group',
        category_orders={"desemp_ingles": ["A-", "A1", "A2", "B1", "B+"]},
        title=f'Distribución del Nivel de Inglés vs. Acceso TIC ({municipio})',
        labels={'desemp_ingles': 'Nivel de Inglés', 'count': 'Frecuencia'},
        color_discrete_map={
            'Internet y Computador': '#2ca02c',
            'Solo Internet': '#1f77b4',
            'Solo Computador': '#ff7f0e',
            'Sin Acceso TIC': '#d62728'
        }
    )
    return fig

def generar_dispersion_regresion(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.scatter(
        dff,
        x='punt_ingles',
        y='punt_global',
        color='Acceso_TIC',
        trendline='ols',
        opacity=0.5,
        title=f'Regresión: Puntaje de Inglés vs Puntaje Global ICFES ({municipio})',
        labels={'punt_ingles': 'Puntaje Inglés', 'punt_global': 'Puntaje Global'}
    )
    return fig

def generar_dispersion_clusters(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    # Dicotomización estricta para clústeres rojo/verde
    dff['Tiene_Internet'] = dff['fami_tieneinternet'].apply(
        lambda x: 'Con Internet' if x == 'Si' else 'Sin Internet'
    )

    fig = px.scatter(
        dff,
        x='punt_ingles',
        y='punt_global',
        color='Tiene_Internet',
        opacity=0.6,
        marginal_x='box', # Añade visualización de densidad en los ejes
        marginal_y='box',
        title=f'Clusters de Desempeño: Con vs Sin Internet ({municipio})',
        labels={'punt_ingles': 'Puntaje Inglés', 'punt_global': 'Puntaje Global'},
        color_discrete_map={
            'Con Internet': '#2ca02c', # Verde
            'Sin Internet': '#d62728'  # Rojo
        }
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def calcular_probabilidad_b1(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    con_internet = dff[dff['fami_tieneinternet'] == 'Si']
    sin_internet = dff[dff['fami_tieneinternet'] == 'No']

    if len(con_internet) == 0 or len(sin_internet) == 0:
        return 0.0

    # Cálculo de probabilidad marginal P(B1 U B+ | Internet)
    prob_con = len(con_internet[con_internet['desemp_ingles'].isin(['B1', 'B+'])]) / len(con_internet) * 100
    prob_sin = len(sin_internet[sin_internet['desemp_ingles'].isin(['B1', 'B+'])]) / len(sin_internet) * 100

    # Diferencial de probabilidad (Z%)
    diferencia_z = prob_con - prob_sin
    return round(diferencia_z, 2)


def generar_serie_tic_ingles_por_periodo(df, municipio='TODOS'):
    """Genera una serie temporal del promedio de `punt_ingles` por año,
    separada por categorías de `Acceso_TIC`.

    Se asume que `df` tiene la columna `periodo` y se usan los primeros 4
    caracteres como año (ej: '2019-1' -> 2019).
    """
    dff = df.copy()
    # Asegurar que exista la columna Acceso_TIC (si no, derivarla)
    if 'Acceso_TIC' not in dff.columns:
        dff['Acceso_TIC'] = dff.apply(
            lambda x: 'Internet y Computador' if x.get('fami_tieneinternet') == 'Si' and x.get('fami_tienecomputador') == 'Si'
            else ('Solo Internet' if x.get('fami_tieneinternet') == 'Si'
                  else ('Solo Computador' if x.get('fami_tienecomputador') == 'Si' else 'Sin Acceso TIC')),
            axis=1
        )

    # Extraer año de la columna periodo (primeros 4 dígitos)
    if 'periodo' in dff.columns:
        dff['year'] = dff['periodo'].astype(str).str[:4]
        # filtrar años válidos numéricos
        dff = dff[dff['year'].str.isnumeric()]
        dff['year'] = dff['year'].astype(int)
    else:
        dff['year'] = None

    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    # Agrupar por año y acceso TIC
    if dff['year'].notna().any():
        df_g = dff.groupby(['year', 'Acceso_TIC'], as_index=False)['punt_ingles'].mean()
        df_g = df_g.sort_values('year')
        fig = px.line(
            df_g,
            x='year',
            y='punt_ingles',
            color='Acceso_TIC',
            markers=True,
            title=f'Promedio Puntaje Inglés por Año y Acceso TIC ({municipio})',
            labels={'punt_ingles': 'Promedio Puntaje Inglés', 'year': 'Año'}
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    else:
        fig = px.line(title='No hay datos de periodo para construir la serie temporal')

    return fig