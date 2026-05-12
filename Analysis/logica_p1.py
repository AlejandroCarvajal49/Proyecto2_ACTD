import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np

def cargar_datos_p1():
    # 1. Cargar datos de Saber 11
    df = pd.read_csv('Data/saber11_Antioquia_clean.csv', dtype={'cole_cod_mcpio_ubicacion': str})
    df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].str.upper().str.strip()
    
    # Manejar el nombre de la columna de área de residencia (puede variar según el dataset de ICFES)
    col_area = 'estu_areareside' if 'estu_areareside' in df.columns else 'cole_area_ubicacion'
    
    # Limpieza: Filtrar 'Sin Información' y estandarizar a Urbano/Rural
    if col_area in df.columns:
        df = df[~df[col_area].astype(str).str.upper().isin(['SIN INFORMACIÓN', 'SIN INFORMACION', 'NAN'])]
        df['Area'] = df[col_area].apply(
            lambda x: 'Urbano' if 'CABECERA' in str(x).upper() or 'URBAN' in str(x).upper() else 'Rural'
        )
    else:
        df['Area'] = 'Desconocido'

    # Limpieza: Tratamiento de nulos en estrato socioeconómico
    col_estrato = 'fami_estratovivienda'
    if col_estrato in df.columns:
        df = df.dropna(subset=[col_estrato])
        df = df[~df[col_estrato].astype(str).str.upper().isin(['SIN INFORMACION', 'SIN INFORMACIÓN'])]

    # 2. Cargar y cruzar datos de PIB
    try:
        # 1. Usamos sep=';' o ',' (pandas puede auto-detectarlo con sep=None y engine='python')
        # 2. encoding='utf-8-sig' elimina los caracteres invisibles (BOM) de Excel
        df_pib = pd.read_csv('Data/PIB_municipios.csv', sep=None, engine='python', encoding='utf-8-sig')
        
        # 3. Limpiamos los nombres de TODAS las columnas por si tienen espacios accidentales
        df_pib.columns = df_pib.columns.str.strip()
        
        # 4. Ahora sí, estandarizamos los datos de la columna
        df_pib['Municipio'] = df_pib['Municipio'].str.upper().str.strip()
        
        # Cruzar los datos
        df = pd.merge(df, df_pib, left_on='cole_mcpio_ubicacion', right_on='Municipio', how='left')
        
    except KeyError as e:
        print(f"Error de columna en PIB_municipios.csv. Las columnas detectadas son: {df_pib.columns.tolist()}")
        df['PIB miles de millones'] = np.nan
    except FileNotFoundError:
        print("Advertencia: No se encontró 'PIB_municipios.csv'. Verifica la ruta.")
        df['PIB miles de millones'] = np.nan
        
    # ... (código anterior del try/except del PIB en cargar_datos_p1) ...
    except FileNotFoundError:
        print("Advertencia: No se encontró 'PIB_municipios.csv'. Verifica la ruta.")
        df['PIB miles de millones'] = np.nan

    # 3. Cargar y cruzar coordenadas espaciales
    try:
        df_coord = pd.read_csv('Data/municipios_unicos.csv')
        # Cruzamos usando el nombre estandarizado del municipio
        df = pd.merge(df, df_coord, on='cole_mcpio_ubicacion', how='left')
    except FileNotFoundError:
        print("Advertencia: No se encontró 'municipios_unicos.csv'. Verifica la ruta.")
        df['lat'] = np.nan
        df['lon'] = np.nan

    return df

def obtener_lista_municipios_p1(df):
    municipios = df['cole_mcpio_ubicacion'].dropna().unique().tolist()
    municipios.sort()
    return ['TODOS'] + municipios

def generar_boxplot_brecha(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    fig = px.box(
        dff,
        x='Area',
        y='punt_global',
        color='Area',
        title=f'Distribución del Puntaje Global: Urbano vs Rural ({municipio})',
        labels={'punt_global': 'Puntaje Global', 'Area': 'Zona de Residencia'},
        color_discrete_map={'Urbano': '#1f77b4', 'Rural': '#2ca02c'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def generar_dispersion_pib_brecha(df):
    # Agrupar para calcular la brecha promedio por municipio
    agrupado = df.groupby(['cole_mcpio_ubicacion', 'Area'])['punt_global'].mean().unstack()
    
    # Si falta alguna de las zonas en un municipio, no se puede calcular la brecha
    if 'Urbano' in agrupado.columns and 'Rural' in agrupado.columns:
        agrupado['Brecha_Puntos'] = agrupado['Urbano'] - agrupado['Rural']
    else:
        agrupado['Brecha_Puntos'] = np.nan
        
    agrupado = agrupado.reset_index()

    # Extraer el PIB correspondiente
    if 'PIB miles de millones' in df.columns:
        pib_df = df[['cole_mcpio_ubicacion', 'PIB miles de millones']].drop_duplicates()
        agrupado = pd.merge(agrupado, pib_df, on='cole_mcpio_ubicacion', how='left')

        fig = px.scatter(
            agrupado,
            x='PIB miles de millones',
            y='Brecha_Puntos',
            hover_name='cole_mcpio_ubicacion',
            trendline='ols',
            title='Correlación: Brecha Urbano-Rural vs PIB Municipal',
            labels={
                'Brecha_Puntos': 'Brecha (Puntos Urbano - Rural)', 
                'PIB miles de millones': 'PIB (Miles de Millones)'
            },
            opacity=0.7
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        return fig
        
    return px.scatter(title="Datos de PIB no disponibles para graficar")

def calcular_estadisticas_brecha(df, municipio):
    dff = df.copy()
    if municipio != 'TODOS':
        dff = dff[dff['cole_mcpio_ubicacion'] == municipio]

    urbano = dff[dff['Area'] == 'Urbano']['punt_global'].dropna()
    rural = dff[dff['Area'] == 'Rural']['punt_global'].dropna()

    if len(urbano) < 2 or len(rural) < 2:
        return "Insight: No hay suficientes datos en ambas zonas para calcular la brecha estadística en este municipio."

    brecha = urbano.mean() - rural.mean()
    
    # T-test asumiendo varianzas distintas (Welch's t-test)
    t_stat, p_val = stats.ttest_ind(urbano, rural, equal_var=False)

    significancia = "SIGNIFICATIVA" if p_val < 0.05 else "NO significativa"

    return f"Insight: Se evidencia una diferencia de {brecha:.1f} puntos promedio a favor de las zonas urbanas. La brecha es estadísticamente {significancia} (p-valor: {p_val:.4f})."

# ... (tu código anterior en logica_p1.py) ...

def generar_barras_brecha_error(df, municipio):
    dff = df.copy()
    
    # Si seleccionamos 'TODOS', mostramos el top 10 municipios con la brecha más grande
    if municipio == 'TODOS':
        # Agrupar por municipio y área
        agrupado = dff.groupby(['cole_mcpio_ubicacion', 'Area'])['punt_global'].agg(['mean', 'std']).reset_index()
        
        # Calcular la brecha para filtrar el Top 10
        brechas = agrupado.pivot(index='cole_mcpio_ubicacion', columns='Area', values='mean').dropna()
        brechas['Diferencia'] = brechas['Urbano'] - brechas['Rural']
        top_municipios = brechas.sort_values(by='Diferencia', ascending=False).head(10).index
        
        # Filtrar solo los datos del top 10
        dff_plot = agrupado[agrupado['cole_mcpio_ubicacion'].isin(top_municipios)]
        titulo = "Top 10 Municipios con Mayor Brecha (Promedio y Desviación)"
        x_col = 'cole_mcpio_ubicacion'
        
    # Si seleccionamos un municipio específico, comparamos con el promedio departamental
    else:
        # Calcular promedio departamental
        dpto_promedio = df.groupby('Area')['punt_global'].agg(['mean', 'std']).reset_index()
        dpto_promedio['cole_mcpio_ubicacion'] = 'PROMEDIO ANTIOQUIA'
        
        # Calcular promedio del municipio
        mpio_promedio = dff[dff['cole_mcpio_ubicacion'] == municipio].groupby('Area')['punt_global'].agg(['mean', 'std']).reset_index()
        mpio_promedio['cole_mcpio_ubicacion'] = municipio
        
        # Unir ambos
        dff_plot = pd.concat([mpio_promedio, dpto_promedio])
        titulo = f"Comparación Local vs Departamental ({municipio})"
        x_col = 'cole_mcpio_ubicacion'

    # Generar la gráfica de barras con barras de error
    fig = px.bar(
        dff_plot,
        x=x_col,
        y='mean',
        color='Area',
        barmode='group',
        error_y='std',
        title=titulo,
        labels={'mean': 'Puntaje Global Promedio', x_col: 'Municipio/Región', 'Area': 'Zona'},
        color_discrete_map={'Urbano': '#1f77b4', 'Rural': '#2ca02c'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def generar_mapa_pib_puntaje(df, municipio):
    dff = df.copy()
    
    # Agrupamos por municipio para obtener el promedio del puntaje global y mantener el PIB y coordenadas
    df_mapa = dff.groupby(['cole_mcpio_ubicacion', 'lat', 'lon']).agg(
        punt_global=('punt_global', 'mean'),
        pib=('PIB miles de millones', 'first') # El PIB es igual para todo el municipio
    ).reset_index()
    
    min_puntaje = df_mapa['punt_global'].min()
    max_puntaje = df_mapa['punt_global'].max()
    
    # Lógica para resaltar el municipio seleccionado (igual que en tu P3)
    if municipio != 'TODOS':
        df_mapa['opacidad'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 1.0 if x == municipio else 0.15)
        df_mapa['tamano'] = df_mapa['cole_mcpio_ubicacion'].apply(lambda x: 15 if x == municipio else 5)
    else:
        df_mapa['opacidad'] = 0.8
        df_mapa['tamano'] = 8

    # Crear el mapa interactivo
    fig = px.scatter_mapbox(
        df_mapa,
        lat='lat',
        lon='lon',
        color='punt_global',
        hover_name='cole_mcpio_ubicacion',
        hover_data={
            'punt_global': ':.1f', # Redondear a 1 decimal
            'pib': ':.2f',         # Mostrar PIB con 2 decimales
            'lat': False,          # Ocultar lat/lon del tooltip para mayor limpieza
            'lon': False,
            'tamano': False,
            'opacidad': False
        },
        labels={'punt_global': 'Puntaje Global Prom.', 'pib': 'PIB (Miles de Millones)'},
        color_continuous_scale='Viridis',
        range_color=[min_puntaje, max_puntaje],
        mapbox_style='carto-positron',
        zoom=6.0,
        center={"lat": 6.2518, "lon": -75.5636}, # Centro aproximado de Antioquia
        title=f'Mapa Espacial: Puntaje Global y PIB ({municipio})',
        size='tamano',
        size_max=15
    )
    
    fig.update_traces(marker=dict(opacity=df_mapa['opacidad']))
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return fig