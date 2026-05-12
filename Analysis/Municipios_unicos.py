

import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import json
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import numpy as np

def cargar_datos_p3():
    df = pd.read_csv('Data/saber11_Antioquia_clean.csv', dtype={'cole_cod_mcpio_ubicacion': str})
    df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].str.upper().str.strip()
    
    # Ingeniería de características: Acceso TIC
    df['Acceso_TIC'] = df.apply(
        lambda x: 'Internet y Computador' if x['fami_tieneinternet'] == 'Si' and x['fami_tienecomputador'] == 'Si'
        else ('Solo Internet' if x['fami_tieneinternet'] == 'Si'
              else ('Solo Computador' if x['fami_tienecomputador'] == 'Si' else 'Sin Acceso TIC')),
        axis=1
    )
    return df

def exportar_municipios_csv(out_path='Data/municipios_unicos.csv'):
    df = cargar_datos_p3()
    municipios = pd.Series(df['cole_mcpio_ubicacion'].dropna().unique())
    municipios = municipios.sort_values().reset_index(drop=True)
    municipios.to_frame('cole_mcpio_ubicacion').to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Guardado {len(municipios)} municipios únicos en: {out_path}')

if __name__ == '__main__':
    exportar_municipios_csv()