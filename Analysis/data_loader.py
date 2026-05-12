import pandas as pd
import os

# Ruta relativa dinámica para evitar errores en AWS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'datos_limpios.csv')

def cargar_datos():
    # Aquí puedes poner lógica de caché si el archivo es muy grande
    df = pd.read_csv(DATA_PATH)
    return df
