import os
import numpy as np
import pandas as pd

RAW_PATH   = os.path.join("Data", "saber11_Antioquia_raw.csv")
CLEAN_PATH = os.path.join("Data", "saber11_Antioquia_clean.csv")

# Separar por rangos lógicos correctos
SCORE_COLS_100 = [
    "punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas",
    "punt_c_naturales", "punt_lectura_critica"
]
SCORE_COL_500 = ["punt_global"]

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    # Seleccionamos las columnas de tipo objeto/texto
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    
    for col in obj_cols:
        # 1. Convertir a string
        # 2. Normalizar Unicode (separa letras de tildes)
        # 3. Codificar a ASCII ignorando errores (elimina las tildes)
        # 4. Decodificar de vuelta a texto normal
        # 5. Quitar espacios a los lados y comillas accidentales
        df[col] = (
            df[col].astype("string")
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.strip()
            .str.replace('^"(.*)"$', r"\1", regex=True)
        )
        
        # Reemplazar valores nulos o vacíos en texto por NA real de Pandas
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "SIN INFORMACION": pd.NA})
        
        # Optimización de memoria: Convertir a categoría si hay pocos valores únicos
        if df[col].nunique() < 50:
            df[col] = df[col].astype("category")
            
    return df

def run(in_path: str = RAW_PATH, out_path: str = CLEAN_PATH) -> pd.DataFrame:
    print("Iniciando proceso de limpieza...")
    df = pd.read_csv(in_path)
    print(f"[load] {df.shape[0]:,} filas × {df.shape[1]} columnas cargadas.")

    # 1) Limpieza general de texto (incluye quitar tildes)
    df = clean_text(df)

    # 2) Regla negocio: cole_bilingue vacío -> 'N'
    if "cole_bilingue" in df.columns:
        # Asegurar compatibilidad si la columna se volvió tipo 'category'
        if pd.api.types.is_categorical_dtype(df["cole_bilingue"]):
            if "N" not in df["cole_bilingue"].cat.categories:
                df["cole_bilingue"] = df["cole_bilingue"].cat.add_categories("N")
        df["cole_bilingue"] = df["cole_bilingue"].fillna("N")

    # 3) Convertir puntajes a numérico
    all_scores = SCORE_COLS_100 + SCORE_COL_500
    for c in all_scores:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) Puntajes fuera de rango -> NaN
    for c in SCORE_COLS_100:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan

    for c in SCORE_COL_500:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > 500), c] = np.nan

    # 5) Eliminar filas si falta algún puntaje
    before = len(df)
    df = df.dropna(subset=[c for c in all_scores if c in df.columns])
    print(f"[drop_scores] Eliminadas {before - len(df):,} filas por puntajes faltantes o anómalos.")

    # 6) Duplicados (Mejor Práctica: usar un ID único si existe)
    before = len(df)
    if "estu_consecutivo" in df.columns:
        df = df.drop_duplicates(subset=["estu_consecutivo"], keep="first")
    else:
        # Si no tienes ID único, revisa la fila completa
        df = df.drop_duplicates()
    print(f"[dedup] Eliminadas {before - len(df):,} filas duplicadas.")

    # 7) Guardar
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[save] Guardado exitosamente en: {out_path}")
    print(f"[done] Base final: {df.shape[0]:,} filas × {df.shape[1]} columnas.")

    return df

if __name__ == "__main__":
    run()