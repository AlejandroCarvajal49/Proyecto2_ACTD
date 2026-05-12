import os
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _first_present_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback por substring
    for c in df.columns:
        low = c.lower()
        for cand in candidates:
            if cand.replace("_", "").lower() in low:
                return c
    return None


def _normalize_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("utf-8")
    return text


def _standardize_zone(series):
    s = series.astype("string").str.upper().str.strip()

    def _map_zone(val):
        if pd.isna(val):
            return pd.NA
        if "RURAL" in val:
            return "RURAL"
        if "URB" in val:
            return "URBANO"
        return val

    return s.map(_map_zone)


def _standardize_school_type(series):
    s = series.astype("string").str.upper().str.strip()

    def _map_type(val):
        if pd.isna(val):
            return pd.NA
        if "NO OFICIAL" in val or "PRIVAD" in val:
            return "PRIVADO"
        if "OFICIAL" in val or "PUBLIC" in val:
            return "PUBLICO"
        return val

    return s.map(_map_type)


def _binary_flag(series):
    yes_vals = {"SI", "S", "1", "TRUE", "T", "Y", "YES"}
    no_vals = {"NO", "N", "0", "FALSE", "F"}
    s = series.astype("string").str.upper().str.strip()

    def _map_flag(val):
        if pd.isna(val):
            return pd.NA
        if val in yes_vals:
            return 1
        if val in no_vals:
            return 0
        return pd.NA

    return s.map(_map_flag)


def _build_access_indicator(df, col_internet, col_computador):
    if col_internet is None and col_computador is None:
        return pd.Series([pd.NA] * len(df), index=df.index)

    if col_internet in df.columns:
        internet = _binary_flag(df[col_internet])
    else:
        internet = pd.Series([pd.NA] * len(df), index=df.index)

    if col_computador in df.columns:
        computador = _binary_flag(df[col_computador])
    else:
        computador = pd.Series([pd.NA] * len(df), index=df.index)

    access = ((internet == 1) | (computador == 1)).astype("Int64")
    return access.map({1: "CON_ACCESO", 0: "SIN_ACCESO"})


def _safe_quantile(series, q, fallback):
    if series is None:
        return fallback
    try:
        val = float(series.quantile(q))
        if np.isnan(val):
            return fallback
        return val
    except Exception:
        return fallback


def _adjusted_means_by_group(df, metric, group_col, strat_col):
    base = df[[metric, group_col, strat_col]].dropna()
    if base.empty:
        return {}

    weights = base[strat_col].value_counts(normalize=True)
    means = base.groupby([group_col, strat_col])[metric].mean().unstack()

    adjusted = {}
    for group in means.index:
        group_means = means.loc[group]
        aligned = weights.reindex(group_means.index)
        valid = group_means.notna() & aligned.notna()
        if valid.any():
            w = aligned[valid]
            adjusted[group] = float((group_means[valid] * w).sum() / w.sum())

    return adjusted


def _load_pib_data(base_dir):
    path = os.path.join(base_dir, "Data", "PIB_municipios.csv")
    if not os.path.exists(path):
        return None

    pib = pd.read_csv(path)
    col_mun = _first_present_column(pib, ["Municipio", "municipio", "MUNICIPIO"])
    col_pib = _first_present_column(pib, ["PIB miles de millones", "PIB", "pib"])
    if col_mun is None or col_pib is None:
        return None

    pib = pib.rename(columns={col_mun: "municipio_raw", col_pib: "pib_miles_millones"})
    pib["MUNICIPIO_NORM"] = pib["municipio_raw"].apply(_normalize_text)
    pib["pib_miles_millones"] = pd.to_numeric(pib["pib_miles_millones"], errors="coerce")
    return pib[["MUNICIPIO_NORM", "pib_miles_millones"]].dropna(subset=["MUNICIPIO_NORM"])


def obtener_figuras_eda(path="Data/saber11_Antioquia_clean.csv"):
    """Genera insights accionables para focalizacion de recursos y cierre de brechas."""
    df = pd.read_csv(path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Detectar columnas candidatas (fallbacks si el nombre varia)
    col_punt_global = _first_present_column(df, ["punt_global", "puntaje_global", "global"])
    col_matematicas = _first_present_column(df, ["punt_matematicas", "matematicas", "matematicas_punt"])
    col_lectura = _first_present_column(df, ["punt_lectura_critica", "punt_lectura", "lectura"])
    col_ciencias = _first_present_column(df, ["punt_c_naturales", "punt_ciencias", "c_naturales", "ciencias"])
    col_sociales = _first_present_column(df, ["punt_sociales_ciudadanas", "punt_sociales", "sociales"])
    col_ingles = _first_present_column(df, ["punt_ingles", "ingles"])

    col_area = _first_present_column(df, ["cole_area_ubicacion", "area_ubicacion", "ubicacion"])
    col_naturaleza = _first_present_column(df, ["cole_naturaleza", "naturaleza"])
    col_estrato = _first_present_column(df, ["fami_estratovivienda", "estrato"])
    col_internet = _first_present_column(df, ["fami_tieneinternet", "tiene_internet", "internet"])
    col_computador = _first_present_column(df, ["fami_tienecomputador", "tiene_computador", "computador"])
    col_municipio = _first_present_column(df, ["cole_mcpio_ubicacion", "mcpio_ubicacion", "municipio"])
    col_genero = _first_present_column(df, ["cole_genero", "genero", "sexo"])

    # Variables de segmentacion para decisiones publicas
    df["ZONA"] = _standardize_zone(df[col_area]) if col_area else pd.NA
    df["TIPO_COLEGIO"] = _standardize_school_type(df[col_naturaleza]) if col_naturaleza else pd.NA
    df["ACCESO_INTERNET"] = _build_access_indicator(df, col_internet, col_computador)

    if col_punt_global:
        df["Puntaje_Global"] = pd.to_numeric(df[col_punt_global], errors="coerce")
    else:
        df["Puntaje_Global"] = pd.NA

    # Targets para modelos (regresion y clasificacion)
    umbral_riesgo = _safe_quantile(df["Puntaje_Global"], 0.25, 250.0)
    df["Prioridad_Intervencion_Binaria"] = (df["Puntaje_Global"] < umbral_riesgo).astype("Int64")

    if col_punt_global:
        q25 = _safe_quantile(df["Puntaje_Global"], 0.25, umbral_riesgo)
        q75 = _safe_quantile(df["Puntaje_Global"], 0.75, umbral_riesgo)
        df["Prioridad_Intervencion"] = pd.cut(
            df["Puntaje_Global"],
            bins=[-np.inf, q25, q75, np.inf],
            labels=["ALTA", "MEDIA", "BAJA"],
            include_lowest=True,
        )
    else:
        df["Prioridad_Intervencion"] = pd.NA

    percentiles_global = {}
    if col_punt_global:
        percentiles_global = df["Puntaje_Global"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

    figs = {}
    kp = {
        "tasa_riesgo_departamental": float(df["Prioridad_Intervencion_Binaria"].mean() * 100) if col_punt_global else None,
        "umbral_riesgo": float(umbral_riesgo) if col_punt_global else None,
        "brecha_rural_urbana_p50": None,
        "impacto_estimado_publico_privado_max": None,
        "impacto_estimado_publico_privado_metric": None,
        "impacto_digital_ingles_gap": None,
        "municipios_prioridad_alta": None,
    }

    # Diagnostico base: mapa de brechas del puntaje global
    if col_punt_global:
        figs["mapa_brechas_puntaje_global"] = px.histogram(
            df,
            x="Puntaje_Global",
            nbins=40,
            title="Mapa de brechas: puntaje global",
            color_discrete_sequence=["#2C3E50"],
        )
    else:
        figs["mapa_brechas_puntaje_global"] = go.Figure()

    # Diagnostico base: composicion por genero
    if col_genero:
        figs["mapa_brechas_genero"] = px.pie(
            df,
            names=col_genero,
            title="Mapa de brechas: composicion por genero",
        )
    else:
        figs["mapa_brechas_genero"] = go.Figure()

    # Diagnostico base: tendencia temporal
    if col_punt_global and "periodo" in df.columns:
        try:
            serie_df = df[["periodo", "Puntaje_Global"]].copy()
            serie_df["anio"] = serie_df["periodo"].astype(str).str[:4]
            serie_df = serie_df[serie_df["anio"].str.isnumeric()]
            serie_df["anio"] = serie_df["anio"].astype(int)
            serie_df = serie_df.groupby("anio")["Puntaje_Global"].mean().reset_index().sort_values("anio")
            figs["serie_punt_global_por_periodo"] = px.line(
                serie_df,
                x="anio",
                y="Puntaje_Global",
                markers=True,
                title="Mapa de brechas: puntaje global a traves del tiempo",
                labels={"Puntaje_Global": "Puntaje Global", "anio": "Anio"},
            )
        except Exception:
            figs["serie_punt_global_por_periodo"] = go.Figure()
    else:
        figs["serie_punt_global_por_periodo"] = go.Figure()

    # Q1: Equidad regional - brecha rural/urbana
    if col_punt_global and df["ZONA"].notna().any():
        zona_order = ["RURAL", "URBANO"]
        figs["mapa_brechas_rural_urbana"] = px.box(
            df,
            x="ZONA",
            y="Puntaje_Global",
            color="ZONA",
            category_orders={"ZONA": zona_order},
            title="Mapa de brechas: Puntaje Global por zona (Rural/Urbana)",
            labels={"ZONA": "Zona", "Puntaje_Global": "Puntaje Global"},
        )

        medians = df.groupby("ZONA")["Puntaje_Global"].median()
        if "RURAL" in medians.index and "URBANO" in medians.index:
            kp["brecha_rural_urbana_p50"] = float(medians["URBANO"] - medians["RURAL"])
    else:
        figs["mapa_brechas_rural_urbana"] = go.Figure()

    # Q1: Municipios de bajo PIB con mayor disparidad
    municipio_gap_df = pd.DataFrame()
    if col_municipio and col_punt_global and df["ZONA"].notna().any():
        base = df[[col_municipio, "ZONA", "Puntaje_Global"]].dropna()
        base["MUNICIPIO_NORM"] = base[col_municipio].apply(_normalize_text)
        mun_display = base.groupby("MUNICIPIO_NORM")[col_municipio].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        stats = base.groupby(["MUNICIPIO_NORM", "ZONA"])["Puntaje_Global"].mean().unstack()
        urbano = stats.get("URBANO")
        rural = stats.get("RURAL")
        if urbano is not None and rural is not None:
            stats["brecha_rural_urbana"] = (urbano - rural).abs()
        else:
            stats["brecha_rural_urbana"] = np.nan
        stats = stats.reset_index().merge(mun_display.rename("municipio_display"), on="MUNICIPIO_NORM", how="left")

        pib = _load_pib_data(base_dir)
        if pib is not None:
            stats = stats.merge(pib, on="MUNICIPIO_NORM", how="left")

        municipio_gap_df = stats.dropna(subset=["brecha_rural_urbana"]).copy()
        if not municipio_gap_df.empty:
            gap_cut = municipio_gap_df["brecha_rural_urbana"].quantile(0.75)
            pib_cut = None
            if "pib_miles_millones" in municipio_gap_df.columns:
                pib_cut = municipio_gap_df["pib_miles_millones"].quantile(0.25)
                if pd.isna(pib_cut):
                    pib_cut = None

            municipio_gap_df["prioridad_alta"] = municipio_gap_df["brecha_rural_urbana"] >= gap_cut
            if pib_cut is not None:
                municipio_gap_df["prioridad_alta"] &= municipio_gap_df["pib_miles_millones"] <= pib_cut

            kp["municipios_prioridad_alta"] = int(municipio_gap_df["prioridad_alta"].sum())

            plot_df = municipio_gap_df.copy()
            if pib_cut is not None:
                plot_df = plot_df[plot_df["pib_miles_millones"] <= pib_cut]
            plot_df = plot_df.sort_values("brecha_rural_urbana", ascending=False).head(15)

            if not plot_df.empty:
                figs["mapa_brechas_municipios_pib"] = px.bar(
                    plot_df,
                    x="municipio_display",
                    y="brecha_rural_urbana",
                    color="prioridad_alta",
                    title="Mapa de brechas: municipios de bajo PIB con mayor disparidad rural-urbana",
                    labels={"municipio_display": "Municipio", "brecha_rural_urbana": "Brecha Rural-Urbana"},
                )
            else:
                figs["mapa_brechas_municipios_pib"] = go.Figure()
        else:
            figs["mapa_brechas_municipios_pib"] = go.Figure()
    else:
        figs["mapa_brechas_municipios_pib"] = go.Figure()

    # Q2: Calidad publico vs privado ajustado por estrato
    metrics = [m for m in [col_matematicas, col_lectura, col_ciencias, col_sociales, col_ingles, col_punt_global] if m]
    adjusted_rows = []
    if col_estrato and metrics and df["TIPO_COLEGIO"].notna().any():
        for metric in metrics:
            adjusted = _adjusted_means_by_group(df, metric, "TIPO_COLEGIO", col_estrato)
            if "PUBLICO" in adjusted and "PRIVADO" in adjusted:
                gap = adjusted["PRIVADO"] - adjusted["PUBLICO"]
                adjusted_rows.append(
                    {
                        "metric": metric,
                        "gap_privado_menos_publico": gap,
                        "publico_ajustado": adjusted["PUBLICO"],
                        "privado_ajustado": adjusted["PRIVADO"],
                    }
                )

    adjusted_df = pd.DataFrame(adjusted_rows)
    if not adjusted_df.empty:
        idx = adjusted_df["gap_privado_menos_publico"].abs().idxmax()
        kp["impacto_estimado_publico_privado_max"] = float(adjusted_df.loc[idx, "gap_privado_menos_publico"])
        kp["impacto_estimado_publico_privado_metric"] = adjusted_df.loc[idx, "metric"]

        figs["impacto_estimado_publico_privado"] = px.bar(
            adjusted_df,
            x="metric",
            y="gap_privado_menos_publico",
            title="Impacto estimado publico vs privado (ajustado por estrato)",
            labels={"metric": "Area", "gap_privado_menos_publico": "Brecha Ajustada (Privado - Publico)"},
        )
    else:
        figs["impacto_estimado_publico_privado"] = go.Figure()

    if col_punt_global and df["TIPO_COLEGIO"].notna().any():
        figs["mapa_brechas_tipo_colegio"] = px.box(
            df,
            x="TIPO_COLEGIO",
            y="Puntaje_Global",
            color="TIPO_COLEGIO",
            title="Mapa de brechas: Puntaje Global por tipo de colegio",
            labels={"TIPO_COLEGIO": "Tipo de Colegio", "Puntaje_Global": "Puntaje Global"},
        )
    else:
        figs["mapa_brechas_tipo_colegio"] = go.Figure()

    # Q3: Competitividad y bilinguismo
    if col_ingles and df["ACCESO_INTERNET"].notna().any():
        figs["mapa_brechas_digital_ingles"] = px.box(
            df,
            x="ACCESO_INTERNET",
            y=col_ingles,
            color="ACCESO_INTERNET",
            title="Mapa de brechas: Ingles vs acceso digital",
            labels={"ACCESO_INTERNET": "Acceso Digital", col_ingles: "Puntaje Ingles"},
        )

        med_ingles = df.groupby("ACCESO_INTERNET")[col_ingles].median()
        if "CON_ACCESO" in med_ingles.index and "SIN_ACCESO" in med_ingles.index:
            kp["impacto_digital_ingles_gap"] = float(med_ingles["CON_ACCESO"] - med_ingles["SIN_ACCESO"])
    else:
        figs["mapa_brechas_digital_ingles"] = go.Figure()

    # Potencial de mejora municipal si se cierra la brecha digital
    potencial_df = pd.DataFrame()
    if col_municipio and col_ingles and df["ACCESO_INTERNET"].notna().any():
        base = df[[col_municipio, "ACCESO_INTERNET", col_ingles]].dropna()
        base["MUNICIPIO_NORM"] = base[col_municipio].apply(_normalize_text)
        mun_display = base.groupby("MUNICIPIO_NORM")[col_municipio].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        means = base.groupby(["MUNICIPIO_NORM", "ACCESO_INTERNET"])[col_ingles].mean().unstack()
        totals = base.groupby("MUNICIPIO_NORM").size()
        share_sin = base.groupby("MUNICIPIO_NORM")["ACCESO_INTERNET"].apply(lambda s: (s == "SIN_ACCESO").mean())

        potencial_df = means.copy()
        con_acceso = means.get("CON_ACCESO")
        sin_acceso = means.get("SIN_ACCESO")
        if con_acceso is not None and sin_acceso is not None:
            potencial_df["gap_ingles_digital"] = (con_acceso - sin_acceso).fillna(0)
        else:
            potencial_df["gap_ingles_digital"] = 0
        potencial_df["share_sin_acceso"] = share_sin
        potencial_df["estudiantes"] = totals
        potencial_df = potencial_df.reset_index().merge(mun_display.rename("municipio_display"), on="MUNICIPIO_NORM", how="left")

        if not potencial_df.empty:
            gap_cut = potencial_df["gap_ingles_digital"].quantile(0.75)
            share_cut = potencial_df["share_sin_acceso"].quantile(0.75)

            def _classify(row):
                if row["gap_ingles_digital"] >= gap_cut and row["share_sin_acceso"] >= share_cut:
                    return "ALTO"
                if row["gap_ingles_digital"] >= gap_cut * 0.5 and row["share_sin_acceso"] >= share_cut * 0.5:
                    return "MEDIO"
                return "BAJO"

            potencial_df["potencial_mejora"] = potencial_df.apply(_classify, axis=1)

            figs["potencial_mejora_digital_municipio"] = px.scatter(
                potencial_df,
                x="share_sin_acceso",
                y="gap_ingles_digital",
                size="estudiantes",
                color="potencial_mejora",
                hover_name="municipio_display",
                title="Impacto estimado: potencial de mejora en ingles al cerrar brecha digital",
                labels={
                    "share_sin_acceso": "% sin acceso digital",
                    "gap_ingles_digital": "Brecha Ingles (Con - Sin acceso)",
                },
            )
        else:
            figs["potencial_mejora_digital_municipio"] = go.Figure()
    else:
        figs["potencial_mejora_digital_municipio"] = go.Figure()

    # Resumenes de riesgo por segmento
    segment_risk = {}
    if df["ZONA"].notna().any():
        segment_risk["ZONA"] = (df.groupby("ZONA")["Prioridad_Intervencion_Binaria"].mean() * 100).to_dict()
    if df["TIPO_COLEGIO"].notna().any():
        segment_risk["TIPO_COLEGIO"] = (df.groupby("TIPO_COLEGIO")["Prioridad_Intervencion_Binaria"].mean() * 100).to_dict()
    if df["ACCESO_INTERNET"].notna().any():
        segment_risk["ACCESO_INTERNET"] = (df.groupby("ACCESO_INTERNET")["Prioridad_Intervencion_Binaria"].mean() * 100).to_dict()

    aux = {
        "df": df,
        "df_columns": df.columns.tolist(),
        "detected": {
            "col_punt_global": col_punt_global,
            "col_matematicas": col_matematicas,
            "col_lectura": col_lectura,
            "col_ciencias": col_ciencias,
            "col_sociales": col_sociales,
            "col_ingles": col_ingles,
            "col_area": col_area,
            "col_naturaleza": col_naturaleza,
            "col_estrato": col_estrato,
            "col_internet": col_internet,
            "col_computador": col_computador,
            "col_municipio": col_municipio,
            "col_genero": col_genero,
        },
        "percentiles_global": percentiles_global,
        "segment_risk": segment_risk,
        "features_sugeridas": [c for c in ["ZONA", "TIPO_COLEGIO", "ACCESO_INTERNET", col_estrato] if c],
        "targets": {
            "regresion": "Puntaje_Global",
            "clasificacion_binaria": "Prioridad_Intervencion_Binaria",
            "clasificacion_multiclase": "Prioridad_Intervencion",
        },
        "q1_municipios_prioridad": municipio_gap_df.head(25).to_dict(orient="records") if not municipio_gap_df.empty else [],
        "q2_brecha_ajustada": adjusted_df.to_dict(orient="records") if not adjusted_df.empty else [],
        "q3_potencial_mejora": potencial_df.head(25).to_dict(orient="records") if not potencial_df.empty else [],
    }

    return kp, figs, aux