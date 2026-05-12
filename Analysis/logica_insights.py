import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _first_present_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback by substring
    for c in df.columns:
        low = c.lower()
        for cand in candidates:
            if cand.replace('_', '').lower() in low:
                return c
    return None


def obtener_figuras_eda(path="Data/saber11_Antioquia_clean.csv"):
    """Carga datos y genera varias figuras y KPIs para el dashboard de insights.

    Retorna una tupla `(kpis, figs)` donde `kpis` es un dict con valores
    y `figs` es un dict con figuras Plotly listos para `dcc.Graph(figure=...)`.
    """
    df = pd.read_csv(path)

    # Detectar columnas candidatas (fallbacks si el nombre varía)
    col_punt_global = _first_present_column(df, ["punt_global", "puntaje_global", "global"])
    col_matematicas = _first_present_column(df, ["punt_matematicas", "matematicas", "matematicas_punt"]) 
    col_lectura = _first_present_column(df, ["punt_lectura", "lectura"])
    col_ciencias = _first_present_column(df, ["punt_ciencias", "ciencias"])
    col_sociales = _first_present_column(df, ["punt_sociales", "sociales", "punt_sociales_ciudadanas"])
    col_ingles = _first_present_column(df, ["punt_ingles", "ingles"])
    

    col_naturaleza = _first_present_column(df, ["cole_naturaleza", "naturaleza"])
    col_area = _first_present_column(df, ["cole_area_ubicacion", "cole_are_ubicacion", "area_ubicacion", "ubicacion"])
    col_caracter = _first_present_column(df, ["cole_caracter", "caracter"])
    col_genero = _first_present_column(df, ["cole_genero", "genero", "sexo"])
    col_estrato = _first_present_column(df, ["fami_estratovivienda", "estrato"])

    # KPIs básicos
    kp = {}
    if col_punt_global and col_punt_global in df.columns:
        kp["max_punt_global"] = float(df[col_punt_global].max())
        kp["min_punt_global"] = float(df[col_punt_global].min())
        kp["mean_punt_global"] = float(df[col_punt_global].mean())
        kp["pct_over_300"] = float((df[col_punt_global] > 300).mean() * 100)
    else:
        kp["max_punt_global"] = kp["min_punt_global"] = kp["mean_punt_global"] = kp["pct_over_300"] = None

    # % de exito por materia (definimos exito como >300 si existe la columna)
    success = {}
    for m in [col_matematicas, col_lectura, col_ciencias, col_punt_global, col_sociales, col_ingles]:
        if m and m in df.columns:
            success[m] = float((df[m] > 300).mean() * 100)
    kp["success_pct_by_metric"] = success

    figs = {}

    # Histograma general
    if col_punt_global:
        figs["hist_global"] = px.histogram(
            df, x=col_punt_global, nbins=40, title="Distribución Puntaje Global",
            color_discrete_sequence=["#2C3E50"]
        )
    else:
        figs["hist_global"] = go.Figure()

    # Histogramas por categorías (si existen)
    if col_area:
        figs["hist_by_area"] = px.histogram(
            df, x=col_punt_global, color=col_area, nbins=30,
            title=f"Distribución Puntaje Global por {col_area}"
        )

    if col_caracter:
        figs["hist_by_caracter"] = px.histogram(
            df, x=col_punt_global, color=col_caracter, nbins=30,
            title=f"Distribución Puntaje Global por {col_caracter}"
        )

    if col_genero:
        figs["hist_by_genero"] = px.histogram(
            df, x=col_punt_global, color=col_genero, nbins=30,
            title=f"Distribución Puntaje Global por {col_genero}"
        )

    # Boxplots: crear varias variantes y controlar visibilidad con update menus
    box_cols = [c for c in [col_naturaleza, col_area, col_caracter, col_genero] if c]
    box_fig = go.Figure()
    visible_masks = []
    for i, c in enumerate(box_cols):
        px_box = px.box(df, x=c, y=col_punt_global, points="outliers", title="Dispersión Puntaje Global")
        # convertir trazas y agregarlas
        traces = px_box.data
        for t in traces:
            t.visible = (i == 0)
            box_fig.add_trace(t)
        visible_masks.append([j >= (i * len(traces)) and j < ((i + 1) * len(traces)) for j in range(len(box_cols) * len(traces))])

    # Agregar menú para cambiar la categoría del boxplot
    if box_cols:
        buttons = []
        traces_per_group = len(px_box.data)
        for i, c in enumerate(box_cols):
            visible = [False] * (traces_per_group * len(box_cols))
            start = i * traces_per_group
            for j in range(start, start + traces_per_group):
                visible[j] = True
            buttons.append(dict(label=c, method="update", args=[{"visible": visible}, {"title": f"Boxplot puntaje global por {c}"}]))
        box_fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.15, xanchor="left")])
        figs["box_global_by_category"] = box_fig
    else:
        figs["box_global_by_category"] = go.Figure()

    # Barra por estrato con selector de métrica
    metrics = [m for m in [col_matematicas, col_lectura, col_ciencias, col_punt_global, col_sociales, col_ingles] if m]
    bar_fig = go.Figure()
    for i, m in enumerate(metrics):
        if col_estrato and m in df.columns:
            df_g = df.groupby(col_estrato)[m].mean().reset_index().sort_values(col_estrato)
            bar = px.bar(df_g, x=col_estrato, y=m)
            for t in bar.data:
                t.visible = (i == 0)
                bar_fig.add_trace(t)

    if metrics:
        buttons = []
        traces_per_group = len(bar.data)
        for i, m in enumerate(metrics):
            visible = [False] * (traces_per_group * len(metrics))
            start = i * traces_per_group
            for j in range(start, start + traces_per_group):
                visible[j] = True
            buttons.append(dict(label=m, method="update", args=[{"visible": visible}, {"title": f"Promedio {m} por {col_estrato or 'estrato'}"}]))
        bar_fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.15, xanchor="left")])
        figs["bar_by_estrato_metric_select"] = bar_fig
    else:
        figs["bar_by_estrato_metric_select"] = go.Figure()

    # Pie por genero
    if col_genero:
        figs["pie_genero"] = px.pie(df, names=col_genero, title="Distribución por Género")

    # Serie temporal: promedio de puntajes generales por año (sin separar por TIC)
    if col_punt_global and 'periodo' in df.columns:
        try:
            df['year'] = df['periodo'].astype(str).str[:4]
            df = df[df['year'].str.isnumeric()]
            df['year'] = df['year'].astype(int)
            serie_df = df.groupby('year')[col_punt_global].mean().reset_index().sort_values('year')
            figs['serie_punt_global_por_periodo'] = px.line(
                serie_df,
                x='year',
                y=col_punt_global,
                markers=True,
                title='Progresión Promedio del Puntaje Global a través de los Años',
                labels={col_punt_global: 'Promedio Puntaje Global', 'year': 'Año'}
            )
        except Exception:
            figs['serie_punt_global_por_periodo'] = go.Figure()
    else:
        figs['serie_punt_global_por_periodo'] = go.Figure()

    # Retornar KPIs y figuras
    # También construir información auxiliar para callbacks interactivos
    aux = {
        "df_columns": df.columns.tolist(),
        "detected": {
            "col_punt_global": col_punt_global,
            "col_matematicas": col_matematicas,
            "col_lectura": col_lectura,
            "col_ciencias": col_ciencias,
            "col_sociales": col_sociales,
            "col_ingles": col_ingles,
            "col_naturaleza": col_naturaleza,
            "col_area": col_area,
            "col_caracter": col_caracter,
            "col_genero": col_genero,
            "col_estrato": col_estrato,
        },
    }
    aux["df"] = df

    # Precomputar medias por estrato para cada métrica
    metrics_list = [m for m in [col_matematicas, col_lectura, col_ciencias, col_punt_global, col_sociales, col_ingles] if m]
    estrato_stats = {}
    if col_estrato:
        for m in metrics_list:
            if m in df.columns:
                df_g = df.groupby(col_estrato)[m].mean().reset_index().sort_values(col_estrato)
                estrato_stats[m] = {
                    "by_estrato": df_g,
                    "range": float(df_g[m].max() - df_g[m].min()) if not df_g[m].empty else 0.0
                }
    aux["metrics_list"] = metrics_list
    aux["estrato_stats"] = estrato_stats

    return kp, figs, aux


def build_bar_with_comparisons(df, metric, estrato_col, metrics_list):
    """Construye figura de barras por `estrato_col` para `metric` y calcula KPIs comparativos.

    Devuelve `(fig, kpis)` donde `kpis` contiene `range`, `by_estrato` (DataFrame dict),
    y `comparisons` con % diferencia vs otras métricas.
    """
    if estrato_col not in df.columns or metric not in df.columns:
        return go.Figure(), {"error": "column missing"}

    df_g = df.groupby(estrato_col)[metric].mean().reset_index().sort_values(estrato_col)
    fig = px.bar(df_g, x=estrato_col, y=metric, title=f"Promedio {metric} por {estrato_col}")

    # rango (diferencia absoluta) entre estratos
    rango = float(df_g[metric].max() - df_g[metric].min()) if not df_g[metric].empty else 0.0

    # calcular rangos para todas las metrics para comparar
    ranges = {}
    for m in metrics_list:
        if m in df.columns:
            dg = df.groupby(estrato_col)[m].mean().reset_index()
            ranges[m] = float(dg[m].max() - dg[m].min()) if not dg[m].empty else 0.0

    comparisons = {}
    for m, r in ranges.items():
        if m == metric:
            continue
        if ranges[m] == 0:
            comparisons[m] = None
        else:
            # % difference: (rango_metric - rango_other)/rango_other *100
            try:
                comparisons[m] = ((rango - r) / r) * 100.0
            except Exception:
                comparisons[m] = None

    # KPIs adicionales para el gráfico
    kpis = {
        "metric": metric,
        "range": rango,
        "by_estrato": df_g.to_dict(orient="records"),
        "comparisons_percent_vs_other": comparisons,
        "ranges_all_metrics": ranges,
    }

    return fig, kpis


    