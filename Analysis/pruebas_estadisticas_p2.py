"""
Pruebas estadísticas para P2: brecha educativa público vs privado en Antioquia.

Ejecutar desde la raíz del proyecto:
    python Analysis/pruebas_estadisticas_p2.py

Salidas en Analysis/resultados_estadisticos_p2/:
  - resumen_pruebas.txt        : texto completo con todos los resultados
  - brecha_por_materia.csv     : tabla t-test por materia ordenada por brecha
  - brecha_por_estrato.csv     : tabla t-test estratificada por nivel socioeco
  - anova_estratos.csv         : resultado ANOVA entre estratos
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "saber11_Antioquia_clean.csv"
OUT_DIR = BASE_DIR / "Analysis" / "resultados_estadisticos_p2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MATERIAS = {
    "Puntaje Global":          "punt_global",
    "Matemáticas":             "punt_matematicas",
    "Lectura Crítica":         "punt_lectura_critica",
    "Ciencias Naturales":      "punt_c_naturales",
    "Sociales y Ciudadanas":   "punt_sociales_ciudadanas",
    "Inglés":                  "punt_ingles",
}

ORDEN_ESTRATOS = [
    "Estrato 1", "Estrato 2", "Estrato 3",
    "Estrato 4", "Estrato 5", "Estrato 6",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cohen_d(grupo_a, grupo_b):
    """Cohen's d con varianza pooled (fórmula estándar)."""
    n_a, n_b = len(grupo_a), len(grupo_b)
    var_a, var_b = np.var(grupo_a, ddof=1), np.var(grupo_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(grupo_a) - np.mean(grupo_b)) / pooled_std


def interpretar_d(d):
    """Clasificación estándar de Cohen (1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def interpretar_p(p):
    if p < 0.001:
        return "p<0.001 *** (altamente significativo)"
    if p < 0.01:
        return "p<0.01 ** (muy significativo)"
    if p < 0.05:
        return "p<0.05 * (significativo)"
    return f"p={p:.4f} (no significativo al 5%)"


def welch_ttest(grupo_a, grupo_b, etiqueta_a="Público", etiqueta_b="Privado"):
    """Prueba t de Welch (no asume igualdad de varianzas) + métricas clave."""
    t, p = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
    d = cohen_d(grupo_a, grupo_b)
    return {
        "media_" + etiqueta_a.lower(): round(np.mean(grupo_a), 2),
        "media_" + etiqueta_b.lower(): round(np.mean(grupo_b), 2),
        "diferencia": round(np.mean(grupo_b) - np.mean(grupo_a), 2),  # privado - público
        "t": round(t, 4),
        "p_valor": p,
        "cohen_d": round(d, 4),
        "efecto": interpretar_d(d),
        "interpretacion_p": interpretar_p(p),
        "n_" + etiqueta_a.lower(): len(grupo_a),
        "n_" + etiqueta_b.lower(): len(grupo_b),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────

print("Cargando datos...")
df = pd.read_csv(str(DATA_PATH))

publico  = df[df["cole_naturaleza"] == "OFICIAL"]
privado  = df[df["cole_naturaleza"] == "NO OFICIAL"]
print(f"  Público (OFICIAL):    {len(publico):,} filas")
print(f"  Privado (NO OFICIAL): {len(privado):,} filas")

lineas = []  # acumula el reporte de texto


def titulo(t):
    sep = "=" * 70
    lineas.append(f"\n{sep}\n{t}\n{sep}")
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def log(s=""):
    lineas.append(s)
    print(s)


# ─────────────────────────────────────────────────────────────────────────────
# PRUEBA 1 — t de Welch: punt_global público vs privado
# ─────────────────────────────────────────────────────────────────────────────

titulo("PRUEBA 1 — t de Welch: punt_global público vs privado")
log("Hipótesis nula: las medias de punt_global son iguales entre colegios públicos y privados.")
log("Se usa la variante de Welch porque los tamaños de grupo son muy distintos (3:1).")

g_pub  = publico["punt_global"].dropna()
g_priv = privado["punt_global"].dropna()
r1 = welch_ttest(g_pub, g_priv)

log(f"  n_publico  = {r1['n_público']:,}")
log(f"  n_privado  = {r1['n_privado']:,}")
log(f"  Media público  = {r1['media_público']}")
log(f"  Media privado  = {r1['media_privado']}")
log(f"  Diferencia (priv - pub) = {r1['diferencia']:.2f} pts")
log(f"  t = {r1['t']:.4f}")
log(f"  {r1['interpretacion_p']}")
log(f"  Cohen's d = {r1['cohen_d']:.4f} ({r1['efecto']})")
log()
# Cohen's d: <0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, ≥0.8 large
log("  Interpretación: la diferencia de medias es estadísticamente significativa")
log("  si p<0.05. El tamaño del efecto (Cohen's d) indica su relevancia práctica.")


# ─────────────────────────────────────────────────────────────────────────────
# PRUEBA 2 — t de Welch por materia
# ─────────────────────────────────────────────────────────────────────────────

titulo("PRUEBA 2 — t de Welch por materia (¿en qué área hay mayor rezago?)")
log("Para cada área se calcula la brecha de medias (privado - público),")
log("el p-valor y Cohen's d. Ordenado de mayor a menor brecha.")

filas_mat = []
for nombre, col in MATERIAS.items():
    gp = publico[col].dropna()
    gv = privado[col].dropna()
    r = welch_ttest(gp, gv)
    filas_mat.append({
        "Materia":    nombre,
        "Público":    r["media_público"],
        "Privado":    r["media_privado"],
        "Brecha":     r["diferencia"],
        "t":          r["t"],
        "p_valor":    r["p_valor"],
        "Cohen_d":    r["cohen_d"],
        "Efecto":     r["efecto"],
    })

df_mat = pd.DataFrame(filas_mat).sort_values("Brecha", ascending=False)
log(df_mat.to_string(index=False))
log()
log("  Interpretación: cuanto mayor la brecha, mayor el rezago del colegio público")
log("  en esa área. Un Cohen's d 'medium' o 'large' indica relevancia práctica.")

df_mat.to_csv(OUT_DIR / "brecha_por_materia.csv", index=False)
log(f"\n  Guardado: {OUT_DIR / 'brecha_por_materia.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# PRUEBA 3 — t de Welch estratificada (público vs privado dentro de cada estrato)
# ─────────────────────────────────────────────────────────────────────────────

titulo("PRUEBA 3 — t de Welch estratificada (¿la brecha persiste al controlar por estrato?)")
log("Si la brecha público-privado se mantiene dentro de cada estrato,")
log("significa que el tipo de colegio tiene un efecto NETO independiente del nivel socioeconómico.")

filas_est = []
for estrato in ORDEN_ESTRATOS:
    sub = df[df["fami_estratovivienda"] == estrato]
    gp = sub[sub["cole_naturaleza"] == "OFICIAL"]["punt_global"].dropna()
    gv = sub[sub["cole_naturaleza"] == "NO OFICIAL"]["punt_global"].dropna()
    if len(gp) < 30 or len(gv) < 30:
        log(f"  {estrato}: muestra insuficiente (pub={len(gp)}, priv={len(gv)})")
        continue
    r = welch_ttest(gp, gv)
    filas_est.append({
        "Estrato":    estrato,
        "Público":    r["media_público"],
        "Privado":    r["media_privado"],
        "Brecha":     r["diferencia"],
        "p_valor":    r["p_valor"],
        "Cohen_d":    r["cohen_d"],
        "Efecto":     r["efecto"],
        "n_pub":      r["n_público"],
        "n_priv":     r["n_privado"],
    })

df_est = pd.DataFrame(filas_est)
log(df_est.to_string(index=False))
log()
log("  Interpretación: si p<0.05 en todos los estratos, la brecha existe")
log("  independientemente del nivel socioeconómico del estudiante.")

df_est.to_csv(OUT_DIR / "brecha_por_estrato.csv", index=False)
log(f"\n  Guardado: {OUT_DIR / 'brecha_por_estrato.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# PRUEBA 4 — ANOVA de una vía: punt_global entre estratos
# ─────────────────────────────────────────────────────────────────────────────

titulo("PRUEBA 4 — ANOVA de una vía: punt_global entre los 6 estratos")
log("Valida el 'efecto escalera': a mayor estrato, mayor puntaje promedio.")
log("H0: las medias de punt_global son iguales en todos los estratos.")

grupos_anova = []
medias_anova = []
for estrato in ORDEN_ESTRATOS:
    g = df[df["fami_estratovivienda"] == estrato]["punt_global"].dropna()
    if len(g) > 0:
        grupos_anova.append(g)
        medias_anova.append((estrato, round(g.mean(), 2), len(g)))

F, p_anova = stats.f_oneway(*grupos_anova)

log(f"\n  Medias por estrato:")
for est, media, n in medias_anova:
    log(f"    {est}: {media:.2f} (n={n:,})")
log(f"\n  F = {F:.4f}")
log(f"  {interpretar_p(p_anova)}")
log()
log("  Interpretación: si p<0.05, existe al menos una diferencia significativa")
log("  entre estratos. Un F grande confirma el efecto escalera socioeconómico.")

df_anova = pd.DataFrame([{
    "prueba": "ANOVA punt_global ~ estrato",
    "F": round(F, 4),
    "p_valor": p_anova,
    "interpretacion": interpretar_p(p_anova),
}])
df_anova.to_csv(OUT_DIR / "anova_estratos.csv", index=False)
log(f"\n  Guardado: {OUT_DIR / 'anova_estratos.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR RESUMEN COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

resumen_path = OUT_DIR / "resumen_pruebas.txt"
with open(resumen_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lineas))

print(f"\n{'='*70}")
print(f"Resumen completo guardado en: {resumen_path}")
print(f"{'='*70}")
