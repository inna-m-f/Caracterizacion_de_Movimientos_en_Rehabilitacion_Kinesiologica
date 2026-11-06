"""
M4 - Comparador visual y alertas inter-sesión (KinePose)

Funciones clave (sección 5.4 del Documento de Análisis y Diseño):
    normalizar_ciclo_movimiento(serie)          -> curva remuestreada a 0..100 % del ciclo (spline)
    superponer_esqueletos_2d(rep_ref, rep_act)  -> esqueletos normalizados (escala/posición) superpuestos
    evaluar_desviaciones_clinicas(ref, act)     -> tabla con Δ% y alertas (> 10 %)
    generar_resumen_texto(tabla)                -> reporte de evolución en lenguaje natural

Unidad de comparación: una REPETICIÓN, o la repetición promedio de una sesión
(media de todas las repeticiones completas, normalizadas a 0-100 %).

Convención clínica: MediaPipe entrega ángulos internos (180° = segmento recto). Para el
kinesiólogo se reporta FLEXIÓN = 180° - ángulo interno (0° = extendido), como con goniómetro.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.interpolate import CubicSpline

from m3_pose import LADO

ANGULOS_FLEXION = {"rodilla", "cadera"}       # se reportan como 180 - ángulo interno
NOMBRES = {"rodilla": "Flexión de rodilla", "cadera": "Flexión de cadera", "tronco": "Inclinación del tronco",
           "muslo": "Inclinación del muslo", "pierna": "Inclinación de la pierna"}


# ---------------------------------------------------------------------------
# Carga de sesiones (salida de analizar_video.py)
# ---------------------------------------------------------------------------
def cargar_sesion(prefijo: str, etiqueta: str | None = None) -> dict:
    """prefijo = ruta sin sufijo, p. ej. 'resultados/4to' -> 4to_series.npz + 4to_reporte.json"""
    d = np.load(prefijo + "_series.npz")
    rep = json.load(open(prefijo + "_reporte.json"))
    series = {k: d[k] for k in d.files if k not in ("landmarks", "fps", "crudo")}
    return {"etiqueta": etiqueta or os.path.basename(prefijo), "series": series, "landmarks": d["landmarks"],
            "fps": float(d["fps"]), "reps": rep["repeticiones"], "lado": rep["lado"],
            "principal": rep["angulo_principal"], "reporte": rep}


def extraer_repeticion(sesion: dict, k: int) -> dict:
    """k = número de repetición (1, 2, ...)."""
    r = sesion["reps"][k - 1]
    sl = slice(r["inicio"], r["fin"] + 1)
    return {"etiqueta": f"{sesion['etiqueta']} · R{k}", "series": {n: s[sl] for n, s in sesion["series"].items()},
            "landmarks": sesion["landmarks"][sl], "fps": sesion["fps"], "lado": sesion["lado"],
            "principal": sesion["principal"], "duracion_s": r["duracion_s"], "completa": r["completa"]}


# ---------------------------------------------------------------------------
# Normalización temporal
# ---------------------------------------------------------------------------
def normalizar_ciclo_movimiento(serie: np.ndarray, n: int = 101) -> np.ndarray:
    """Remuestrea la curva al 0..100 % del ciclo con interpolación spline cúbica, para poder
    comparar repeticiones de distinta duración punto a punto."""
    s = np.asarray(serie, np.float64)
    ok = ~np.isnan(s)
    if ok.sum() < 4:
        return np.full(n, np.nan)
    t = np.linspace(0, 1, len(s))
    cs = CubicSpline(t[ok], s[ok])
    return cs(np.linspace(0, 1, n))


def curva_promedio(reps: list[dict], angulo: str, n: int = 101):
    """Media y desviación estándar de varias repeticiones normalizadas (repetición 'típica' de una sesión)."""
    M = np.stack([normalizar_ciclo_movimiento(r["series"][angulo], n) for r in reps])
    return np.nanmean(M, 0), np.nanstd(M, 0)


def dtw(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Dynamic Time Warping: alinea dos curvas aunque una fase (bajada/subida) sea más lenta
    en una sesión que en la otra. Devuelve la diferencia media (°) a lo largo del camino óptimo."""
    a, b = np.asarray(a), np.asarray(b)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    cost = np.abs(a[:, None] - b[None, :])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, camino = n, m, []
    while i > 0 and j > 0:
        camino.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    camino = np.array(camino[::-1])
    return float(cost[camino[:, 0], camino[:, 1]].mean()), camino


# ---------------------------------------------------------------------------
# Métricas y alertas
# ---------------------------------------------------------------------------
def a_clinico(angulo: str, valores: np.ndarray) -> np.ndarray:
    return 180.0 - valores if angulo in ANGULOS_FLEXION else valores


def metricas_repeticion(rep: dict, angulos: list[str]) -> dict:
    m = {"duracion_s": rep["duracion_s"]}
    for a in angulos:
        v = a_clinico(a, rep["series"][a])
        m[f"{a}_max"] = float(np.nanmax(v))
        m[f"{a}_rom"] = float(np.nanmax(v) - np.nanmin(v))
    return m


def metricas_promedio(reps: list[dict], angulos: list[str]) -> dict:
    ms = [metricas_repeticion(r, angulos) for r in reps]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def evaluar_desviaciones_clinicas(ref: dict, act: dict, umbral_pct: float = 10.0,
                                  dif_min_grados: float = 5.0, dif_min_s: float = 0.5) -> list[dict]:
    """Δ% = (act - ref) / |ref| · 100. Alerta si |Δ%| > umbral (regla del informe) Y además la
    diferencia absoluta supera el error de medición (MediaPipe 2D: RMSE ~2-4° en rodilla según la
    literatura citada en el informe). Sin este segundo criterio, ángulos pequeños (p. ej. 25°)
    disparan alertas por ruido. El signo se conserva: + aumentó, - disminuyó."""
    filas = []
    for k in ref:
        if k not in act:
            continue
        r, a = ref[k], act[k]
        delta = (a - r) / abs(r) * 100 if abs(r) > 1e-6 else np.nan
        if k == "duracion_s":
            nombre, unidad = "Duración de la repetición", "s"
        else:
            ang, tipo = k.rsplit("_", 1)
            nombre = f"{NOMBRES.get(ang, ang)} {'máx.' if tipo == 'max' else '(ROM)'}"
            unidad = "°"
        dif_min = dif_min_s if unidad == "s" else dif_min_grados
        filas.append({"metrica": k, "nombre": nombre, "unidad": unidad, "ref": r, "act": a,
                      "dif": a - r, "delta_pct": delta,
                      "alerta": bool(abs(delta) > umbral_pct and abs(a - r) >= dif_min)})
    return filas


def generar_resumen_texto(filas: list[dict], etiqueta_ref: str, etiqueta_act: str, dtw_grados: dict | None = None) -> str:
    alertas = [f for f in filas if f["alerta"]]
    lineas = [f"Comparación {etiqueta_act} respecto de {etiqueta_ref}."]
    if not alertas:
        lineas.append("Sin cambios mayores al 10 % en las métricas evaluadas: ejecución estable.")
    for f in alertas:
        verbo = "aumentó" if f["dif"] > 0 else "disminuyó"
        lineas.append(f"• {f['nombre']} {verbo} {abs(f['dif']):.1f}{f['unidad']} "
                      f"({f['delta_pct']:+.0f} %): {f['ref']:.1f} → {f['act']:.1f}{f['unidad']}.")
    if dtw_grados:
        partes = ", ".join(f"{NOMBRES.get(k, k).lower()} {v:.1f}°" for k, v in dtw_grados.items())
        lineas.append(f"Diferencia media de la forma del movimiento (DTW): {partes}.")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Esqueletos superpuestos
# ---------------------------------------------------------------------------
CADENA = ("hombro", "cadera", "rodilla", "tobillo", "pie")


def esqueleto_normalizado(lm_frame: np.ndarray, lado: str, escala: float, origen: str = "tobillo") -> np.ndarray:
    """Puntos de la cadena hombro-cadera-rodilla-tobillo-pie trasladados al origen (tobillo,
    punto apoyado) y divididos por el largo de la pierna: permite superponer sesiones
    grabadas a distintas distancias de la cámara."""
    idx = [int(LADO[lado][k]) for k in CADENA]
    p = lm_frame[idx, :2].astype(np.float64)
    o = lm_frame[int(LADO[lado][origen]), :2]
    q = (p - o) / escala
    q[:, 1] *= -1                           # eje y hacia arriba para graficar
    return q


def largo_pierna(lm: np.ndarray, lado: str) -> float:
    c, r, t = (int(LADO[lado][k]) for k in ("cadera", "rodilla", "tobillo"))
    return float(np.nanmedian(np.linalg.norm(lm[:, c, :2] - lm[:, r, :2], axis=1) +
                              np.linalg.norm(lm[:, r, :2] - lm[:, t, :2], axis=1)))


def superponer_esqueletos_2d(ax, rep_ref: dict, rep_act: dict, fases=(0, 25, 50, 75, 100),
                             colores=("#2563eb", "#dc2626")):
    """Dibuja en `ax` los dos esqueletos en las fases indicadas del ciclo, uno al lado del otro."""
    esc = [largo_pierna(r["landmarks"], r["lado"]) for r in (rep_ref, rep_act)]
    sep = 1.1
    for j, f in enumerate(fases):
        for rep, e, c in zip((rep_ref, rep_act), esc, colores):
            n = len(rep["landmarks"])
            i = int(round(f / 100 * (n - 1)))
            q = esqueleto_normalizado(rep["landmarks"][i], rep["lado"], e)
            if rep["lado"] == "izq":
                q[:, 0] *= -1                # misma orientación visual para ambos lados
            q[:, 0] += j * sep
            ax.plot(q[:, 0], q[:, 1], "-o", color=c, lw=2.5, ms=4, alpha=.85)
        ax.text(j * sep, -0.12, f"{f} %", ha="center", fontsize=9, color="#555")
    ax.set_aspect("equal")
    ax.axis("off")
