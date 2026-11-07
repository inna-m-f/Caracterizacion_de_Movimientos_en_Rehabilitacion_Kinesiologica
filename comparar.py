"""
M4 - Genera el panel comparativo entre dos ejecuciones.

Ejemplos:
  # dos repeticiones del mismo video (demo H2.1)
  python comparar.py --ref resultados/4to --rep-ref 1 --act resultados/4to --rep-act 2 --salida comparacion/
  # dos sesiones (repetición promedio de cada una)
  python comparar.py --ref resultados/lunes --act resultados/viernes --salida comparacion/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m4_comparador import (NOMBRES, a_clinico, cargar_sesion, curva_promedio, dtw, evaluar_desviaciones_clinicas,
                           extraer_repeticion, generar_resumen_texto, metricas_promedio, metricas_repeticion,
                           normalizar_ciclo_movimiento, superponer_esqueletos_2d)

AZUL, ROJO = "#2563eb", "#dc2626"

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--act", required=True)
ap.add_argument("--rep-ref", type=int, default=None, help="n° de repetición; omitir = promedio de la sesión")
ap.add_argument("--rep-act", type=int, default=None)
ap.add_argument("--etq-ref", default=None); ap.add_argument("--etq-act", default=None)
ap.add_argument("--umbral", type=float, default=10.0)
ap.add_argument("--dif-min", type=float, default=5.0, help="diferencia mínima en grados para alertar")
ap.add_argument("--salida", default="comparacion"); ap.add_argument("--titulo", default=None)
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)


def preparar(prefijo, k, etq):
    ses = cargar_sesion(prefijo, etq)
    completas = [i + 1 for i, r in enumerate(ses["reps"]) if r["completa"]]
    reps = [extraer_repeticion(ses, k)] if k else [extraer_repeticion(ses, i) for i in completas]
    etiqueta = reps[0]["etiqueta"] if k else f"{ses['etiqueta']} (prom. {len(reps)} rep.)"
    return ses, reps, etiqueta


ses_r, reps_r, etq_r = preparar(a.ref, a.rep_ref, a.etq_ref)
ses_a, reps_a, etq_a = preparar(a.act, a.rep_act, a.etq_act)
angulos = list(ses_r["series"].keys())
principal = ses_r["principal"]
angulos = [principal] + [x for x in angulos if x != principal]

m_ref = metricas_promedio(reps_r, angulos)
m_act = metricas_promedio(reps_a, angulos)
filas = evaluar_desviaciones_clinicas(m_ref, m_act, a.umbral, a.dif_min)

curvas, dtw_g = {}, {}
for ang in angulos:
    cr, sr = curva_promedio(reps_r, ang)
    ca, sa = curva_promedio(reps_a, ang)
    cr, ca = a_clinico(ang, cr), a_clinico(ang, ca)
    curvas[ang] = (cr, sr, ca, sa)
    dtw_g[ang], _ = dtw(cr, ca)
resumen = generar_resumen_texto(filas, etq_r, etq_a, dtw_g)

# --------------------------- figura ---------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, len(angulos), height_ratios=[0.95, 1, 1.05], hspace=.4, wspace=.25)
fig.suptitle(a.titulo or f"Comparador KinePose: {etq_r} vs {etq_a}", fontsize=14, weight="bold")

ax0 = fig.add_subplot(gs[0, :])
superponer_esqueletos_2d(ax0, reps_r[0], reps_a[0], colores=(AZUL, ROJO))
ax0.set_title("Esqueletos superpuestos en 0-25-50-75-100 % del ciclo (normalizados por largo de pierna)", fontsize=10)
ax0.plot([], [], color=AZUL, lw=3, label=etq_r); ax0.plot([], [], color=ROJO, lw=3, label=etq_a)
ax0.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)
ax0.margins(x=0.02, y=0.02)

x = np.linspace(0, 100, 101)
for j, ang in enumerate(angulos):
    cr, sr, ca, sa = curvas[ang]
    ax = fig.add_subplot(gs[1, j])
    ax.plot(x, cr, color=AZUL, lw=2.2, label=etq_r)
    ax.plot(x, ca, color=ROJO, lw=2.2, label=etq_a)
    if len(reps_r) > 1: ax.fill_between(x, cr - sr, cr + sr, color=AZUL, alpha=.12)
    if len(reps_a) > 1: ax.fill_between(x, ca - sa, ca + sa, color=ROJO, alpha=.12)
    ax.fill_between(x, cr, ca, color="#999", alpha=.15)
    ax.set_title(f"{NOMBRES.get(ang, ang)} (°)", fontsize=10)
    ax.set_xlabel("% del ciclo"); ax.grid(alpha=.3)
    ax.text(.02, .04, f"DTW: {dtw_g[ang]:.1f}°", transform=ax.transAxes, fontsize=8, color="#444")

axt = fig.add_subplot(gs[2, :])
axt.axis("off")
celdas = [[f["nombre"], f"{f['ref']:.1f} {f['unidad']}", f"{f['act']:.1f} {f['unidad']}",
           f"{f['dif']:+.1f} {f['unidad']}", f"{f['delta_pct']:+.1f} %", "ALERTA" if f["alerta"] else "OK"] for f in filas]
tabla = axt.table(cellText=celdas, colLabels=["Métrica", etq_r, etq_a, "Diferencia", "Δ %", "Alerta"],
                  loc="upper center", cellLoc="center", colWidths=[.28, .16, .16, .13, .1, .1])
tabla.auto_set_font_size(False); tabla.set_fontsize(9); tabla.scale(1, 1.35)
for (r, c), cell in tabla.get_celld().items():
    if r == 0:
        cell.set_facecolor("#0f766e"); cell.set_text_props(color="white", weight="bold")
    elif filas[r - 1]["alerta"]:
        cell.set_facecolor("#fee2e2")
        if c == 5: cell.set_text_props(color=ROJO, weight="bold")
axt.text(0.0, -0.02, resumen, transform=axt.transAxes, fontsize=9, va="top", family="DejaVu Sans",
         bbox=dict(boxstyle="round", fc="#f8fafc", ec="#cbd5e1"))
nombre = f"comparacion_{os.path.basename(a.ref)}{a.rep_ref or ''}_vs_{os.path.basename(a.act)}{a.rep_act or ''}"
plt.savefig(os.path.join(a.salida, nombre + ".png"), dpi=120, bbox_inches="tight")
json.dump({"ref": etq_r, "act": etq_a, "metricas": filas, "dtw_grados": dtw_g, "resumen": resumen},
          open(os.path.join(a.salida, nombre + ".json"), "w"), indent=2, ensure_ascii=False, default=float)
print(resumen)
