"""
Evalúa los métodos de segmentación de M2 sobre un video con verdad de terreno.
Uso: python evaluar_m2.py sentadilla_sintetica.mp4 sentadilla_sintetica_gt.npz salida/
Genera: tabla de IoU, figura comparativa y video con la máscara superpuesta.
"""
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m1_adquisicion import extraer_cuadros_rgb  # noqa: E402
from m2_segmentacion import (SegmentadorFondo, dibujar_resultado, iou,  # noqa: E402
                             modelo_fondo_mediana, segmentar_sujeto)

video, gt_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
datos = np.load(gt_path)
gt, n_fondo = datos["gt"], int(datos["n_fondo"])

cuadros = [f for _, f in extraer_cuadros_rgb(video)]
fondo = modelo_fondo_mediana(cuadros[:n_fondo])

metodos = {
    "Umbral fijo 100 (notebook)": lambda f: segmentar_sujeto(f, 100)[0],
    "Umbral Otsu + morfología": lambda f: segmentar_sujeto(f, None)[0],
}
seg_dif = SegmentadorFondo(fondo, metodo="diferencia", umbral=25)
seg_mog = SegmentadorFondo(metodo="mog2", umbral=16)
for f in cuadros[:n_fondo]:            # MOG2 adaptativo: aprende siempre (también a la persona)
    seg_mog.segmentar(f)
seg_mog_c = SegmentadorFondo(metodo="mog2", umbral=16)
seg_mog_c.aprender_fondo(cuadros[:n_fondo], congelar=True)
metodos["Diferencia con fondo (Lab + sombras)"] = lambda f: seg_dif.segmentar(f)[0]
metodos["MOG2 adaptativo"] = lambda f: seg_mog.segmentar(f)[0]
metodos["MOG2 con fondo congelado"] = lambda f: seg_mog_c.segmentar(f)[0]

resultados = {k: [] for k in metodos}
tiempos = {k: 0.0 for k in metodos}
idx_fig = n_fondo + 37
mascaras_fig = {}
for i in range(n_fondo, len(cuadros)):
    for nombre, fn in metodos.items():
        t0 = time.perf_counter()
        m = fn(cuadros[i])
        tiempos[nombre] += time.perf_counter() - t0
        resultados[nombre].append(iou(m, gt[i]))
        if i == idx_fig:
            mascaras_fig[nombre] = m

n = len(cuadros) - n_fondo
print(f"{'Método':40s} {'IoU medio':>9s} {'IoU min':>8s} {'ms/cuadro':>10s}")
for k, v in resultados.items():
    print(f"{k:40s} {np.mean(v):9.3f} {np.min(v):8.3f} {1000 * tiempos[k] / n:10.1f}")

# Figura comparativa
fig, ax = plt.subplots(2, 3, figsize=(16, 7))
ax[0, 0].imshow(cuadros[idx_fig]); ax[0, 0].set_title("Cuadro original")
ax[0, 1].imshow(fondo); ax[0, 1].set_title("Modelo de fondo (mediana 2 s vacíos)")
ax[0, 2].imshow(gt[idx_fig], cmap="gray"); ax[0, 2].set_title("Verdad de terreno")
for a, (k, m) in zip(ax[1], list(mascaras_fig.items())[:3]):
    a.imshow(m, cmap="gray")
    a.set_title(f"{k}\nIoU = {iou(m, gt[idx_fig]):.2f}", fontsize=10)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out, "comparacion_metodos_m2.png"), dpi=110)

# Curva IoU por cuadro
plt.figure(figsize=(10, 3.5))
for k, v in resultados.items():
    plt.plot(v, label=k)
plt.ylim(0, 1.02); plt.xlabel("cuadro (desde que entra el sujeto)"); plt.ylabel("IoU")
plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(out, "iou_por_cuadro.png"), dpi=110)

# Video de salida con el mejor método
seg_dif = SegmentadorFondo(fondo, metodo="diferencia", umbral=25)
H, W = cuadros[0].shape[:2]
vw = cv2.VideoWriter(os.path.join(out, "m2_segmentacion_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in cuadros[n_fondo:]:
    m, bbox, met = seg_dif.segmentar(f)
    o = dibujar_resultado(f, m, bbox)
    cv2.putText(o, f"Area sujeto {met['area_sujeto_pct']:.1f}%  Ruido {met['ruido_fondo_px']} px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
vw.release()
print("Salidas en", out)
