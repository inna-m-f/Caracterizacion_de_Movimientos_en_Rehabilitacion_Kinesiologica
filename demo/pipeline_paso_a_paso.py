"""
Muestra, para UN cuadro de un video real, cada etapa del pipeline PDI de KinePose:
M1 (cuadro) -> M2 (preprocesamiento, segmentación clásica y con deep learning, morfología, ROI)
-> M3 (MediaPipe sobre la ROI, ángulos).

Uso: python pipeline_paso_a_paso.py VIDEO INDICE_CUADRO salida.png [--rotacion cw|ccw] [--lado der|izq]
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m2_segmentacion import (aplicar_binarizacion, componente_principal, convertir_espacio_color,  # noqa: E402
                             filtrar_ruido_morfologico, iou, obtener_roi_sujeto, rellenar_huecos)
from m3_pose import LADO, abrir_video, angulo_3p  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("indice", type=int); ap.add_argument("salida")
ap.add_argument("--lado", default="der"); ap.add_argument("--ancho", type=int, default=720)
a = ap.parse_args()

# ---------------- M1: cuadro con orientación corregida ----------------
cap = abrir_video(a.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.indice)
ok, bgr = cap.read()
cap.release()
s = a.ancho / bgr.shape[1]
bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# ---------------- M2a: preprocesamiento ----------------
gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
pre = convertir_espacio_color(rgb, blur=5, clahe=True)          # CLAHE + Gauss

# ---------------- M2b: segmentación clásica ----------------
otsu, t_otsu = aplicar_binarizacion(pre, None, invertir=True)    # sujeto oscuro sobre fondo claro
bordes = cv2.Canny(pre, 40, 120)

# ---------------- M2c: segmentación deep learning (MediaPipe) ----------------
with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True) as pose:
    res = pose.process(rgb)
dl = (res.segmentation_mask > 0.5).astype(np.uint8) * 255

# ---------------- M2d: refinamiento clásico guiado por DL (GrabCut) ----------------
# La máscara DL da una "semilla": seguro-sujeto (erosionada), seguro-fondo (lejos de la máscara),
# y una banda de duda donde GrabCut decide por color (modelos de mezcla gaussiana).
gc = np.full((H, W), cv2.GC_BGD, np.uint8)
dil = cv2.dilate(dl, np.ones((41, 41), np.uint8))
ero = cv2.erode(dl, np.ones((15, 15), np.uint8))
gc[dil > 0] = cv2.GC_PR_BGD
gc[dl > 0] = cv2.GC_PR_FGD
gc[ero > 0] = cv2.GC_FGD
bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(bgr, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
grab = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

# ---------------- M2e: morfología + componente mayor + ROI ----------------
final = rellenar_huecos(componente_principal(filtrar_ruido_morfologico(grab, 5, 15)))
bbox = obtener_roi_sujeto(final, margen=0.06)
x, y, w, h = bbox

# ---------------- M3: MediaPipe sobre la ROI + ángulos ----------------
recorte = np.ascontiguousarray(rgb[y:y + h, x:x + w])
with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
    r2 = pose.process(recorte)
lm = np.array([(x + p.x * w, y + p.y * h, p.visibility) for p in r2.pose_landmarks.landmark])
idx = {k: int(v) for k, v in LADO[a.lado].items()}
P = lambda k: lm[idx[k], :2][None]
cad = 180 - angulo_3p(P("hombro"), P("cadera"), P("rodilla"))[0]
rod = 180 - angulo_3p(P("cadera"), P("rodilla"), P("tobillo"))[0]
tro = None

esq = rgb.copy()
cadena = [idx[k] for k in ("hombro", "cadera", "rodilla", "tobillo", "talon", "pie")] + [idx["tobillo"]]
cv2.rectangle(esq, (x, y), (x + w, y + h), (255, 80, 80), 3)
for p1, p2 in zip(cadena[:-1], cadena[1:]):
    cv2.line(esq, tuple(map(int, lm[p1, :2])), tuple(map(int, lm[p2, :2])), (77, 196, 196), 6, cv2.LINE_AA)
for k in cadena[:-1]:
    cv2.circle(esq, tuple(map(int, lm[k, :2])), 9, (255, 255, 255), -1)
for k, txt in ((idx["cadera"], f"{cad:.0f}°"), (idx["rodilla"], f"{rod:.0f}°")):
    pass

superp = rgb.copy()
capa = np.zeros_like(superp); capa[final > 0] = (77, 196, 196)
superp = cv2.addWeighted(superp, 1, capa, .45, 0)
cv2.rectangle(superp, (x, y), (x + w, y + h), (255, 80, 80), 3)

# ---------------- figura ----------------
pasos = [
    (rgb, None, "1 · Cuadro original (M1)", "Orientación corregida, 720 px de ancho"),
    (pre, "gray", "2 · Gris + CLAHE + Gauss", "Contraste local y menos ruido"),
    (otsu, "gray", f"3 · Umbral Otsu (t={t_otsu})", f"Clásico: IoU vs final {iou(otsu, final):.2f}"),
    (bordes, "gray", "4 · Bordes Canny", "Contornos: silueta + puerta, radiador..."),
    (dl, "gray", "5 · Máscara deep learning", f"MediaPipe · IoU vs final {iou(dl, final):.2f}"),
    (grab, "gray", "6 · GrabCut guiado por DL", "Refina el borde por color (GMM)"),
    (final, "gray", "7 · Apertura + cierre + componente", "Silueta final limpia"),
    (superp, None, "8 · Silueta + ROI", f"ROI {w}×{h} px = {100 * w * h / (W * H):.0f} % del cuadro"),
    (recorte, None, "9 · Recorte ROI → MediaPipe", "La persona ocupa todo el recorte"),
    (esq, None, "10 · Esqueleto y ángulos (M3)", f"Cadera {cad:.0f}° · Rodilla {rod:.0f}° de flexión"),
]
fig, axs = plt.subplots(2, 5, figsize=(20, 15.5))
for ax, (img, cmap, tit, sub) in zip(axs.ravel(), pasos):
    ax.imshow(img, cmap=cmap)
    ax.set_title(tit, fontsize=13, weight="bold", loc="left")
    ax.text(0, -0.03, sub, transform=ax.transAxes, fontsize=10.5, va="top", color="#444")
    ax.axis("off")
fig.suptitle(f"Pipeline KinePose paso a paso · {os.path.basename(a.video)} · cuadro {a.indice}", fontsize=17, weight="bold")
plt.subplots_adjust(left=.02, right=.98, top=.92, bottom=.04, wspace=.08, hspace=.22)
plt.savefig(a.salida, dpi=100)
print(f"Otsu IoU={iou(otsu, final):.2f}  DL IoU={iou(dl, final):.2f}  ROI={bbox}  cadera={cad:.1f}  rodilla={rod:.1f}")
