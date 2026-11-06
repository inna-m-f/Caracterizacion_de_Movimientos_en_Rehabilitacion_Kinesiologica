"""
Pipeline M1 -> M3 sobre un video: pose, ángulos suavizados, repeticiones, gráfico y video anotado.
Uso: python analizar_video.py VIDEO --rotacion ccw --angulo rodilla --salida out/
"""
import argparse, json, os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m1_adquisicion import cargar_video, verificar_protocolo_captura, estimar_movimiento_camara
from m3_pose import (LADO, abrir_video, calcular_angulos_articulares, detectar_repeticiones, dibujar_esqueleto,
                     estimar_pose_video, lado_mas_visible, suavizar_senales_temporales, validar_longitud_segmentos)

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--rotacion", default=None, choices=[None, "cw", "ccw"])
ap.add_argument("--angulo", default="rodilla"); ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--lado", default=None); ap.add_argument("--prominencia", type=float, default=10)
ap.add_argument("--estatico", action="store_true", help="detección cuadro a cuadro sin tracking de MediaPipe")
ap.add_argument("--ejercicio", default=None, help="nombre del ejercicio para el panel")
ap.add_argument("--salida", default="salida"); ap.add_argument("--titulo", default=None)
a = ap.parse_args()
os.makedirs(a.salida, exist_ok=True)
nombre = os.path.splitext(os.path.basename(a.video))[0]

info = cargar_video(a.video)
reporte = {"video": nombre, "info": {k: info[k] for k in ("ancho", "alto", "fps", "duracion_s")},
           "protocolo": verificar_protocolo_captura(info), "camara": estimar_movimiento_camara(a.video)}

r = estimar_pose_video(a.video, rotacion=a.rotacion, modo_estatico=a.estatico)
lm, fps = r["landmarks"], r["fps"]
lado = a.lado or lado_mas_visible(lm)
validos = validar_longitud_segmentos(lm, lado)
lm_val = lm.copy(); lm_val[~validos, :, 3] = 0
ang = calcular_angulos_articulares(lm_val, lado)
series = {k: suavizar_senales_temporales(ang[k], fps) for k in [a.angulo] + a.extra}
reps = detectar_repeticiones(series[a.angulo], fps, "valle", a.prominencia)
reporte.update({"lado": lado, "rotacion": a.rotacion, "modo_estatico": a.estatico,
                "cuadros_con_pose_pct": float(np.mean(~np.isnan(lm[:, 0, 0])) * 100),
                "cuadros_descartados_por_anatomia_pct": float(np.mean(~validos) * 100),
                "angulo_principal": a.angulo, "repeticiones": reps})
np.savez_compressed(os.path.join(a.salida, f"{nombre}_series.npz"), landmarks=lm, fps=fps,
                    **{k: v for k, v in series.items()}, crudo=ang[a.angulo])

# Gráfico
t = np.arange(len(lm)) / fps
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, ang[a.angulo], ".", ms=2, color="#bbb", label=f"{a.angulo} crudo")
colores = ["#1f6feb", "#d97706", "#16a34a"]
for c, (k, v) in zip(colores, series.items()):
    ax.plot(t, v, color=c, lw=2, label=f"{k} suavizado (Savitzky-Golay)")
for k, rp in enumerate(reps):
    ax.axvspan(rp["inicio"] / fps, rp["fin"] / fps, color="#1f6feb", alpha=.06 if k % 2 else .12)
    ax.annotate(f"R{k+1}{'' if rp['completa'] else ' (incompleta)'}\n{rp['angulo_pico']:.0f}°", (rp["pico"] / fps, rp["angulo_pico"]),
                textcoords="offset points", xytext=(0, -28), ha="center", fontsize=8)
ax.set_xlabel("tiempo (s)"); ax.set_ylabel("ángulo (°)")
ax.set_title(a.titulo or f"{nombre} – lado {lado} – {len(reps)} repeticiones")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(a.salida, f"{nombre}_angulos.png"), dpi=120); plt.close()

# Video anotado (pantalla 3 del prototipo)
from m3_visualizacion import renderizar_video_anotado
renderizar_video_anotado(a.video, lm_val, series, reps, lado, a.angulo,
                         os.path.join(a.salida, f"{nombre}_anotado.mp4"), fps, r["tam"],
                         titulo=a.ejercicio or nombre)
json.dump(reporte, open(os.path.join(a.salida, f"{nombre}_reporte.json"), "w"), indent=2, default=float)
print(json.dumps({k: reporte[k] for k in ("video", "lado", "cuadros_con_pose_pct", "cuadros_descartados_por_anatomia_pct")}, default=float))
for k, rp in enumerate(reps):
    print(f"  R{k+1}: pico {rp['angulo_pico']:.0f}°  ROM {rp['rom']:.0f}°  dur {rp['duracion_s']:.1f}s  completa={rp['completa']}")
