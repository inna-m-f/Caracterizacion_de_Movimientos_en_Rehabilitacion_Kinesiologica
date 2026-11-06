"""
Video anotado para el kinesiólogo (pantalla 3 del prototipo: "Estimación de pose").

En el cuadro: solo la cadena que se mide (hombro-cadera-rodilla-tobillo-pie) del lado
analizado, arco goniométrico en cadera y rodilla, y trayectoria reciente de la rodilla.
En el panel lateral: ángulos clínicos en vivo, fase del movimiento, contador de
repeticiones y la curva del ángulo principal con el cursor de tiempo.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from m3_pose import LADO, abrir_video

# Paleta (RGB) alineada con el prototipo de interfaz
FONDO = (13, 22, 26)
TARJETA = (22, 34, 40)
TEAL = (77, 196, 196)
TEXTO = (232, 240, 242)
TENUE = (140, 160, 166)
NARANJA = (245, 158, 11)
VERDE = (74, 222, 128)

NOMBRE_CLINICO = {"cadera": "Flexión de cadera", "rodilla": "Flexión de rodilla", "tronco": "Inclinación del tronco"}
ES_FLEXION = {"cadera", "rodilla"}


def _fuente(tam, negrita=False):
    candidatos = ["/usr/share/fonts/opentype/inter/Inter-SemiBold.otf" if negrita else "/usr/share/fonts/opentype/inter/Inter-Regular.otf"]
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        candidatos.append(os.path.join(base, "DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"))
    except Exception:
        pass
    for c in candidatos:
        if os.path.exists(c):
            return ImageFont.truetype(c, tam)
    return ImageFont.load_default()


def _clinico(nombre, v):
    return 180.0 - v if nombre in ES_FLEXION else v


def _arco(img, vertice, p1, p2, radio, color, alpha=0.35):
    """Sector circular entre los segmentos vertice->p1 y vertice->p2 (arco goniométrico)."""
    a1 = np.degrees(np.arctan2(p1[1] - vertice[1], p1[0] - vertice[0]))
    a2 = np.degrees(np.arctan2(p2[1] - vertice[1], p2[0] - vertice[0]))
    d = (a2 - a1 + 540) % 360 - 180
    capa = img.copy()
    c = tuple(int(v) for v in vertice)
    cv2.ellipse(capa, c, (radio, radio), 0, a1, a1 + d, color, -1, cv2.LINE_AA)
    cv2.addWeighted(capa, alpha, img, 1 - alpha, 0, img)
    cv2.ellipse(img, c, (radio, radio), 0, a1, a1 + d, color, 2, cv2.LINE_AA)
    medio = np.radians(a1 + d / 2)
    return (int(vertice[0] + (radio + 34) * np.cos(medio)), int(vertice[1] + (radio + 34) * np.sin(medio)))


def renderizar_video_anotado(ruta_video: str, landmarks: np.ndarray, series: dict, reps: list[dict], lado: str,
                             principal: str, ruta_salida: str, fps: float, tam: tuple, titulo: str = "",
                             ancho_panel: int = 460, vis_min: float = 0.5, rastro_s: float = 1.5):
    W, H = tam
    idx = {k: int(v) for k, v in LADO[lado].items()}
    cadena = [idx[k] for k in ("hombro", "cadera", "rodilla", "tobillo", "talon", "pie")] + [idx["tobillo"]]
    nombres = [principal] + [k for k in series if k != principal]
    clin = {k: _clinico(k, series[k]) for k in nombres}
    vel = np.gradient(np.nan_to_num(series[principal], nan=np.nanmean(series[principal]))) * fps
    T = len(landmarks)

    # Pre-render del gráfico del panel (curva completa del ángulo principal)
    gx0, gy0, gw, gh = 24, H - 330, ancho_panel - 48, 220
    cp = clin[principal]
    lo, hi = np.nanmin(cp) - 5, np.nanmax(cp) + 5
    to_xy = lambda i, v: (gx0 + i / max(T - 1, 1) * gw, gy0 + gh - (v - lo) / (hi - lo) * gh)

    f_tit, f_sub, f_lab = _fuente(34, True), _fuente(20), _fuente(19)
    f_val, f_big, f_small = _fuente(44, True), _fuente(58, True), _fuente(16)

    cap = abrir_video(ruta_video)
    vw = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + ancho_panel, H))
    rastro = []
    for i in range(T):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (W, H))
        lm = landmarks[i]
        vis = lm[:, 3] >= vis_min
        P = lambda k: lm[idx[k], :2]
        # --- cadena articular ---
        for a, b in zip(cadena[:-1], cadena[1:]):
            if vis[a] and vis[b]:
                cv2.line(f, tuple(map(int, lm[a, :2])), tuple(map(int, lm[b, :2])), TEAL[::-1], 6, cv2.LINE_AA)
        # --- rastro de la rodilla ---
        if vis[idx["rodilla"]]:
            rastro.append(P("rodilla").copy())
        rastro = rastro[-int(rastro_s * fps):]
        for j in range(1, len(rastro)):
            a = j / len(rastro)
            cv2.line(f, tuple(map(int, rastro[j - 1])), tuple(map(int, rastro[j])),
                     tuple(int(c * a) for c in NARANJA[::-1]), 3, cv2.LINE_AA)
        # --- arcos ---
        etiquetas = []
        # Flexión como con goniómetro: ángulo entre la PROLONGACIÓN del segmento proximal
        # (línea punteada) y el segmento distal. 0° = articulación extendida.
        for art, (prox, dist) in {"cadera": ("hombro", "rodilla"), "rodilla": ("cadera", "tobillo")}.items():
            if art in clin and all(vis[idx[k]] for k in (art, prox, dist)) and not np.isnan(clin[art][i]):
                v = P(art)
                u = (v - P(prox)) / (np.linalg.norm(v - P(prox)) + 1e-6)
                ref = v + u * 150
                for t in np.linspace(0, 1, 9)[::2]:
                    a_, b_ = v + u * 150 * t, v + u * 150 * min(t + 1 / 8, 1)
                    cv2.line(f, tuple(map(int, a_)), tuple(map(int, b_)), (230, 230, 230), 2, cv2.LINE_AA)
                pos = _arco(f, v, ref, P(dist), 70, (NARANJA if art == principal else TEAL)[::-1])
                etiquetas.append((pos, f"{clin[art][i]:.0f}°"))
        for k in cadena[:-1]:
            if vis[k]:
                cv2.circle(f, tuple(map(int, lm[k, :2])), 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(f, tuple(map(int, lm[k, :2])), 9, TEAL[::-1], 3, cv2.LINE_AA)

        # --- lienzo completo + panel (texto con PIL para tildes) ---
        lienzo = np.zeros((H, W + ancho_panel, 3), np.uint8)
        lienzo[:, :W] = f
        img = Image.fromarray(cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.rectangle([W, 0, W + ancho_panel, H], fill=FONDO)
        for pos, txt in etiquetas:
            x, y = pos
            d.rounded_rectangle([x - 34, y - 18, x + 34, y + 18], 8, fill=(0, 0, 0))
            d.text((x, y), txt, font=f_lab, fill=TEXTO, anchor="mm")
        X = W + 24
        d.rounded_rectangle([X, 26, X + 34, 60], 9, fill=TEAL)
        d.text((X + 46, 43), "KinePose", font=f_tit, fill=TEXTO, anchor="lm")
        d.text((X, 78), titulo, font=f_sub, fill=TENUE)
        y = 128
        for k in nombres:
            v = clin[k][i]
            col = NARANJA if k == principal else TEAL
            d.rounded_rectangle([X, y, X + ancho_panel - 48, y + 96], 14, fill=TARJETA)
            d.text((X + 18, y + 14), NOMBRE_CLINICO.get(k, k), font=f_lab, fill=TENUE)
            d.text((X + 18, y + 40), "--" if np.isnan(v) else f"{v:.0f}°", font=f_val, fill=col)
            y += 110
        # fase y repeticiones
        n_rep = sum(1 for r in reps if r["fin"] <= i and r["completa"])
        fase = "Pausa" if abs(vel[i]) < 8 else ("Flexionando" if vel[i] < 0 else "Volviendo")
        d.rounded_rectangle([X, y + 6, X + 196, y + 150], 14, fill=TARJETA)
        d.text((X + 18, y + 20), "Repeticiones", font=f_lab, fill=TENUE)
        d.text((X + 18, y + 48), f"{n_rep}", font=f_big, fill=TEXTO)
        d.rounded_rectangle([X + 212, y + 6, X + ancho_panel - 48, y + 150], 14, fill=TARJETA)
        d.text((X + 230, y + 20), "Fase", font=f_lab, fill=TENUE)
        d.text((X + 230, y + 66), fase, font=_fuente(28, True), fill=VERDE if fase != "Pausa" else TENUE)
        # repeticiones terminadas
        yr = y + 172
        for k, r in enumerate(reps):
            if r["fin"] > i:
                break
            cr = clin[principal][r["inicio"]:r["fin"] + 1]
            txt = f"R{k + 1}   máx {np.nanmax(cr):.0f}°   ROM {np.nanmax(cr) - np.nanmin(cr):.0f}°   {r['duracion_s']:.1f} s"
            d.text((X, yr), txt + ("" if r["completa"] else "  (incompleta)"), font=f_lab,
                   fill=TEXTO if r["completa"] else TENUE)
            yr += 32
        # gráfico
        d.text((gx0 + W, gy0 - 34), f"{NOMBRE_CLINICO.get(principal, principal)} en el tiempo", font=f_lab, fill=TENUE)
        d.rounded_rectangle([gx0 + W - 10, gy0 - 8, gx0 + W + gw + 10, gy0 + gh + 8], 12, fill=TARJETA)
        for r in reps:
            xa, xb = to_xy(r["inicio"], 0)[0] + W, to_xy(r["fin"], 0)[0] + W
            d.rectangle([xa, gy0, xb, gy0 + gh], fill=(28, 46, 52) if r["completa"] else (30, 36, 40))
        pts = [(to_xy(j, v)[0] + W, to_xy(j, v)[1]) for j, v in enumerate(cp) if not np.isnan(v)]
        if len(pts) > 1:
            d.line(pts, fill=NARANJA, width=3)
        cx = to_xy(i, 0)[0] + W
        d.line([(cx, gy0), (cx, gy0 + gh)], fill=TEXTO, width=2)
        d.text((gx0 + W, gy0 + gh + 18), f"{i / fps:4.1f} s", font=f_small, fill=TENUE)
        d.text((gx0 + W + gw, gy0 + gh + 18), f"lado {'derecho' if lado == 'der' else 'izquierdo'}",
               font=f_small, fill=TENUE, anchor="ra")
        vw.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    cap.release()
    vw.release()
