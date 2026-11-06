"""
M3 - Estimación de pose y tracking articular (KinePose)

    estimar_pose_video(ruta)            -> dict con landmarks (T,33,4) en píxeles + visibilidad,
                                           máscara de segmentación DL (reducida) y metadatos
    calcular_angulos_articulares(lm)    -> series de ángulos (rodilla, cadera, tronco, pierna)
    suavizar_senales_temporales(serie)  -> interpolación de huecos + Savitzky-Golay
    validar_longitud_segmentos(lm)      -> descarta cuadros con segmentos anatómicamente imposibles

Usa MediaPipe Pose (BlazePose, 33 landmarks).
"""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter

L = mp.solutions.pose.PoseLandmark
# índices por lado
LADO = {
    "izq": dict(hombro=L.LEFT_SHOULDER, cadera=L.LEFT_HIP, rodilla=L.LEFT_KNEE, tobillo=L.LEFT_ANKLE,
                talon=L.LEFT_HEEL, pie=L.LEFT_FOOT_INDEX),
    "der": dict(hombro=L.RIGHT_SHOULDER, cadera=L.RIGHT_HIP, rodilla=L.RIGHT_KNEE, tobillo=L.RIGHT_ANKLE,
                talon=L.RIGHT_HEEL, pie=L.RIGHT_FOOT_INDEX),
}
CONEXIONES = mp.solutions.pose.POSE_CONNECTIONS


def abrir_video(ruta: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(ruta)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)     # videos de celular grabados en vertical
    return cap


def _rotar(img, rot):
    if rot == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _desrotar_puntos(xy, rot, w, h):
    """Lleva puntos (N,2) de la imagen rotada a la imagen original de tamaño (w,h)."""
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if rot == "cw":       # rotada: x' = h-1-y, y' = x
        return np.stack([y, h - 1 - x], 1)
    if rot == "ccw":      # rotada: x' = y, y' = w-1-x
        return np.stack([w - 1 - y, x], 1)
    return xy


def estimar_pose_video(ruta: str, ancho_max: int = 720, complejidad: int = 1, roi_fn=None,
                       escala_mascara: float = 0.25, callback=None, rotacion: str | None = None,
                       max_cuadros: int | None = None, paso: int = 1, modo_estatico: bool = False) -> dict:
    """Corre MediaPipe Pose cuadro a cuadro.
    roi_fn(frame_rgb, i) -> (x,y,w,h) opcional: si se entrega, la pose se estima dentro del
    recorte (ROI de M2) y se re-proyecta a coordenadas de la imagen completa."""
    cap = abrir_video(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # modo_estatico=True: detección independiente en cada cuadro (sin tracking interno).
    # Más lento, pero un cuadro malo no contamina a los siguientes y el resultado es reproducible.
    pose = mp.solutions.pose.Pose(static_image_mode=modo_estatico, model_complexity=complejidad,
                                  enable_segmentation=True, smooth_landmarks=not modo_estatico,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    lms, mascaras = [], []
    i = 0
    tam = None
    while True:
        ok, bgr = cap.read()
        if not ok or (max_cuadros and len(lms) >= max_cuadros):
            break
        if i % paso:
            i += 1
            continue
        if bgr.shape[1] > ancho_max:
            s = ancho_max / bgr.shape[1]
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        tam = (W, H)
        x0, y0, sub = 0, 0, rgb
        if roi_fn is not None:
            roi = roi_fn(rgb, i)
            if roi is not None:
                x0, y0, w, h = roi
                sub = np.ascontiguousarray(rgb[y0:y0 + h, x0:x0 + w])
        # Rotación geométrica: MediaPipe está entrenado con personas de pie; en ejercicios
        # acostados, girar 90° el cuadro mejora mucho la detección.
        h_, w_ = sub.shape[:2]
        sub_r = np.ascontiguousarray(_rotar(sub, rotacion))
        res = pose.process(sub_r)
        arr = np.full((33, 4), np.nan, np.float32)
        mask = np.zeros((int(H * escala_mascara), int(W * escala_mascara)), np.uint8)
        if res.pose_landmarks:
            hr, wr = sub_r.shape[:2]
            raw = np.array([(p.x * wr, p.y * hr, p.z * wr, p.visibility) for p in res.pose_landmarks.landmark])
            raw[:, :2] = _desrotar_puntos(raw[:, :2], rotacion, w_, h_)
            raw[:, 0] += x0
            raw[:, 1] += y0
            arr[:] = raw
        if res.segmentation_mask is not None:
            full = np.zeros((H, W), np.float32)
            seg = res.segmentation_mask
            if rotacion == "cw":
                seg = cv2.rotate(seg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotacion == "ccw":
                seg = cv2.rotate(seg, cv2.ROTATE_90_CLOCKWISE)
            full[y0:y0 + h_, x0:x0 + w_] = seg
            mask = cv2.resize((full > 0.5).astype(np.uint8) * 255, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        # visibilidad efectiva = 0 si el punto cae fuera de la imagen (MediaPipe lo "imagina")
        fuera = (arr[:, 0] < 0) | (arr[:, 0] >= W) | (arr[:, 1] < 0) | (arr[:, 1] >= H)
        arr[fuera, 3] = np.minimum(arr[fuera, 3], 0.0)
        lms.append(arr)
        mascaras.append(mask)
        if callback:
            callback(i, rgb, arr)
        i += 1
    cap.release()
    pose.close()
    return {"landmarks": np.stack(lms), "mascaras_dl": np.stack(mascaras), "fps": fps / paso,
            "tam": tam, "ruta": ruta, "rotacion": rotacion}


def elegir_rotacion(ruta: str, n_muestra: int = 60, paso: int = 5, **kw) -> tuple[str | None, dict]:
    """Prueba las 3 orientaciones sobre una muestra de cuadros y elige la que da más
    detecciones con mejor visibilidad media de cadera-rodilla-tobillo."""
    puntajes = {}
    piernas = [23, 24, 25, 26, 27, 28]
    for rot in (None, "cw", "ccw"):
        r = estimar_pose_video(ruta, rotacion=rot, max_cuadros=n_muestra, paso=paso, **kw)
        v = r["landmarks"][:, piernas, 3]
        puntajes[rot] = float(np.nan_to_num(v, nan=0).mean())
    return max(puntajes, key=puntajes.get), puntajes


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def angulo_3p(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ángulo ABC en grados (vértice en b) para arreglos (T,2). 180° = segmento recto."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def angulo_con_referencia(p_ini: np.ndarray, p_fin: np.ndarray, ref_deg: float = 0.0) -> np.ndarray:
    """Inclinación del segmento p_ini->p_fin respecto a una dirección de referencia (grados,
    0 = horizontal en la imagen). Devuelve el ángulo en [0, 180)."""
    d = p_fin - p_ini
    ang = np.degrees(np.arctan2(-d[:, 1], d[:, 0]))      # eje y de la imagen hacia abajo
    return np.abs(((ang - ref_deg) + 180) % 360 - 180)


def lado_mas_visible(lm: np.ndarray) -> str:
    """Elige el lado (izq/der) cuyos landmarks de pierna son más visibles a la cámara."""
    vis = {s: np.nanmean(lm[:, [int(LADO[s][k]) for k in ("cadera", "rodilla", "tobillo")], 3]) for s in LADO}
    return max(vis, key=vis.get)


def calcular_angulos_articulares(lm: np.ndarray, lado: str | None = None, vis_min: float = 0.5,
                                 ref_horizontal_deg: float = 0.0) -> dict:
    """Ángulos por cuadro (NaN si los puntos no son confiables).
    rodilla: cadera-rodilla-tobillo (180 = extendida)
    cadera : hombro-cadera-rodilla  (180 = extendida)
    tronco : inclinación hombro-cadera respecto a la vertical
    muslo / pierna: inclinación del segmento respecto a la horizontal de referencia
                    (p. ej. la camilla detectada con Hough), útil en ejercicios acostados."""
    lado = lado or lado_mas_visible(lm)
    idx = {k: int(v) for k, v in LADO[lado].items()}
    P = lambda k: lm[:, idx[k], :2]
    V = lambda *ks: np.all([lm[:, idx[k], 3] >= vis_min for k in ks], axis=0)

    def mask(serie, *ks):
        s = serie.astype(np.float64)
        s[~V(*ks)] = np.nan
        return s

    out = {
        "lado": lado,
        "rodilla": mask(angulo_3p(P("cadera"), P("rodilla"), P("tobillo")), "cadera", "rodilla", "tobillo"),
        "cadera": mask(angulo_3p(P("hombro"), P("cadera"), P("rodilla")), "hombro", "cadera", "rodilla"),
        "tronco": mask(angulo_con_referencia(P("cadera"), P("hombro"), ref_horizontal_deg + 90), "hombro", "cadera"),
        "muslo": mask(angulo_con_referencia(P("cadera"), P("rodilla"), ref_horizontal_deg), "cadera", "rodilla"),
        "pierna": mask(angulo_con_referencia(P("rodilla"), P("tobillo"), ref_horizontal_deg), "rodilla", "tobillo"),
    }
    return out


def validar_longitud_segmentos(lm: np.ndarray, lado: str, tol: float = 0.35) -> np.ndarray:
    """Regla de consistencia anatómica: la RAZÓN muslo/tibia es propia de la persona y no
    depende de la distancia a la cámara (invariante a escala, sirve aunque la cámara se mueva).
    Marca como inválido el cuadro cuya razón se aleja más de `tol` (relativo) de la mediana."""
    idx = {k: int(v) for k, v in LADO[lado].items()}
    muslo = np.linalg.norm(lm[:, idx["cadera"], :2] - lm[:, idx["rodilla"], :2], axis=1)
    tibia = np.linalg.norm(lm[:, idx["rodilla"], :2] - lm[:, idx["tobillo"], :2], axis=1)
    razon = muslo / (tibia + 1e-6)
    med = np.nanmedian(razon)
    validos = ~(np.abs(razon - med) > tol * med)
    return validos | np.isnan(razon)     # sin datos no es "inválido": ya es NaN


def suavizar_senales_temporales(serie: np.ndarray, fps: float = 30, ventana_s: float = 0.33,
                                orden: int = 2, hueco_max_s: float = 0.5) -> np.ndarray:
    """Interpola huecos cortos (pérdidas de tracking), elimina picos aislados con un filtro
    de mediana (outliers de MediaPipe) y luego aplica Savitzky-Golay (suaviza sin desfasar)."""
    from scipy.signal import medfilt
    s = np.asarray(serie, np.float64).copy()
    n = len(s)
    ok = ~np.isnan(s)
    if ok.sum() < 5:
        return s
    t = np.arange(n)
    interp = np.interp(t, t[ok], s[ok])
    # no inventar datos en huecos largos
    hueco_max = int(hueco_max_s * fps)
    run = 0
    largos = np.zeros(n, bool)
    for i in range(n + 1):
        if i < n and not ok[i]:
            run += 1
        else:
            if run > hueco_max:
                largos[i - run:i] = True
            run = 0
    win = max(5, int(ventana_s * fps) | 1)
    out = savgol_filter(medfilt(interp, 7), win, orden)
    out[largos] = np.nan
    out[: np.argmax(ok)] = np.nan
    out[n - np.argmax(ok[::-1]):] = np.nan
    return out


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def dibujar_esqueleto(img, lm_frame, vis_min=0.5, color=(0, 255, 200), grosor=3, destacar=None):
    out = img.copy()
    pts = lm_frame[:, :2]
    vis = lm_frame[:, 3]
    for a, b in CONEXIONES:
        if vis[a] >= vis_min and vis[b] >= vis_min and not np.isnan(pts[[a, b]]).any():
            cv2.line(out, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color, grosor, cv2.LINE_AA)
    for k in range(33):
        if vis[k] >= vis_min and not np.isnan(pts[k]).any():
            r = 7 if destacar and k in destacar else 4
            cv2.circle(out, tuple(map(int, pts[k])), r, (255, 80, 80) if destacar and k in destacar else color, -1)
    return out


def detectar_repeticiones(serie: np.ndarray, fps: float, tipo: str = "valle", prominencia: float = 10,
                          dist_min_s: float = 1.0, dur_min_s: float = 1.5) -> list[dict]:
    """Segmenta el ejercicio en repeticiones a partir de la curva angular suavizada.
    tipo='valle': el punto de máximo esfuerzo es el ángulo mínimo (flexión).
    Cada repetición va de un extremo opuesto al siguiente, pasando por el pico detectado."""
    from scipy.signal import find_peaks
    s = np.asarray(serie, np.float64)
    ok = ~np.isnan(s)
    if ok.sum() < 10:
        return []
    x = np.interp(np.arange(len(s)), np.flatnonzero(ok), s[ok])
    sig = -x if tipo == "valle" else x
    picos, _ = find_peaks(sig, prominence=prominencia, distance=int(dist_min_s * fps))
    reps = []
    for k, p in enumerate(picos):
        ini = picos[k - 1] if k > 0 else 0
        fin = picos[k + 1] if k + 1 < len(picos) else len(s) - 1
        a = ini + int(np.argmax(-sig[ini:p] if p > ini else [0]))      # extremo opuesto antes
        b = p + int(np.argmax(-sig[p:fin + 1]))                         # extremo opuesto después
        seg = s[a:b + 1]
        bajada, subida = abs(x[a] - x[p]), abs(x[b] - x[p])
        reps.append({"inicio": int(a), "pico": int(p), "fin": int(b),
                     "angulo_pico": float(x[p]), "rom": float(np.nanmax(seg) - np.nanmin(seg)),
                     "duracion_s": (b - a) / fps,
                     # completa = vuelve al menos al 60 % de la posición inicial
                     "completa": bool(subida >= 0.6 * bajada and bajada >= 0.6 * subida)})
    return [r for r in reps if r["duracion_s"] >= dur_min_s]
