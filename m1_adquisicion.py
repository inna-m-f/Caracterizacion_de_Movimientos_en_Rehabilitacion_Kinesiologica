"""
M1 - Adquisición y carga de video (KinePose)

Funciones clave (según Documento de Análisis y Diseño, sección 5.1):
    cargar_video(filepath)              -> dict con metadatos o ValueError (R_VideoInvalido)
    verificar_protocolo_captura(info)   -> dict con el estado de cada condición del protocolo
    extraer_cuadros_rgb(filepath, ...)  -> generador de cuadros en los canales RGB (np.ndarray HxWx3, uint8)
"""
from __future__ import annotations

import os
import cv2
import numpy as np

FORMATOS_VALIDOS = {".mp4", ".avi", ".mov"}

# Parámetros del protocolo de captura (RF1). Ajustar aquí si el equipo cambia el protocolo.
PROTOCOLO = {
    "fps_objetivo": 30.0,
    "fps_tolerancia": 2.0,       # celulares graban 29.97, 30.0, a veces 29.x variable
    "resolucion_min": (1280, 720),
    "duracion_min_s": 2.0,
}


def cargar_video(filepath: str) -> dict:
    """Abre el video, valida formato y devuelve sus metadatos."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in FORMATOS_VALIDOS:
        raise ValueError(f"Formato no soportado: {ext}. Use {sorted(FORMATOS_VALIDOS)}")
    if not os.path.exists(filepath):
        raise ValueError(f"No existe el archivo: {filepath}")

    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)      # videos verticales de celular
    if not cap.isOpened():
        raise ValueError("OpenCV no pudo abrir el video (códec no soportado o archivo dañado).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    h, w = (f0.shape[:2] if ok else (0, 0))
    cap.release()
    if not ok:
        raise ValueError("El video no tiene cuadros legibles.")

    return {
        "ruta": filepath,
        "fps": fps,
        "ancho": w,
        "alto": h,
        "n_cuadros": n,
        "duracion_s": n / fps if fps > 0 else 0.0,
        "orientacion": "vertical" if h > w else "horizontal",
    }


def verificar_protocolo_captura(info: dict, protocolo: dict = PROTOCOLO) -> dict:
    """Chequeos automáticos del protocolo. Las condiciones que no se pueden medir
    desde los metadatos (plano sagital, distancia, iluminación) se validan en el
    checklist manual de la GUI o, después, con M2/M3 (p. ej. cuerpo completo visible)."""
    wmin, hmin = protocolo["resolucion_min"]
    lado_largo, lado_corto = max(info["ancho"], info["alto"]), min(info["ancho"], info["alto"])
    chequeos = {
        "fps_ok": abs(info["fps"] - protocolo["fps_objetivo"]) <= protocolo["fps_tolerancia"],
        "resolucion_ok": lado_largo >= wmin and lado_corto >= hmin,
        "duracion_ok": info["duracion_s"] >= protocolo["duracion_min_s"],
    }
    chequeos["protocolo_ok"] = all(chequeos.values())
    return chequeos


def extraer_cuadros_rgb(filepath: str, paso: int = 1, max_cuadros: int | None = None,
                        ancho_max: int | None = None):
    """Genera (indice, cuadro_rgb). `paso` permite saltar cuadros; `ancho_max`
    reescala para acelerar el procesamiento (el 1080p no es necesario para PDI)."""
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    i = entregados = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % paso == 0:
                if ancho_max and frame.shape[1] > ancho_max:
                    escala = ancho_max / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
                yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                entregados += 1
                if max_cuadros and entregados >= max_cuadros:
                    break
            i += 1
    finally:
        cap.release()


def leer_cuadro(filepath: str, indice: int = 0) -> np.ndarray:
    """Lee un único cuadro RGB (útil en los notebooks de calibración)."""
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"No se pudo leer el cuadro {indice}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def estimar_movimiento_camara(filepath: str, paso: int = 3, ancho: int = 360) -> dict:
    """Chequeo de protocolo 'cámara fija en trípode': estima el desplazamiento global entre
    cuadros con correlación de fase (FFT) sobre la imagen en gris reducida.
    Devuelve el desplazamiento medio y máximo en % del ancho de la imagen."""
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    prev, desp = None, []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % paso == 0:
            g = cv2.cvtColor(cv2.resize(f, (ancho, int(f.shape[0] * ancho / f.shape[1]))), cv2.COLOR_BGR2GRAY)
            g = np.float32(g)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
                desp.append(np.hypot(dx, dy) / ancho * 100)
            prev = g
        i += 1
    cap.release()
    desp = np.array(desp)
    return {"desp_medio_pct": float(desp.mean()), "desp_p95_pct": float(np.percentile(desp, 95)),
            "camara_fija": bool(np.percentile(desp, 95) < 0.5)}


def medir_calidad_imagen(frame_rgb: np.ndarray) -> dict:
    """Nitidez (varianza del Laplaciano) y exposición (brillo medio, % píxeles saturados)."""
    g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return {"nitidez": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "brillo_medio": float(g.mean()),
            "saturados_pct": float(np.mean((g > 250) | (g < 5)) * 100)}
