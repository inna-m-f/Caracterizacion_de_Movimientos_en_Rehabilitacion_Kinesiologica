"""
M2 - Preprocesamiento y segmentación PDI (KinePose)

Compatible con el notebook de calibración existente:
    mascara, bbox = segmentar_sujeto(frame_rgb, UMBRAL)

Pipeline por cuadro (sección 5.2 del documento):
    RGB -> gris -> suavizado gaussiano -> binarización (umbral manual u Otsu)
        -> apertura (quita manchas) -> cierre (rellena huecos)
        -> componente conexa más grande (= el sujeto) -> ROI con margen -> métricas

Para videos reales (fondo de casa, sombras, ropa de colores) el umbral global
suele fallar; por eso se incluye además SegmentadorFondo, que compara cada
cuadro contra un modelo del fondo vacío (cámara fija en trípode = supuesto del protocolo).
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bloques básicos
# ---------------------------------------------------------------------------
def convertir_espacio_color(frame_rgb: np.ndarray, blur: int = 5, clahe: bool = False) -> np.ndarray:
    """RGB -> gris suavizado. CLAHE opcional para videos con poca luz o contraluz."""
    gris = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    if clahe:
        gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    if blur and blur > 1:
        gris = cv2.GaussianBlur(gris, (blur | 1, blur | 1), 0)
    return gris


def aplicar_binarizacion(gris: np.ndarray, umbral: int | None = None, invertir: bool = True):
    """Devuelve (mascara 0/255, umbral_usado).
    umbral=None -> Otsu elige el umbral automáticamente.
    invertir=True -> sujeto oscuro sobre fondo claro (caso de los videos de prueba)."""
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    if umbral is None:
        t, mask = cv2.threshold(gris, 0, 255, tipo | cv2.THRESH_OTSU)
    else:
        t, mask = cv2.threshold(gris, int(umbral), 255, tipo)
    return mask, int(t)


def _kernel(tam: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))


def filtrar_ruido_morfologico(mask: np.ndarray, k_apertura: int = 5, k_cierre: int = 15) -> np.ndarray:
    """Apertura (erosión+dilatación) elimina puntos aislados;
    cierre (dilatación+erosión) rellena huecos dentro de la silueta."""
    if k_apertura:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(k_apertura))
    if k_cierre:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(k_cierre))
    return mask


def componente_principal(mask: np.ndarray, area_min_frac: float = 0.005) -> np.ndarray:
    """Se queda con la componente conexa más grande (se asume un único sujeto)."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] < area_min_frac * mask.size:
        return np.zeros_like(mask)
    return np.where(etiquetas == k, 255, 0).astype(np.uint8)


def rellenar_huecos(mask: np.ndarray) -> np.ndarray:
    """Rellena huecos internos de la silueta (p. ej. una polera clara sobre fondo claro)."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contornos, -1, 255, thickness=cv2.FILLED)
    return out


def obtener_roi_sujeto(mask: np.ndarray, margen: float = 0.0):
    """Bounding box (x, y, w, h) de la máscara, con margen relativo opcional. None si vacía."""
    pts = cv2.findNonZero(mask)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if margen:
        H, W = mask.shape[:2]
        dx, dy = int(w * margen), int(h * margen)
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
    return (x, y, w, h)


def metricas_segmentacion(mask_cruda: np.ndarray, mask_final: np.ndarray, bbox=None) -> dict:
    """Métricas que muestra la GUI de segmentación (R_SegmentacionActualizada)."""
    total = mask_final.size
    area = int(np.count_nonzero(mask_final))
    # ruido = píxeles marcados como primer plano en la binarización que NO son el sujeto
    ruido = int(np.count_nonzero((mask_cruda > 0) & (mask_final == 0)))
    m = {
        "area_sujeto_pct": 100.0 * area / total,
        "ruido_fondo_px": ruido,
        "ruido_fondo_pct": 100.0 * ruido / total,
        "roi_detectada": bbox is not None,
    }
    if bbox is not None:
        H, W = mask_final.shape[:2]
        x, y, w, h = bbox
        # útil para el protocolo: ¿el cuerpo está completo o cortado por el borde?
        m["toca_borde"] = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        m["alto_roi_pct"] = 100.0 * h / H
    return m


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union entre dos máscaras (para evaluar métodos de segmentación)."""
    a, b = mask_a > 0, mask_b > 0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


# ---------------------------------------------------------------------------
# Función principal (misma firma que usa el notebook del grupo)
# ---------------------------------------------------------------------------
def segmentar_sujeto(frame_rgb: np.ndarray, umbral: int | None = None, usar_morfologia: bool = True,
                     invertir: bool = True, clahe: bool = False, margen_roi: float = 0.0,
                     devolver_metricas: bool = False):
    """Segmentación por umbral global.
    umbral: 0-255, o None para Otsu automático.
    Devuelve (mascara, bbox) o (mascara, bbox, metricas) si devolver_metricas=True."""
    gris = convertir_espacio_color(frame_rgb, clahe=clahe)
    cruda, t = aplicar_binarizacion(gris, umbral, invertir)
    mask = filtrar_ruido_morfologico(cruda) if usar_morfologia else cruda.copy()
    mask = componente_principal(mask)
    if usar_morfologia:
        mask = rellenar_huecos(mask)
    bbox = obtener_roi_sujeto(mask, margen_roi)
    if devolver_metricas:
        m = metricas_segmentacion(cruda, mask, bbox)
        m["umbral_usado"] = t
        return mask, bbox, m
    return mask, bbox


# ---------------------------------------------------------------------------
# Segmentación por modelo de fondo (recomendada para videos reales)
# ---------------------------------------------------------------------------
def modelo_fondo_mediana(cuadros_rgb: list[np.ndarray]) -> np.ndarray:
    """Fondo = mediana temporal de varios cuadros. Ideal: 2-3 s grabados SIN la persona
    al inicio del video (propuesta de agregarlo al protocolo de captura)."""
    return np.median(np.stack(cuadros_rgb, axis=0), axis=0).astype(np.uint8)


class SegmentadorFondo:
    """Sustracción de fondo con supresión de sombras.

    metodo='diferencia': compara contra un fondo fijo (mediana de cuadros vacíos). Más estable.
    metodo='mog2': fondo adaptativo de OpenCV (no necesita cuadros vacíos, pero si la persona
                   se queda quieta mucho rato termina 'absorbida' por el fondo).
    """

    def __init__(self, fondo_rgb: np.ndarray | None = None, metodo: str = "diferencia",
                 umbral: int = 25, k_apertura: int = 5, k_cierre: int = 21, suavizado_roi: float = 0.6):
        self.metodo = metodo
        self.umbral = umbral
        self.k_apertura, self.k_cierre = k_apertura, k_cierre
        self.alpha = suavizado_roi          # EMA sobre la ROI para que no "tiemble" entre cuadros
        self._roi_prev = None
        self.fondo_congelado = False
        if metodo == "diferencia":
            if fondo_rgb is None:
                raise ValueError("metodo='diferencia' necesita fondo_rgb (use modelo_fondo_mediana)")
            self.fondo_lab = cv2.cvtColor(cv2.GaussianBlur(fondo_rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.int16)
        elif metodo == "mog2":
            self.mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=umbral, detectShadows=True)
        else:
            raise ValueError("metodo debe ser 'diferencia' o 'mog2'")

    def aprender_fondo(self, cuadros_rgb, congelar: bool = True):
        """(MOG2) Entrena el modelo con cuadros sin persona y, opcionalmente, lo congela."""
        for f in cuadros_rgb:
            self.mog.apply(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        self.fondo_congelado = congelar

    def _mascara_cruda(self, frame_rgb):
        if self.metodo == "mog2":
            # learningRate=0 tras aprender_fondo(): el fondo queda congelado y la persona,
            # que en una sentadilla casi no se desplaza, no termina absorbida por el modelo.
            lr = 0 if self.fondo_congelado else -1
            m = self.mog.apply(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), learningRate=lr)
            return np.where(m == 255, 255, 0).astype(np.uint8)   # 127 = sombra -> fondo
        lab = cv2.cvtColor(cv2.GaussianBlur(frame_rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.int16)
        dL = lab[..., 0] - self.fondo_lab[..., 0]
        dab = np.abs(lab[..., 1] - self.fondo_lab[..., 1]) + np.abs(lab[..., 2] - self.fondo_lab[..., 2])
        dist = np.abs(dL) + dab
        # Sombra: más oscuro que el fondo (ratio de luminancia 0.45-0.92) pero mismo color
        ratio = (lab[..., 0] + 1) / (self.fondo_lab[..., 0] + 1)
        sombra = (ratio > 0.45) & (ratio < 0.92) & (dab < 12)
        return np.where((dist > self.umbral) & ~sombra, 255, 0).astype(np.uint8)

    def _suavizar_roi(self, bbox):
        if bbox is None or self._roi_prev is None or not self.alpha:
            self._roi_prev = bbox if bbox is not None else self._roi_prev
            return bbox
        b = tuple(int(round(self.alpha * p + (1 - self.alpha) * c)) for p, c in zip(self._roi_prev, bbox))
        self._roi_prev = b
        return b

    def segmentar(self, frame_rgb: np.ndarray, margen_roi: float = 0.05):
        cruda = self._mascara_cruda(frame_rgb)
        mask = filtrar_ruido_morfologico(cruda, self.k_apertura, self.k_cierre)
        mask = rellenar_huecos(componente_principal(mask))
        bbox = self._suavizar_roi(obtener_roi_sujeto(mask, margen_roi))
        return mask, bbox, metricas_segmentacion(cruda, mask, bbox)


# ---------------------------------------------------------------------------
# Utilidades de visualización
# ---------------------------------------------------------------------------
def dibujar_resultado(frame_rgb, mask, bbox, color=(0, 220, 200), alpha=0.35):
    """Superpone la máscara y la ROI sobre el cuadro (para GUI / informes)."""
    out = frame_rgb.copy()
    capa = np.zeros_like(out)
    capa[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, capa, alpha, 0)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 60, 60), 3)
    return out
