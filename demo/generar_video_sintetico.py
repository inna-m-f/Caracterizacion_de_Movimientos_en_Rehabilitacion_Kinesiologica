"""
Genera un video de prueba con "verdad de terreno" (máscara real del sujeto) para
evaluar M2 de forma objetiva (IoU) antes de tener videos reales.

Escena "tipo casa": pared con gradiente de luz, piso, un mueble oscuro y una ventana
clara (que engañan al umbral global), sombra del sujeto en la pared y ruido de sensor.
Los primeros 2 s el encuadre está vacío (fondo), luego el sujeto hace 3 sentadillas.
"""
import sys
import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
SEG_FONDO, N_SENTADILLAS, SEG_SENTADILLA = 2.0, 3, 2.5


def fondo():
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    pared = 205 - 55 * x - 20 * y                       # luz lateral desde la izquierda
    img = np.repeat(pared[..., None], 3, axis=2) * np.array([1.0, 0.97, 0.92])
    img[int(H * 0.82):] = np.array([120, 105, 90])       # piso de madera
    img[80:420, 60:300] = np.array([235, 240, 245])      # ventana clara
    img[330:int(H * 0.82), 900:1180] = np.array([55, 45, 40])   # mueble oscuro
    return np.clip(img, 0, 255).astype(np.uint8)


def dibujar_sujeto(canvas, mask, fase, cx=620, dx_sombra=0):
    """Figura de perfil (plano sagital). fase 0 = de pie, 1 = sentadilla profunda."""
    suelo = int(H * 0.86)
    musl, tib, tronco = 175, 175, 230
    rodilla_ang = np.deg2rad(0 + 100 * fase)            # flexión de rodilla
    cadera_ang = np.deg2rad(0 + 85 * fase)
    tobillo = np.array([cx, suelo])
    rodilla = tobillo + tib * np.array([np.sin(rodilla_ang * 0.45), -np.cos(rodilla_ang * 0.45)])
    cadera = rodilla + musl * np.array([-np.sin(rodilla_ang * 0.55 + 0.0), -np.cos(rodilla_ang * 0.55)])
    hombro = cadera + tronco * np.array([np.sin(cadera_ang * 0.45), -np.cos(cadera_ang * 0.45)])
    cabeza = hombro + np.array([12, -62])
    mano = hombro + np.array([150 * fase + 40 * (1 - fase), 40 + 120 * (1 - fase)])
    pie = tobillo + np.array([55, 0])
    segs = [(tobillo, rodilla, 34), (rodilla, cadera, 44), (cadera, hombro, 58),
            (hombro, mano, 26), (tobillo, pie, 22)]
    for c, m, color in ((canvas, mask, None),):
        for a, b, g in segs:
            pa, pb = tuple(map(int, a + [dx_sombra, 0])), tuple(map(int, b + [dx_sombra, 0]))
            if color is None:
                cv2.line(c, pa, pb, (35, 38, 48), g, cv2.LINE_AA)
                cv2.line(m, pa, pb, 255, g)
        cv2.circle(c, tuple(map(int, cabeza)), 48, (60, 45, 40), -1, cv2.LINE_AA)
        cv2.circle(m, tuple(map(int, cabeza)), 48, 255, -1)
    # zapatillas blancas (como en el video de prueba): parte del sujeto, difícil para el umbral
    cv2.line(canvas, tuple(map(int, tobillo + [0, 6])), tuple(map(int, pie + [8, 6])), (240, 240, 240), 18)
    cv2.line(mask, tuple(map(int, tobillo + [0, 6])), tuple(map(int, pie + [8, 6])), 255, 18)


def generar(ruta_video, ruta_gt):
    rng = np.random.default_rng(0)
    base = fondo()
    vw = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gts = []
    n_fondo = int(SEG_FONDO * FPS)
    n_total = n_fondo + int(N_SENTADILLAS * SEG_SENTADILLA * FPS)
    for i in range(n_total):
        img = base.copy()
        gt = np.zeros((H, W), np.uint8)
        if i >= n_fondo:
            t = (i - n_fondo) / FPS
            fase = 0.5 - 0.5 * np.cos(2 * np.pi * t / SEG_SENTADILLA)
            # sombra en la pared: más oscura, desplazada, desenfocada
            sombra = np.zeros((H, W), np.uint8)
            dibujar_sujeto(np.zeros_like(img), sombra, fase, dx_sombra=70)
            sombra = cv2.GaussianBlur(sombra, (41, 41), 0).astype(np.float32) / 255 * 0.35
            img = (img * (1 - sombra[..., None])).astype(np.uint8)
            dibujar_sujeto(img, gt, fase)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        gts.append(gt)
    vw.release()
    np.savez_compressed(ruta_gt, gt=np.stack(gts), n_fondo=n_fondo)
    print(f"{ruta_video}: {n_total} cuadros ({n_fondo} de fondo vacío)")


if __name__ == "__main__":
    generar(sys.argv[1] if len(sys.argv) > 1 else "sentadilla_sintetica.mp4",
            sys.argv[2] if len(sys.argv) > 2 else "sentadilla_sintetica_gt.npz")
