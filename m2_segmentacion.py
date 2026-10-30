import cv2
import numpy as np
from typing import Tuple, Optional

def segmentar_sujeto(frame_rgb: np.ndarray, umbral: int = 127) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Recibe un cuadro RGB, lo convierte a escala de grises y aplica binarización
    junto con filtros morfológicos para aislar la silueta del sujeto.
    Retorna la máscara binaria limpia y las coordenadas de la Región de Interés (ROI).
    """
    # 1. Conversión de espacio de color (RGB a Escala de Grises)
    # Requisito fundamental para que cv2.threshold pueda operar sobre un solo canal.
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Binarización con umbral ajustable
    # Nota: Se utiliza THRESH_BINARY_INV asumiendo que la persona genera contraste oscuro 
    # frente a un fondo claro. Si el fondo es oscuro, cambiar a cv2.THRESH_BINARY.
    _, mascara_binaria = cv2.threshold(frame_gray, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Filtro morfológico (Apertura)
    # Define un kernel estructurante de 5x5 píxeles. La operación MORPH_OPEN elimina 
    # artefactos pequeños y ruido aislado del fondo sin alterar el tamaño de la silueta.
    kernel_apertura = np.ones((5, 5), np.uint8)
    mascara_limpia = cv2.morphologyEx(mascara_binaria, cv2.MORPH_OPEN, kernel_apertura)
    
    # Opcional (recomendado por diseño M2): Cierre morfológico para rellenar pequeños huecos dentro de la silueta
    # mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel_apertura)
    
    # 4. Extracción de las coordenadas delimitadoras (Bounding Box / ROI)
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = None
    if contornos:
        # Se asume que el contorno continuo con mayor área corresponde al cuerpo del paciente
        contorno_paciente = max(contornos, key=cv2.contourArea)
        
        # Filtro de seguridad: ignorar ruido que haya sobrevivido y que sea demasiado pequeño
        if cv2.contourArea(contorno_paciente) > 500: 
            # cv2.boundingRect retorna una tupla: (x, y, ancho, alto)
            bbox = cv2.boundingRect(contorno_paciente)
            
    return mascara_limpia, bbox

# ==========================================
# Ejemplo de acoplamiento con Módulo 1:
# ==========================================
if __name__ == "__main__":
    # Supongamos que importas tu generador del archivo M1
    # from m1_adquisicion import cargar_video
    
    # Simulamos la obtención de un cuadro RGB
    # cuadros, fps, res = cargar_video("sentadilla_prueba.mp4")
    # frame_actual = next(cuadros) 
    
    # Creación de un cuadro ficticio para probar el script sin video
    frame_simulado = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    cv2.circle(frame_simulado, (640, 360), 150, (50, 50, 50), -1) # Sujeto simulado
    
    # Ejecución del Módulo 2
    valor_umbral = 100 
    mascara_resultante, roi_bbox = segmentar_sujeto(frame_simulado, valor_umbral)
    
    if roi_bbox:
        x, y, w, h = roi_bbox
        print(f"Sujeto aislado con éxito. ROI detectada en: X={x}, Y={y}, Ancho={w}, Alto={h}")
    else:
        print("No se detectó un sujeto válido en el cuadro.")