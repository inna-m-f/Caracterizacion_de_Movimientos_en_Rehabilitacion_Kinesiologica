import cv2
from typing import Tuple, Generator, Any

def cargar_video(filepath: str) -> Tuple[Generator[Any, None, None], int, Tuple[int, int]]:
    """
    Carga un archivo de video local, valida su apertura y extrae metadatos.
    Retorna un generador de cuadros en formato RGB, los FPS y la resolución.
    """
    cap = cv2.VideoCapture(filepath)
    
    # Validación de apertura del archivo
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir o inicializar el video en la ruta: {filepath}")
    
    # Extracción de metadatos de la secuencia
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolucion = (ancho, alto)
    
    # Definición del generador para la extracción secuencial
    def generador_cuadros():
        while cap.isOpened():
            ret, frame = cap.read()
            
            # 'ret' es False cuando se alcanza el final del video o hay un corte
            if not ret:
                break
                
            # OpenCV decodifica los cuadros nativamente en BGR. 
            # Es obligatorio convertir a RGB para la compatibilidad posterior con MediaPipe.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb
            
        # Liberación estricta del recurso multimedia al terminar el ciclo
        cap.release()
        
    return generador_cuadros(), fps, resolucion

# ==========================================
# Ejemplo de inicialización y acoplamiento:
# ==========================================
if __name__ == "__main__":
    ruta_video = "sentadilla_prueba.mp4"
    
    try:
        cuadros_rgb, fps_video, res_video = cargar_video(ruta_video)
        print(f"Video cargado exitosamente. Resolución: {res_video}, FPS: {fps_video}")
        
        # Iteración controlada para el flujo hacia el Módulo 2
        for frame in cuadros_rgb:
            # Aquí se inyectará el frame matricial hacia cv2.cvtColor (Escala de Grises/HSV) 
            # y cv2.threshold para aislar la silueta (Módulo 2).
            pass
            
    except IOError as e:
        print(f"Error de Adquisición: {e}")