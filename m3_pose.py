import cv2
import numpy as np
from typing import Tuple, Any, Optional

# Importación directa y segura para versiones modernas de MediaPipe
import mediapipe as mp

# Inicialización segura utilizando los submódulos oficiales
mp_pose = mp.solutions.pose if hasattr(mp, 'solutions') else None
mp_dibujo = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None

# Si la versión de mediapipe viene empaquetada sin 'solutions' accesible por atributo directo:
if mp_pose is None:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_dibujo

def calcular_angulo(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calcula el ángulo en el vértice 'b' formado por los segmentos de recta ab y bc.
    Utiliza trigonometría con NumPy (arctan2) para manejar correctamente los cuadrantes.
    """
    a = np.array(a) # Nodo proximal (Ej. Cadera)
    b = np.array(b) # Vértice del ángulo (Ej. Rodilla)
    c = np.array(c) # Nodo distal (Ej. Tobillo)
    
    # Cálculo del ángulo en radianes y conversión a grados
    radianes = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = np.abs(radianes * 180.0 / np.pi)
    
    # Normalización para mantener el ángulo de flexión anatómica entre 0° y 180°
    if angulo > 180.0:
        angulo = 360.0 - angulo
        
    return angulo

def estimar_pose_y_angulo(frame_rgb: np.ndarray, estimador_pose: Any) -> Tuple[Any, Optional[float]]:
    """
    Infiere los 33 landmarks anatómicos sobre el cuadro RGB y calcula el ángulo 
    de flexión de la rodilla en el instante actual.
    Retorna los landmarks crudos (para renderizado posterior) y el valor angular.
    """
    # 1. Inferencia del esqueleto digital
    # El frame ya debe venir en RGB desde el Módulo 1 o 2.
    resultados = estimador_pose.process(frame_rgb)
    
    angulo_rodilla = None
    
    # 2. Extracción de nodos anatómicos clave (si se detectó un sujeto)
    if resultados.pose_landmarks:
        landmarks = resultados.pose_landmarks.landmark
        
        # Mapeo de índices MediaPipe (se utiliza el perfil derecho por defecto para plano sagital)
        # 24: RIGHT_HIP (Cadera) | 26: RIGHT_KNEE (Rodilla) | 28: RIGHT_ANKLE (Tobillo)
        idx_cadera = mp_pose.PoseLandmark.RIGHT_HIP.value
        idx_rodilla = mp_pose.PoseLandmark.RIGHT_KNEE.value
        idx_tobillo = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        
        # Extracción de coordenadas 2D normalizadas (X, Y)
        cadera = (landmarks[idx_cadera].x, landmarks[idx_cadera].y)
        rodilla = (landmarks[idx_rodilla].x, landmarks[idx_rodilla].y)
        tobillo = (landmarks[idx_tobillo].x, landmarks[idx_tobillo].y)
        
        # 3. Cálculo trigonométrico
        angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
        
    return resultados.pose_landmarks, angulo_rodilla

# ==========================================
# Ejemplo de inicialización y acoplamiento:
# ==========================================
if __name__ == "__main__":
    # Configuración del estimador Pose (Model Complexity 1 es el balance ideal para CPU)
    estimador = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    # Cuadro RGB simulado para prueba independiente
    frame_simulado = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Ejecución de Módulo 3
    nodos_detectados, valor_angulo = estimar_pose_y_angulo(frame_simulado, estimador)
    
    if nodos_detectados and valor_angulo is not None:
        print(f"Esqueleto detectado. Ángulo de flexión de rodilla: {valor_angulo:.2f}°")
    else:
        print("Esperando detección de sujeto...")
        
    # Liberar recursos del estimador al finalizar todo el procesamiento
    estimador.close()