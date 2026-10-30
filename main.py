import cv2
import mediapipe as mp
from m1_adquisicion import cargar_video
from m2_segmentacion import segmentar_sujeto
from m3_estimacion_pose import estimar_pose_y_angulo, mp_pose, mp_dibujo
from m4_tracking import TrackerArticular
from m5_comparador import comparar_ejecuciones

def procesar_sesion(ruta_video: str, umbral_seg: int = 100):
    print(f"\n--- Procesando Sesión: {ruta_video} ---")
    cuadros_rgb, fps, res = cargar_video(ruta_video)
    
    tracker = TrackerArticular()
    estimador = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    for frame_rgb in cuadros_rgb:
        # 1. Segmentación PDI
        mascara, roi = segmentar_sujeto(frame_rgb, umbral_seg)
        
        # 2. Estimación de Pose y Ángulo
        landmarks, angulo = estimar_pose_y_angulo(frame_rgb, estimador)
        
        # 3. Seguimiento Temporal
        if angulo is not None:
            tracker.registrar_angulo(angulo)
        
        # 4. GUI Visual básica interactiva
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if landmarks:
            mp_dibujo.draw_landmarks(frame_bgr, landmarks, mp_pose.POSE_CONNECTIONS)
            if angulo is not None:
                cv2.putText(frame_bgr, f"Angulo Rodilla: {angulo:.1f}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Opcional: mostrar la caja delimitadora (ROI) del módulo 2
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        cv2.imshow('KinePose - Visor', frame_bgr)
        
        # Salida anticipada con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Procesamiento interrumpido por el usuario.")
            break
            
    cv2.destroyAllWindows()
    estimador.close()
    
    serie_final = tracker.obtener_serie()
    print(f"Sesión completada. Cuadros procesados exitosamente: {len(serie_final)}")
    return serie_final

if __name__ == "__main__":
    print("Iniciando Plataforma KinePose...")
    
    # Para ejecutar con tus videos reales, asegúrate de colocar tus archivos
    # en la misma carpeta que este script y descomenta las siguientes líneas:
    
    ruta_sesion1 = "sentadilla_sesion1.mp4"
    ruta_sesion2 = "sentadilla_sesion2.mp4"
    
    try:
        serie_s1 = procesar_sesion(ruta_sesion1)
        serie_s2 = procesar_sesion(ruta_sesion2)
        
        comparar_ejecuciones(serie_s1, serie_s2)
    except Exception as e:
        print(f"Error durante la ejecución del pipeline: {e}")
    
    print("Pipeline ensamblado y listo para ejecutarse.")
