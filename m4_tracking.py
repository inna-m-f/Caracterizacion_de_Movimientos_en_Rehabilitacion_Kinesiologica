import numpy as np
from typing import List

# ==========================================
# Módulo 4: Seguimiento Temporal (Tracking)
# ==========================================
class TrackerArticular:
    """
    Acumula secuencialmente los ángulos calculados en el Módulo 3 
    para construir la serie de tiempo completa de una ejecución.
    """
    def __init__(self):
        self.serie_angular = []

    def registrar_angulo(self, angulo: float):
        """Almacena el ángulo validando que no sea nulo."""
        if angulo is not None:
            self.serie_angular.append(angulo)

    def obtener_serie(self) -> List[float]:
        """Retorna la serie temporal completa de la sesión."""
        return self.serie_angular

# ==========================================
# Ejemplo de inicialización independiente:
# ==========================================
if __name__ == "__main__":
    # Simulación Módulo 4: Grabación de una sesión
    tracker_sesion1 = TrackerArticular()

    # Simulamos una sentadilla de 45 cuadros llegando a 110° de flexión
    for angulo in np.sin(np.linspace(0, np.pi, 45)) * 110:
        tracker_sesion1.registrar_angulo(angulo)

    datos_s1 = tracker_sesion1.obtener_serie()
    print(f"Módulo 4 funcionando. Se rastrearon {len(datos_s1)} cuadros.")