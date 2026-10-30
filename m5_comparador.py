import numpy as np
from typing import List

def comparar_ejecuciones(serie_referencia: List[float], serie_actual: List[float]) -> None:
    """
    Recibe las series de tiempo de dos sesiones, las normaliza a una longitud
    del 0% al 100% del ciclo de movimiento y compara sus valores máximos.
    Activa una alerta si la desviación supera el 10%.
    """
    if len(serie_referencia) < 2 or len(serie_actual) < 2:
        print("Error: Se requieren al menos dos cuadros por sesión para comparar.")
        return

    # Normalización del ciclo de movimiento (0% a 100%)
    eje_x_ref = np.linspace(0, 100, num=len(serie_referencia))
    eje_x_act = np.linspace(0, 100, num=len(serie_actual))
    
    porcentaje_ciclo = np.linspace(0, 100, num=100)

    curva_ref_norm = np.interp(porcentaje_ciclo, eje_x_ref, serie_referencia)
    curva_act_norm = np.interp(porcentaje_ciclo, eje_x_act, serie_actual)

    # Extracción de los valores pico
    max_ref = np.max(curva_ref_norm)
    max_act = np.max(curva_act_norm)

    # Cálculo de la diferencia porcentual relativa
    desviacion_absoluta = abs(max_act - max_ref)
    diferencia_porcentual = (desviacion_absoluta / max_ref) * 100

    # Evaluación
    print("-" * 50)
    print("REPORTE COMPARATIVO DE SESIONES")
    print("-" * 50)
    print(f"Pico Sesión Referencia: {max_ref:.1f}°")
    print(f"Pico Sesión Actual:     {max_act:.1f}°")
    print(f"Desviación Detectada:   {diferencia_porcentual:.1f}%\n")

    if diferencia_porcentual > 10.0:
        print("[ALERTA CRÍTICA] La desviación supera el umbral del 10%.")
        print("Posible compensación corporal o déficit en el rango de movimiento.")
    else:
        print("[OK] Ejecución estable. La diferencia está dentro del margen permitido.")
    print("-" * 50)
