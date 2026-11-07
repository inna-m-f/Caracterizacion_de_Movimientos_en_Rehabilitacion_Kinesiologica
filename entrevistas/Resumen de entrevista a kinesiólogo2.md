# Resumen de Entrevista a kinesiólogo 

## 1. Conceptos Biomecánicos y Captura de Video
* **Planos Anatómicos:** La posición de la cámara debe ser ortogonal al plano del movimiento. El **plano sagital** (perfil) es el más sencillo y utilizado para medir flexo-extensión (ej. rodilla, cadera). Para abducción de hombro se utiliza una cámara de frente.
* **Construcción del Ángulo:** Los ángulos articulares son relativos. El ángulo de la rodilla se calcula mediante la posición de tres puntos anatómicos: cadera, rodilla y tobillo.

## 2. Algoritmos y Procesamiento de Datos
* **Estimación de Pose (Markerless):** Se recomienda el uso de **MediaPipe**. Estos modelos pre-entrenados reconocen la persona y estiman las coordenadas de las articulaciones sin necesidad de marcadores físicos.
* **Análisis Cinemático:** 
  * **Amplitud (ROM):** Medir los grados de flexión alcanzados (ej. 60° vs 100° en sentadilla).
  * **Velocidad Angular (Derivada):** Fundamental en evaluaciones clínicas. Por ejemplo, en el test *Sit-to-Stand*, demorar más de 10-20 segundos indica un riesgo de mortalidad 70% mayor debido a déficit de fuerza.

## 3. Especificaciones de Hardware
* **Frecuencia de Muestreo (FPS):**
  * **30 FPS:** Marcha normal y ejercicios de baja velocidad (sentadillas, elevación de brazos).
  * **60 FPS:** Trote o saltos repetidos.
  * **120 FPS:** Carrera a alta velocidad (sobre 10 km/h en trotadora).
* **Lente y Encuadre:** Capturar el cuerpo completo (aprox. a 2.5 metros de distancia). **Evitar lentes gran angular** porque deforman los bordes del cuadro y alteran la precisión geométrica. La resolución Full HD de un celular actual es suficiente.

## 4. Control del Entorno de Grabación
* **Vestimenta:** Ropa deportiva ajustada (shorts y manga corta) para no ocluir la silueta articular.
* **Iluminación:** Luz blanca directa. Evitar contraluz.
* **Fondo:** Muros sólidos, sin espejos y sin personas adicionales transitando por detrás para no confundir al algoritmo.

## 5. Ética y Consideraciones de Desarrollo
* **Protección de Datos:** **Bajo ninguna circunstancia** utilizar videos de pacientes reales provenientes de clínicas (implica infracciones legales graves de privacidad).
* **Validación MVP:** Seleccionar 4 o 5 ejercicios de peso corporal (sentadilla normal, sentadilla a un pie, abducción de hombro). Grabar las pruebas con compañeros de equipo y utilizar esos datos como demostración para validar la utilidad clínica con kinesiólogos.