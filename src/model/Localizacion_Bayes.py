import numpy as np
import cv2
from src.model.NaiveBayes import NaiveBayes
from src.model.Entrenar_Naive_Bayes import cargar_y_describir_dataset, entrenar_modelo_final


# Localización de Ventana Deslizante con Naive Bayes (3 clases: Circulo, Poligonos, Fondo)
def sliding_window_localization_bayes(board_image, model, extractor,
                                       window_size=(100, 100), step=20,
                                       confidence_threshold=0.6,
                                       clase_fondo="Fondo",
                                       norm_stats=None, pca=None):
    """

    Se recorre usando el metodo de ventana deslizante para localizar y clasificar los elementos usando NaiveBayes
        board_image: imagen completa del tablero (numpy array, BGR).
    model: instancia de NaiveBayes ya entrenada con 3 clases.
    extractor: instancia de features_extractor (Extraccion_caracteristicas.py).
    window_size: tamaño fijo de la ventana deslizante (ancho, alto).
    step: desplazamiento en píxeles entre ventanas.
    confidence_threshold: probabilidad mínima para aceptar una detección de
        objeto (no aplica a Fondo, que siempre se descarta).
    clase_fondo: nombre de la clase que representa "no hay objeto".
    
    """
    h, w = board_image.shape[:2]
    detections = []

    for y in range(0, h - window_size[1] + 1, step):
        for x in range(0, w - window_size[0] + 1, step):
            window = board_image[y:y + window_size[1], x:x + window_size[0]]

            # Ventana literalmente vacía: se salta sin llamar al extractor/modelo
            if np.sum(window) == 0:
                continue


            """
            Se extraen caracteristicas en la ventana. Puede fallar o devolver un vector de tamaño distinto
            al esperado por el modelo en caso de que no se encuentre en alguna variable contorno y devulve None, descartando la ventana sin frenar el barrido
            """
            try:
                features = extractor.extract(window)
            except Exception:
                continue

            if features is None:
                continue

            features = np.asarray(features, dtype=float)
            if features.shape[0] != model.mean[model.classes[0]].shape[0] and pca is None:
                continue

            if norm_stats is not None:
                features = (features - norm_stats["medias"]) / norm_stats["stds"]
            if pca is not None:
                features = pca.transform(features.reshape(1, -1))[0]

            # Predecir clase y confianza (probabilidad posterior)
            label, confidence = model.predict_with_confidence(features)

            # Descartar Fondo explícitamente, y exigir confianza mínima
            if label == clase_fondo:
                continue
            if confidence < confidence_threshold:
                continue

            detections.append({
                'x': x + window_size[0] // 2,
                'y': y + window_size[1] // 2,
                'class': label,
                'confidence': confidence
            })

    # Eliminar detecciones duplicadas, conservando la de mayor confianza
    final_detections = []
    for det in detections:
        is_duplicate = False
        for final_det in final_detections:
            if det['class'] == final_det['class'] and \
               abs(det['x'] - final_det['x']) < window_size[0] // 2 and \
               abs(det['y'] - final_det['y']) < window_size[1] // 2:
                if det['confidence'] > final_det['confidence']:
                    final_det.update(det)
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append(det)

    return final_detections


def ejecutar_deteccion(imagen, modelo, extractor, norm_stats=None, pca=None,
                        window_size=(100, 100), step=20,
                        confidence_threshold=0.6, clase_fondo="Fondo"):
    """
    Ejecuta la localización por ventana deslizante sobre una imagen y
    devuelve, para cada objeto detectado, solo lo esencial: su clase y su
    posición (x, y) en la imagen.

    imagen: numpy array BGR (o ruta a un archivo, se acepta por comodidad).
    modelo: instancia de NaiveBayes ya entrenada.
    extractor: mismo extractor usado en el entrenamiento.
    norm_stats: dict {"medias": ..., "stds": ...} usado al entrenar
        (obligatorio si el modelo se entrenó con datos normalizados).
    pca: instancia de PCA ya ajustada, o None si no se usó PCA.

    Devuelve una lista de tuplas (clase, (x, y)), ordenada de mayor a
    menor confianza. Si no se detecta nada, devuelve una lista vacía.
    """
    if isinstance(imagen, str):
        import cv2
        imagen_cargada = cv2.imread(imagen)
        if imagen_cargada is None:
            print("  [ejecutar_deteccion] no se pudo leer la imagen.")
            return []
        imagen = imagen_cargada

    detecciones = sliding_window_localization_bayes(
        imagen, modelo, extractor,
        window_size=window_size, step=step,
        confidence_threshold=confidence_threshold,
        clase_fondo=clase_fondo,
        norm_stats=norm_stats, pca=pca
    )

    # Ordenar de mayor a menor confianza para que el resultado sea consistente
    detecciones_ordenadas = sorted(detecciones, key=lambda d: d['confidence'], reverse=True)

    resultado = [(d['class'], (d['x'], d['y'])) for d in detecciones_ordenadas]

    if not resultado:
        print("No se detectaron objetos.")
    else:
        for clase, (x, y) in resultado:
            print(f"Detectado: {clase} en posición ({x}, {y})")

    return resultado

#Posible uso
if __name__ == "__main__":
    ruta_dataset = r"dataset\Entrenamiento"
    ruta_imagen_tablero = r"dataset\prueba1\ala.jpg"

    X, y, extractor = cargar_y_describir_dataset(ruta_dataset)
    modelo, extractor, norm_stats, pca = entrenar_modelo_final(X, y, extractor, usar_pca=False)

    resultado = ejecutar_deteccion(
        ruta_imagen_tablero, modelo, extractor,
        norm_stats=norm_stats, pca=pca,
        window_size=(100, 100), step=20, confidence_threshold=0.6
    )
    # resultado -> [("Circulo", (140, 260)), ("Poligonos", (320, 180)), ...]
