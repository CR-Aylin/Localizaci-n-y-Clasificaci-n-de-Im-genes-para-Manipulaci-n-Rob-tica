import numpy as np
from NaiveBayes import NaiveBayes


# Localización de Ventana Deslizante con Naive Bayes (3 clases: Circulo, Poligonos, Fondo)
def sliding_window_localization_bayes(board_image, model, extractor,
                                       window_size=(100, 100), step=20,
                                       confidence_threshold=0.6,
                                       clase_fondo="Fondo"):
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
            if features.shape[0] != model.mean[model.classes[0]].shape[0]:
                continue

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

#Posible uso
if __name__ == "__main__":
    # Ejemplo de uso:
    #
    # from extractor_seguro_bayes import SafeFeaturesExtractor
    # from entrenar_naive_bayes import entrenar_y_evaluar
    # import cv2
    #
    # modelo, _, _ = entrenar_y_evaluar(ruta_dataset="dataset_imagenes")
    # extractor = SafeFeaturesExtractor()  # mismo extractor usado en el entrenamiento
    # tablero = cv2.imread("tablero.jpg")
    #
    # detecciones = sliding_window_localization_bayes(
    #     tablero, modelo, extractor,
    #     window_size=(100, 100), step=20, confidence_threshold=0.6
    # )
    # for d in detecciones:
    #     print(d)


    
    pass
