from collections import Counter

import cv2
import numpy as np
import os
import time
from SVM import SVM, SVM_PCA, sliding_window_localization_svm
from Extraccion_caracteristicas import features_extractor


def cargar_dataset(ruta_dataset):
    X = []
    y = []

    extractor = features_extractor(hist_bins=16, debug=False)

    if not os.path.isdir(ruta_dataset):
        raise NotADirectoryError(f"La ruta especificada no es un directorio válido: {ruta_dataset}")

    for clase in sorted(os.listdir(ruta_dataset)):
        ruta_clase = os.path.join(ruta_dataset, clase)

        # Ignora la carpeta de pruebas o archivos sueltos para no contaminar el entrenamiento
        if not os.path.isdir(ruta_clase) or clase == "prueba1":
            continue

        print(f"Cargando clase: {clase}")

        for archivo in sorted(os.listdir(ruta_clase)):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_imagen = os.path.join(ruta_clase, archivo)

                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"  ⚠ No se pudo cargar: {archivo}")
                    continue

                features = extractor.extract(imagen)

                # Solo guardar muestras válidas
                if np.linalg.norm(features) > 1e-6:
                    X.append(features.astype(np.float32))
                    y.append(clase)

        # Se imprime una sola vez al terminar cada clase (no por cada imagen)
        print("Cantidad de muestras por clase hasta ahora:")
        print(Counter(y))

    return np.array(X), np.array(y)


def entrenar_svm(ruta_dataset, usar_pca=False, n_components=50):
    print("CARGANDO DATASET DE ENTRENAMIENTO")
    X, y = cargar_dataset(ruta_dataset)

    if len(X) == 0:
        raise ValueError("No se encontraron muestras válidas para entrenar la SVM.")

    print(f"\n Dataset cargado: {len(X)} muestras")
    print(f" Clases encontradas: {np.unique(y)}")

    if usar_pca:
        print(f"\nEntrenando SVM con PCA (n_components={n_components})...")
        model = SVM_PCA(n_components=n_components, kernel='rbf', C=10.0, gamma='scale')
    else:
        print("\nEntrenando SVM estándar...")
        model = SVM(kernel='rbf', C=10.0, gamma='scale')

    inicio = time.time()
    model.fit(X, y)
    fin = time.time()

    print(f"Modelo entrenado exitosamente")
    print(f"Tiempo de entrenamiento: {fin - inicio:.4f} segundos")

    return model, X, y


def probar_svm_en_imagen(model, ruta_imagen, extractor, confidence_threshold=0.5):
    print(f"PROBANDO SVM EN IMAGEN: {ruta_imagen}")

    board_image = cv2.imread(ruta_imagen)
    if board_image is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    print(f"Imagen cargada: {board_image.shape}")
    print("\nAplicando ventana deslizante...")

    inicio = time.time()
    detections = sliding_window_localization_svm(
        board_image,
        model=model,
        extractor=extractor,
        window_size=(100, 100),
        step=20,
        confidence_threshold=confidence_threshold
    )
    fin = time.time()

    print(f"\nObjetos detectados inicialmente por SVM: {len(detections)}")
    print(f"Tiempo de inferencia: {fin - inicio:.4f} segundos")

    # ===== IMPRIMIR DETECCIONES EN CONSOLA =====
    print("\n===== DETECCIONES SVM =====")

    if len(detections) == 0:
        print("No se detectó ningún objeto.")
    else:
        for i, det in enumerate(detections, start=1):
            print(
                f"{i}. Clase: {det['class']} | "
                f"Confianza: {det['confidence']:.3f} | "
                f"Centro: ({det['x']}, {det['y']}) | "
                f"BBox: {det['bbox']}"
            )

    # --- FILTRADO DE FONDO POR VARIANZA (SOLUCIÓN AL FONDO VERDE) ---
    detections_filtradas = []
    for det in detections:
        x, y = det['x'], det['y']

        # Extraer el fragmento de la imagen evaluada (Ventana de 100x100 centrada en x, y)
        h, w, _ = board_image.shape
        x_min = max(0, x - 50)
        x_max = min(w, x + 50)
        y_min = max(0, y - 50)
        y_max = min(h, y + 50)

        ventana = board_image[y_min:y_max, x_min:x_max]
        if ventana.size == 0:
            continue

        # Calcular desviación estándar de los colores
        ventana_gris = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
        _, desviacion = cv2.meanStdDev(ventana_gris)

        # Si la ventana es muy homogénea (puro fondo gris liso), su desviación será muy baja.
        # Solo conservamos ventanas con texturas o cambios de color considerables (> 12.0)
        if desviacion[0][0] > 12.0:
            detections_filtradas.append(det)

    print(f"\nObjetos reales post-filtrado de fondo: {len(detections_filtradas)}")

    print("\n===== DETECCIONES FINALES =====")

    if len(detections_filtradas) == 0:
        print("No quedaron detecciones después del filtrado.")
    else:
        for i, det in enumerate(detections_filtradas, start=1):
            print(
                f"{i}. Clase: {det['class']} | "
                f"Confianza: {det['confidence']:.3f} | "
                f"Centro: ({det['x']}, {det['y']}) | "
                f"BBox: {det['bbox']}"
            )

    img_detecciones = board_image.copy()
    for det in detections_filtradas:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(img_detecciones, (x1, y1), (x2, y2), (0, 255, 0), 2)

        clase_str = str(det['class'][0]) if isinstance(det['class'], np.ndarray) else str(det['class'])
        texto = f"{clase_str} ({det['confidence']:.2f})"

        cv2.putText(img_detecciones, texto, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    nombre_base, extension = os.path.splitext(ruta_imagen)
    ruta_salida = f"{nombre_base}_detecciones_svm{extension}"

    cv2.imwrite(ruta_salida, img_detecciones)
    print(f"\nImagen con detecciones guardada en: {ruta_salida}")

    cv2.imshow("Detecciones SVM", img_detecciones)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Se retorna la lista de detecciones finales (cada una con su clase y posición)
    # para que quien llame a la función pueda usarlas más allá de la consola/imagen.
    return detections_filtradas


if __name__ == "__main__":

    RUTA_DATASET = r"dataset\Entrenamiento"

    RUTA_IMAGEN_TABLERO = r"dataset\prueba1\ala.jpg"
    RUTA_IMAGEN_Circulo = r"dataset\Entrenamiento\Class_2\WIN_20260702_19_15_53_Pro.jpg"
    RUTA_IMAGEN_Cuadrado = r"dataset\Entrenamiento\Class_1\WIN_20260702_19_10_43_Pro.jpg"

    # Se entrena con la CARPETA del dataset completo, no con una imagen suelta
    model_svm, X_train, y_train = entrenar_svm(RUTA_DATASET, usar_pca=False)
    extractor = features_extractor(hist_bins=16, debug=False)
    CONFIDENCE_THRESHOLD = 0.65

    resultados = probar_svm_en_imagen(model_svm, RUTA_IMAGEN_TABLERO, extractor, CONFIDENCE_THRESHOLD)

    print("\n===== RESUMEN: OBJETOS Y POSICIÓN =====")
    if not resultados:
        print("No se obtuvieron objetos.")
    else:
        for i, det in enumerate(resultados, start=1):
            clase_str = str(det['class'][0]) if isinstance(det['class'], np.ndarray) else str(det['class'])
            x1, y1, x2, y2 = det['bbox']
            print(
                f"{i}. Clase: {clase_str} | "
                f"Centro (x, y): ({det['x']}, {det['y']}) | "
                f"BBox (x1,y1,x2,y2): ({x1}, {y1}, {x2}, {y2})"
            )

    print("PRUEBA COMPLETADA")