from collections import Counter

import cv2
import numpy as np
import os
import time
from src.model.SVM import SVM, SVM_PCA, sliding_window_localization_svm
from src.model import Extraccion_caracteristicas as ec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_dataset(ruta_dataset):
    X = []
    y = []

    extractor = ec.features_extractor(hist_bins=16, debug=False)

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

def calcular_confidence_threshold(X_ref, model, percentil=5, factor=0.5):
    confidencias = [model.predict_with_distance(x)[1] for x in X_ref]
    base = np.percentile(confidencias, percentil)
    return base * factor

def evaluar_svm(X, y, usar_pca=False, n_components=50, test_size=0.2, random_state=42):
    """
    Evalúa el SVM con un split hold-out y muestra su matriz de confusión,
    igual que se hace para KNN.
    """
    clases = sorted(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if usar_pca:
        modelo_eval = SVM_PCA(n_components=n_components, kernel='rbf', C=10.0, gamma='scale')
    else:
        modelo_eval = SVM(kernel='rbf', C=10.0, gamma='scale')

    modelo_eval.fit(X_train, y_train)
    predicciones = modelo_eval.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN SVM {'CON PCA' if usar_pca else 'SIN PCA'}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test, predicciones, labels=clases, target_names=clases, zero_division=0))

    cm = confusion_matrix(y_test, predicciones, labels=clases)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz Confusión SVM {'(PCA)' if usar_pca else '(Sin PCA)'}")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout(); plt.show()

    return accuracy

def probar_svm_en_imagen(model, ruta_imagen, extractor, confidence_threshold=0.5,
                          window_size=(100, 100), step=20, mostrar_ventana=True):
    print(f"PROBANDO SVM EN IMAGEN: {ruta_imagen}")

    board_image = cv2.imread(ruta_imagen)
    if board_image is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return None

    print(f"Imagen cargada: {board_image.shape}")
    print("\nAplicando ventana deslizante...")

    inicio = time.time()
    detections = sliding_window_localization_svm(
        board_image,
        model=model,
        extractor=extractor,
        window_size=window_size,
        step=step,
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

    if mostrar_ventana:
        cv2.imshow("Detecciones SVM", img_detecciones)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Se retorna la lista de detecciones finales (cada una con su clase y posición)
    # para que quien llame a la función pueda usarlas más allá de la consola/imagen.
    return detections_filtradas


def ejecutar_deteccion(
    ruta_dataset=r"dataset\Entrenamiento",
    ruta_imagen_tablero=r"dataset\prueba1\ala.jpg",
    confidence_threshold=0.65,
    usar_pca=False,
    hist_bins=16,
    window_size=(100, 100),
    step=20,
    mostrar_ventana=True
):

    # Se entrena con la CARPETA del dataset completo, no con una imagen suelta
    model_svm, X_train, y_train = entrenar_svm(ruta_dataset, usar_pca=usar_pca)

    evaluar_svm(X_train, y_train, usar_pca=usar_pca)

    #confidence_threshold_calibrado = calcular_confidence_threshold(X_train, model_svm)
    #print(f"Umbral de confianza calibrado: {confidence_threshold_calibrado:.4f} "f"(se ignora el valor fijo recibido: {confidence_threshold})")

    extractor = ec.features_extractor(hist_bins=hist_bins, debug=False)

    resultados = probar_svm_en_imagen(
        model_svm,
        ruta_imagen_tablero,
        extractor,
        confidence_threshold=0.05,
        window_size=window_size,
        step=step,
        mostrar_ventana=mostrar_ventana
    )

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

    return resultados

def ejecutar_deteccion_PCA(
    ruta_dataset=r"dataset\Entrenamiento",
    ruta_imagen_tablero=r"dataset\prueba1\ala.jpg",
    confidence_threshold=0.65,
    n_components=50,
    hist_bins=16,
    window_size=(100, 100),
    step=20,
    mostrar_ventana=True
):
    """
    Ejecuta la detección utilizando un clasificador SVM + PCA.
    """

    print("===== SVM + PCA =====")

    # Entrenar usando PCA
    model_svm, X_train, y_train = entrenar_svm(
        ruta_dataset=ruta_dataset,
        usar_pca=True,
        n_components=n_components
    )

    evaluar_svm(X_train, y_train, usar_pca=True, n_components=n_components)

    #confidence_threshold_calibrado = calcular_confidence_threshold(X_train, model_svm)
    #print(f"Umbral de confianza calibrado: {confidence_threshold_calibrado:.4f} "f"(se ignora el valor fijo recibido: {confidence_threshold})")

    extractor = ec.features_extractor(
        hist_bins=hist_bins,
        debug=False
    )

    resultados = probar_svm_en_imagen(
        model=model_svm,
        ruta_imagen=ruta_imagen_tablero,
        extractor=extractor,
        confidence_threshold=0.05,
        window_size=window_size,
        step=step,
        mostrar_ventana=mostrar_ventana
    )

    print("\n===== RESUMEN =====")

    if not resultados:
        print("No se detectaron objetos.")
    else:
        for i, det in enumerate(resultados, start=1):
            clase = (
                str(det["class"][0])
                if isinstance(det["class"], np.ndarray)
                else str(det["class"])
            )

            x1, y1, x2, y2 = det["bbox"]

            print(
                f"{i}. Clase: {clase} | "
                f"Centro: ({det['x']}, {det['y']}) | "
                f"BBox: ({x1}, {y1}, {x2}, {y2}) | "
                f"Confianza: {det['confidence']:.3f}"
            )

    print("PRUEBA COMPLETADA")

    return resultados


if __name__ == "__main__":
    ejecutar_deteccion(
        ruta_dataset=r"dataset\Entrenamiento",
        ruta_imagen_tablero=r"dataset\prueba1\ala.jpg",
        confidence_threshold=0.65
    )

    ejecutar_deteccion_PCA(
        ruta_dataset=r"dataset\Entrenamiento",
        ruta_imagen_tablero=r"dataset\prueba1\ala.jpg",
        confidence_threshold=0.65,
        n_components=50
    )