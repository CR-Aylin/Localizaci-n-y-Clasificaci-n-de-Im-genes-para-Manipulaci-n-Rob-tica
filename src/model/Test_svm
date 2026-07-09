import cv2
import numpy as np
import os
import time
from SVM import SVM, SVM_PCA, sliding_window_localization_svm
from Extraccion_caracteristicas import features_extractor


def cargar_dataset(ruta_dataset):
    """
    Carga el dataset de entrenamiento desde una estructura de carpetas.
    
    Estructura esperada:
    ruta_dataset/
        Class_1/
            imagen1.jpg
            imagen2.jpg
        Class_2/
            imagen3.jpg
            imagen4.jpg
    
    Retorna:
    - X: Array de características extraídas
    - y: Array de etiquetas (nombres de clases)
    """
    X = []
    y = []
    
    extractor = features_extractor(hist_bins=16, debug=False)
    
    # Recorrer todas las carpetas de clases
    for clase in sorted(os.listdir(ruta_dataset)):
        ruta_clase = os.path.join(ruta_dataset, clase)
        
        if not os.path.isdir(ruta_clase):
            continue
        
        print(f"Cargando clase: {clase}")
        
        # Recorrer todas las imágenes de la clase
        for archivo in sorted(os.listdir(ruta_clase)):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_imagen = os.path.join(ruta_clase, archivo)
                
                # Cargar imagen
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"  ⚠ No se pudo cargar: {archivo}")
                    continue
                
                # Extraer características
                features = extractor.extract(imagen)
                
                if np.sum(features) > 0:  # Solo si no está vacía
                    X.append(features)
                    y.append(clase)
    
    return np.array(X), np.array(y)


def entrenar_svm(ruta_dataset, usar_pca=False, n_components=50):
    """
    Entrena el modelo SVM con el dataset.
    """
    print("=" * 60)
    print("CARGANDO DATASET DE ENTRENAMIENTO")
    print("=" * 60)
    
    X, y = cargar_dataset(ruta_dataset)
    
    print(f"\n✓ Dataset cargado: {len(X)} muestras")
    print(f"✓ Clases encontradas: {np.unique(y)}")
    
    # Crear y entrenar modelo
    if usar_pca:
        print(f"\nEntrenando SVM con PCA (n_components={n_components})...")
        model = SVM_PCA(n_components=n_components, kernel='rbf', C=10.0, gamma='scale')
    else:
        print("\nEntrenando SVM estándar...")
        model = SVM(kernel='rbf', C=10.0, gamma='scale')  
    
    inicio = time.time()
    model.fit(X, y)
    fin = time.time()
    
    print(f"✓ Modelo entrenado exitosamente")
    print(f"⏱ Tiempo de entrenamiento: {fin - inicio:.4f} segundos")
    
    return model, X, y


def probar_svm_en_imagen(model, ruta_imagen, extractor, dist_threshold=15.0):
    """
    Prueba el modelo SVM en una imagen del tablero.
    """
    print("\n" + "=" * 60)
    print(f"PROBANDO SVM EN IMAGEN: {ruta_imagen}")
    print("=" * 60)
    
    # Cargar imagen del tablero
    board_image = cv2.imread(ruta_imagen)
    if board_image is None:
        print(f"✗ Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    print(f"✓ Imagen cargada: {board_image.shape}")
    
    # Aplicar ventana deslizante
    print("\nAplicando ventana deslizante...")
    inicio = time.time()
    detections = sliding_window_localization_svm(
        board_image, 
        model=model, 
        extractor=extractor, 
        window_size=(100, 100), 
        step=20, 
        dist_threshold=dist_threshold
    )
    fin = time.time()
    
    print(f"\n✓ Objetos detectados: {len(detections)}")
    print(f"⏱ Tiempo de inferencia: {fin - inicio:.4f} segundos")
    
    # Mostrar detecciones
    for i, det in enumerate(detections, 1):
        print(f"\n  Detección {i}:")
        print(f"    Clase: {det['class']}")
        print(f"    Posición: ({det['x']}, {det['y']})")
        print(f"    Confianza: {det['confidence']:.4f}")
    
    # Dibujar detecciones en la imagen
    img_detecciones = board_image.copy()
    for det in detections:
        # Dibujar círculo en la posición detectada
        cv2.circle(img_detecciones, (det['x'], det['y']), 10, (0, 255, 0), 2)
        
        # Mostrar etiqueta de clase
        texto = f"{det['class']}"
        cv2.putText(img_detecciones, texto, (det['x'] - 30, det['y'] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Guardar imagen con detecciones
    ruta_salida = ruta_imagen.replace('.jpg', '_detecciones_svm.jpg')
    cv2.imwrite(ruta_salida, img_detecciones)
    print(f"\n✓ Imagen con detecciones guardada en: {ruta_salida}")
    
    # Mostrar imagen
    cv2.imshow("Detecciones SVM", img_detecciones)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ==========================================================
    # CONFIGURACIÓN - AJUSTA ESTAS RUTAS SEGÚN TU PROYECTO
    # ==========================================================
    
    # Ruta al dataset de entrenamiento (carpeta con subcarpetas de clases)
    RUTA_DATASET = ("C:\\Users\\josep\\OneDrive\\Documentos\\GitHub\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\\dataset\\Entrenamiento")
    
    # Ruta a la imagen del tablero para probar (UNA imagen con ambos objetos)
    RUTA_IMAGEN_TABLERO = ("C:\\Users\\josep\\OneDrive\\Documentos\\GitHub\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\\dataset\\Test\\WIN_20260702_17_07_52_Pro.jpg")
    
    # ==========================================================
    # PASO 1: Entrenar el modelo SVM
    # ==========================================================
    
    # Opción A: SVM estándar
    model_svm, X_train, y_train = entrenar_svm(RUTA_DATASET, usar_pca=False)
    
    # Opción B: SVM con PCA (descomenta si quieres probarlo)
    # model_svm, X_train, y_train = entrenar_svm(RUTA_DATASET, usar_pca=True, n_components=50)
    
    # ==========================================================
    # PASO 2: Crear extractor de características
    # ==========================================================
    extractor = features_extractor(hist_bins=16, debug=False)
    
    # ==========================================================
    # PASO 3: Probar el SVM en una imagen del tablero
    # ==========================================================
    
    # NOTA: Deberás calibrar el dist_threshold según tus resultados
    DIST_THRESHOLD = 2.0
    
    probar_svm_en_imagen(
        model=model_svm,
        ruta_imagen=RUTA_IMAGEN_TABLERO,
        extractor=extractor,
        dist_threshold=DIST_THRESHOLD
    )
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA")
    print("=" * 60)