import cv2
import numpy as np
import os
import time
from SVM import SVM, SVM_PCA, sliding_window_localization_svm
from Extraccion_caracteristicas import features_extractor
#Algo mal Aqui
def cargar_dataset(ruta_dataset):
    X = []
    y = []
    
    extractor = features_extractor(hist_bins=16, debug=False)
    
    # Valida que la ruta de origen sea realmente una carpeta antes de iterar
    if not os.path.isdir(ruta_dataset):
        raise NotADirectoryError(f"La ruta especificada no es un directorio válido: {ruta_dataset}")
    
    for clase in sorted(os.listdir(ruta_dataset)):
        ruta_clase = os.path.join(ruta_dataset, clase)
        
        if not os.path.isdir(ruta_clase):
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
                
                if np.sum(features) > 0:  
                    X.append(features)
                    y.append(clase)
    
    return np.array(X), np.array(y)


def entrenar_svm(ruta_dataset, usar_pca=False, n_components=50):
    """
    Entrena el modelo SVM con el dataset.
    """
    print("CARGANDO DATASET DE ENTRENAMIENTO")
    
    X, y = cargar_dataset(ruta_dataset)
    
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
    
    print(f"\nObjetos detectados: {len(detections)}")
    print(f"Tiempo de inferencia: {fin - inicio:.4f} segundos")
    
    for i, det in enumerate(detections, 1):
        print(f"\n  Detección {i}:")
        print(f"    Clase: {det['class']}")
        print(f"    Posición: ({det['x']}, {det['y']})")
        print(f"    Confianza: {det['confidence']:.4f}")
    
    img_detecciones = board_image.copy()
    for det in detections:
        cv2.circle(img_detecciones, (det['x'], det['y']), 10, (0, 255, 0), 2)
        
        # Como det['class'] es un array de numpy devuelto por scikit-learn, 
        # extraemos el string usando [0] para evitar errores de putText
        clase_str = str(det['class'][0]) if isinstance(det['class'], np.ndarray) else str(det['class'])
        texto = f"{clase_str} ({det['confidence']:.2f})"
        
        cv2.putText(img_detecciones, texto, (det['x'] - 50, det['y'] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    nombre_base, extension = os.path.splitext(ruta_imagen)
    ruta_salida = f"{nombre_base}_detecciones_svm{extension}"
    
    cv2.imwrite(ruta_salida, img_detecciones)
    print(f"\nImagen con detecciones guardada en: {ruta_salida}")
    
    cv2.imshow("Detecciones SVM", img_detecciones)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # IMPORTANTE: Se añade 'r' para que Windows interprete correctamente las barras invertidas.
    # Esta carpeta DEBE contener los subdirectorios de cada clase (ej: tornillos, tuercas, etc.)
    RUTA_DATASET = r"C:\Users\alumno\Documents\GitHub\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\dataset"
    
    # CORREGIDO: Se añade 'r' al inicio y la extensión '.jpg' al final del archivo.
    RUTA_IMAGEN_TABLERO = r"C:\Users\alumno\Documents\GitHub\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\dataset\prueba1\ala.jpg.jpg"

    # Entrenar SVM estándar
    model_svm, X_train, y_train = entrenar_svm(RUTA_DATASET, usar_pca=False)
    
    # Si prefieres probar con PCA en el futuro, descomenta la siguiente línea:
    # model_svm, X_train, y_train = entrenar_svm(RUTA_DATASET, usar_pca=True, n_components=50)
    
    extractor = features_extractor(hist_bins=16, debug=False)
    CONFIDENCE_THRESHOLD = 0.5
    
    # Ejecutar la detección en la imagen de prueba
    probar_svm_en_imagen(
        model=model_svm,
        ruta_imagen=RUTA_IMAGEN_TABLERO,
        extractor=extractor,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    print("PRUEBA COMPLETADA")
