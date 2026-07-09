#ENTREnar
#ARCHIVO CVS 
#aLGORITMO
#TEST
#resive la ruta de una imagen x
#PROCESO IMAGEN , GUARDAS EL VECTOR

import cv2
import numpy as np
import os
# from SVM_algorithm import SVM, SVM_PCA, sliding_window_localization_svm
from Extraccion_caracteristicas import features_extractor
from sklearn.svm import SVC
from sklearn.decomposition import PCA

### NUEVO ###

class SVM:
    """
    Clasificador SVM estándar con preprocesamiento.
    """
    def __init__(self, kernel='rbf', C=10.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Entrena el modelo SVM"""
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X):
        """Predice la clase para cada muestra"""
        return self.model.predict(X)
    
    def decision_function(self, X):
        """Devuelve la distancia al hiperplano de decisión"""
        return self.model.decision_function(X)


class SVM_PCA:
    """
    Clasificador SVM con reducción de dimensionalidad PCA.
    """
    def __init__(self, n_components=50, kernel='rbf', C=10.0, gamma='scale'):
        self.n_components = n_components
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.pca = PCA(n_components=n_components)
        self.model = None
        self.classes_ = None
    
    def fit(self, X, y):
        """Entrena el modelo SVM con PCA"""
        X_pca = self.pca.fit_transform(X)
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.model.fit(X_pca, y)
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X):
        """Predice usando PCA + SVM"""
        X_pca = self.pca.transform(X)
        return self.model.predict(X_pca)
    
    def decision_function(self, X):
        """Devuelve la distancia al hiperplano de decisión"""
        X_pca = self.pca.transform(X)
        return self.model.decision_function(X_pca)


def sliding_window_localization_svm(board_image, model, extractor, window_size=(100, 100), step=20, dist_threshold=15.0):
    """
    Localiza objetos en una imagen usando ventana deslizante y SVM.
    
    Parámetros:
    - board_image: Imagen del tablero
    - model: Modelo SVM entrenado
    - extractor: Extractor de características
    - window_size: Tamaño de la ventana (ancho, alto)
    - step: Paso de deslizamiento (píxeles)
    - dist_threshold: Umbral de distancia para filtrado
    
    Retorna:
    - Lista de detecciones con formato:
        {'x': x_centro, 'y': y_centro, 'class': clase, 'confidence': confianza}
    """
    detections = []
    h, w = board_image.shape[:2]
    win_w, win_h = window_size
    
    # Recorrer la imagen con ventana deslizante
    for y in range(0, h - win_h + 1, step):
        for x in range(0, w - win_w + 1, step):
            # Extraer la ventana
            window = board_image[y:y+win_h, x:x+win_w]
            
            # Extraer características de la ventana
            features = extractor.extract(window)
            
            if np.sum(features) == 0:
                continue
            
            features = np.array(features).reshape(1, -1)
            
            # Predecir con SVM
            try:
                pred = model.predict(features)[0]
                confidence = model.decision_function(features)
                
                # Si la confianza supera el umbral, guardar detección
                if np.max(confidence) > dist_threshold:
                    # Centro de la ventana
                    centro_x = x + win_w // 2
                    centro_y = y + win_h // 2
                    
                    detections.append({
                        'x': centro_x,
                        'y': centro_y,
                        'class': pred,
                        'confidence': np.max(confidence)
                    })
            except:
                continue
    
    # Filtrar detecciones duplicadas (NMS simple)
    detections = non_max_suppression(detections)
    
    return detections


def non_max_suppression(detections, overlap_thresh=0.3):
    """
    Supresión no máxima para eliminar detecciones duplicadas.
    """
    if len(detections) == 0:
        return detections
    
    # Ordenar por confianza descendente
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Tomar la detección con mayor confianza
        best = detections.pop(0)
        keep.append(best)
        
        # Filtrar las que se solapan demasiado
        detections = [d for d in detections if not boxes_overlap(best, d, overlap_thresh)]
    
    return keep


def boxes_overlap(det1, det2, threshold=0.3):
    """
    Verifica si dos detecciones se solapan.
    """
    x1, y1 = det1['x'], det1['y']
    x2, y2 = det2['x'], det2['y']
    
    # Distancia euclidiana entre centros
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Si están demasiado cerca, se solapan
    if dist < 50:  # Umbral fijo (puedes ajustarlo)
        return True
    
    return False

### NUEVO ###

def cargar_dataset(ruta_dataset):
    """
    Carga el dataset de entrenamiento desde una estructura de carpetas.
    
    Estructura esperada:
    dataset/
        Class_1/
            imagen1.jpg
            imagen2.jpg
        Class_2/
            imagen3.jpg
            imagen4.jpg
        ...
    
    Retorna:
    - X: Array de características extraídas
    - y: Array de etiquetas (nombres de clases)
    """
    X = []
    y = []
    
    extractor = features_extractor(hist_bins=16, debug=False)
    
    # Recorrer todas las carpetas de clases
    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        
        if not os.path.isdir(ruta_clase):
            continue
        
        print(f"Cargando clase: {clase}")
        
        # Recorrer todas las imágenes de la clase
        for archivo in os.listdir(ruta_clase):
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
    
    Parámetros:
    - ruta_dataset: Ruta al directorio del dataset
    - usar_pca: Si True, usa SVM con PCA
    - n_components: Número de componentes principales (solo si usar_pca=True)
    
    Retorna:
    - model: Modelo SVM entrenado
    - X_train: Datos de entrenamiento
    - y_train: Etiquetas de entrenamiento
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
    
    model.fit(X, y)
    print("✓ Modelo entrenado exitosamente")
    
    return model, X, y


def probar_svm_en_imagen(model, ruta_imagen, extractor, dist_threshold=15.0):
    """
    Prueba el modelo SVM en una imagen del tablero.
    
    Parámetros:
    - model: Modelo SVM entrenado
    - ruta_imagen: Ruta a la imagen del tablero
    - extractor: Extractor de características
    - dist_threshold: Umbral de distancia para filtrar fondo
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
    detections = sliding_window_localization_svm(
        board_image, 
        model=model, 
        extractor=extractor, 
        window_size=(100, 100), 
        step=20, 
        dist_threshold=dist_threshold
    )
    
    print(f"\n✓ Objetos detectados: {len(detections)}")
    
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
    
    # Mostrar imagen (opcional)
    cv2.imshow("Detecciones SVM", img_detecciones)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ==========================================================
    # CONFIGURACIÓN - AJUSTA ESTAS RUTAS SEGÚN TU PROYECTO
    # ==========================================================
    
    # Ruta al dataset de entrenamiento
    RUTA_DATASET = ("C:\\Users\\josep\\OneDrive\\Documentos\\GitHub\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica-main\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica-main\\dataset\\Entrenamiento")
    
    # Ruta a la imagen del tablero para probar
    RUTA_IMAGEN_TABLERO = ("C:\\Users\\josep\\OneDrive\\Documentos\\GitHub\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica-main\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica-main\\dataset\\Test\\WIN_20260702_17_07_52_Pro.jpg")
    
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
    # Valores típicos: 10.0 - 20.0 para SVM
    DIST_THRESHOLD = 15.0
    
    probar_svm_en_imagen(
        model=model_svm,
        ruta_imagen=RUTA_IMAGEN_TABLERO,
        extractor=extractor,
        dist_threshold=DIST_THRESHOLD
    )
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA")
    print("=" * 60)