#ENTREnar
#ARCHIVO CVS 
#aLGORITMO
#TEST
#resive la ruta de una imagen x
#PROCESO IMAGEN , GUARDAS EL VECTOR

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA


class SVM:
    """Clasificador SVM independiente (usa scikit-learn)"""
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
        
    def fit(self, X, y):
        """Entrena el modelo SVM."""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Predice las clases para un conjunto de datos."""
        return self.model.predict(X)
        
    def predict_with_distance(self, x):
        """
        Predice la clase y calcula la distancia al vector de soporte más cercano.
        Esta distancia se usa como métrica de confianza (menor = más confiable).
        """
        x = np.array(x).reshape(1, -1)
        
        # 1. Predecir la clase
        label = self.model.predict(x)[0]
        
        # 2. Calcular distancia al vector de soporte más cercano
        support_vectors = self.model.support_vectors_
        distances = np.sqrt(np.sum((support_vectors - x)**2, axis=1))
        min_dist = np.min(distances)
        
        return label, min_dist


class SVM_PCA:
    """SVM con reducción dimensional PCA"""
    def __init__(self, n_components=50, kernel='rbf', C=1.0, gamma='scale'):
        self.pca = PCA(n_components=n_components)
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
        
    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        self.model.fit(X_reduced, y)
        
    def predict(self, X):
        X_reduced = self.pca.transform(X)
        return self.model.predict(X)
        
    def predict_with_distance(self, x):
        x = np.array(x).reshape(1, -1)
        x_reduced = self.pca.transform(x)
        
        label = self.model.predict(x_reduced)[0]
        
        support_vectors = self.model.support_vectors_
        distances = np.sqrt(np.sum((support_vectors - x_reduced)**2, axis=1))
        min_dist = np.min(distances)
        
        return label, min_dist


def sliding_window_localization_svm(board_image, model, extractor, 
                                     window_size=(100, 100), step=20, dist_threshold=15.0):
    """
    Recorre el tablero con una ventana deslizante para localizar y clasificar objetos usando SVM.
    
    Parámetros:
    - board_image: Imagen del tablero a analizar
    - model: Modelo SVM (instancia de SVM o SVM_PCA)
    - extractor: Extractor de características
    - window_size: Tamaño de la ventana (ancho, alto)
    - step: Paso del deslizamiento
    - dist_threshold: Umbral de distancia para considerar que hay un objeto
    
    Retorna:
    - Lista de detecciones con posición, clase y confianza
    """
    h, w = board_image.shape[:2]
    detections = []
    
    for y in range(0, h - window_size[1] + 1, step):
        for x in range(0, w - window_size[0] + 1, step):
            window = board_image[y:y+window_size[1], x:x+window_size[0]]
            
            # Extraer características de la ventana
            features = extractor.extract(window)
            
            # Si la ventana está vacía (fondo), el extractor retorna ceros
            if np.sum(features) == 0:
                continue
            
            # Predecir clase y distancia (confianza)
            label, min_dist = model.predict_with_distance(features)
            
            # Si la distancia es menor al umbral, consideramos que hay un objeto
            if min_dist < dist_threshold:
                detections.append({
                    'x': x + window_size[0]//2,
                    'y': y + window_size[1]//2,
                    'class': label,
                    'confidence': 1.0 / (min_dist + 1e-6)
                })
    
    # Eliminar detecciones duplicadas (Non-Maximum Suppression simple)
    final_detections = []
    for det in detections:
        is_duplicate = False
        for final_det in final_detections:
            if det['class'] == final_det['class'] and \
               abs(det['x'] - final_det['x']) < window_size[0]//2 and \
               abs(det['y'] - final_det['y']) < window_size[1]//2:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_detections.append(det)
    
    return final_detections