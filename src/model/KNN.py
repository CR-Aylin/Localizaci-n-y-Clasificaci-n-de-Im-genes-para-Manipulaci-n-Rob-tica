import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

# Ventana Deslizante
    def predict_with_distance(self, x):
        x = np.array(x)
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]
        label = Counter(k_labels).most_common(1)[0][0]
        mean_dist = np.mean(k_distances)
        return label, mean_dist

# Localización de Ventana Deslizante
def sliding_window_localization(board_image, model, extractor, window_size=(100, 100), step=20, dist_threshold=500):
    """
    Recorre el tablero con una ventana deslizante para localizar y clasificar objetos.
    dist_threshold: Umbral de distancia euclidiana para considerar que hay un objeto y no fondo.
    """
    h, w = board_image.shape[:2]
    detections = []

    for y in range(0, h - window_size[1] + 1, step):
        for x in range(0, w - window_size[0] + 1, step):
            window = board_image[y:y+window_size[1], x:x+window_size[0]]

            # Extraer características de la ventana
            features = extractor.extract(window)

            # Si la ventana está vacía (fondo), el extractor retorna ceros y la distancia será alta
            if np.sum(features) == 0:
                continue

            # Predecir clase y distancia (confianza)
            label, mean_dist = model.predict_with_distance(features)

            # Si la distancia es menor al umbral, consideramos que hay un objeto
            if mean_dist < dist_threshold:
                detections.append({
                    'x': x + window_size[0]//2,
                    'y': y + window_size[1]//2,
                    'class': label,
                    'confidence': 1.0 / (mean_dist + 1e-6) # Mayor confianza si la distancia es menor
                })

    # Eliminar detecciones duplicadas 
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
