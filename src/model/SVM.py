import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA

class SVM:

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'): #clasifica
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
        
    def fit(self, X, y): #Entrena

        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_with_distance(self, x):
        """
        Predice la clase y calcula la distancia absoluta al hiperplano de decisión.
        A mayor distancia, mayor es la certeza/confianza de la predicción.
        """
        x = np.array(x).reshape(1, -1)
        
        label = self.model.predict(x)[0]
        
        # 2. Calcular la distancia al hiperplano de decisión (Métrica de confianza real)
        # decision_function devuelve un arreglo. Tomamos el valor absoluto del primer elemento.
        # En clasificación multiclase, devuelve la distancia para cada combinación de clases (one-vs-one)
        # por lo que tomamos la distancia máxima o promedio como métrica de certeza.
        scores = self.model.decision_function(x)[0]
        confidence_score = np.max(np.abs(scores)) 
        
        return label, confidence_score


class SVM_PCA:
    def __init__(self, n_components=50, kernel='rbf', C=1.0, gamma='scale'):
        self.pca = PCA(n_components=n_components)
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=False)
        
    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        self.model.fit(X_reduced, y)
        
    def predict(self, X):
        X_reduced = self.pca.transform(X)
        return self.model.predict(X_reduced) 
        
    def predict_with_distance(self, x):
        x = np.array(x).reshape(1, -1)
        x_reduced = self.pca.transform(x)
        
        label = self.model.predict(x_reduced)[0]
        
        #Calculado sobre el espacio transformado de PCA usando decision_function
        scores = self.model.decision_function(x_reduced)[0]
        confidence_score = np.max(np.abs(scores))
        
        return label, confidence_score


def sliding_window_localization_svm(board_image, model, extractor, window_size=(100, 100), step=20, confidence_threshold=0.5):
    """ ventana deslizante para localizar y clasificar objetos usando SVM."""
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
            
            # Predecir clase y confianza (distancia al hiperplano)
            label, confidence = model.predict_with_distance(features)
            
            # CORREGIDO: Al usar decision_function, buscamos que supere un umbral mínimo de certeza
            if confidence > confidence_threshold:
                detections.append({
                    'x': x + window_size[0]//2,
                    'y': y + window_size[1]//2,
                    'class': label,
                    'confidence': confidence
                })
    
    # CORREGIDO: Ordenar detecciones de mayor a menor confianza para asegurar que el NMS
    # conserve la ventana que mejor centrada está sobre el objeto.
    detections = sorted(detections, key=lambda k: k['confidence'], reverse=True)
    
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
