#ENTREnar
#ARCHIVO CVS 
#aLGORITMO
#TEST
#resive la ruta de una imagen x
#PROCESO IMAGEN , GUARDAS EL VECTOR

import numpy as np

class SVM:
    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000, tol=1e-4, seed=42):
        """
        Parámetros:
            C: Parámetro de regularización. Mayor C = menos tolerante a errores (menos margen).
            learning_rate: Tasa de aprendizaje inicial.
            n_iterations: Número de épocas de entrenamiento.
            tol: Tolerancia para early stopping.
            seed: Semilla para reproducibilidad.
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.seed = seed
        self.w = None
        self.b = None

    def fit(self, X, y):
        #Entrena el SVM binario y debe contener valores +1 o -1.
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape

        # Inicializar pesos y bias en ceros
        self.w = np.zeros(n_features)
        self.b = 0.0

        # SGD: iterar sobre los datos múltiples veces (épocas)
        for epoch in range(self.n_iterations):
            # Barajar los índices en cada época
            indices = rng.permutation(n_samples)
            prev_w = self.w.copy()

            for i in indices:
                xi = X[i]
                yi = y[i]

                # Margen funcional: y * (w·x + b)
                margin = yi * (np.dot(self.w, xi) + self.b)

                # Actualización del gradiente
                if margin >= 1:
                    # Clasificación correcta con margen suficiente: solo regularización
                    self.w -= self.learning_rate * (self.w)
                else:
                    # Error o margen insuficiente: hinge loss + regularización
                    self.w -= self.learning_rate * (self.w - self.C * yi * xi)
                    self.b += self.learning_rate * self.C * yi

            # Tasa de aprendizaje decreciente (schedule)
            self.learning_rate = self.learning_rate / (1 + 0.001 * epoch)

            # Early stopping: si los pesos casi no cambian, detener
            if np.linalg.norm(self.w - prev_w) < self.tol:
                break

        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        #Predice la clase (+1 o -1) para cada muestra.
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


class SVM_OvR:
    #SVM Multiclase usando la estrategia One-vs-Rest (OvR).
    #Entrena un SVM binario por cada clase (clase vs todas las demás).

    def __init__(self, C=1.0, learning_rate=0.01, n_iterations=1000, tol=1e-4, seed=42):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.seed = seed
        self.classes = None
        self.classifiers = {}  # {clase: SVM_binario}

    def fit(self, X, y):
        #Entrena un SVM binario por cada clase.
        #y: array con etiquetas de clase (strings o enteros).
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes = np.unique(y)

        for c in self.classes:
            # Crear etiquetas binarias: +1 para la clase c, -1 para el resto
            y_bin = np.where(y == c, 1, -1)

            # Cada clasificador usa su propia tasa de aprendizaje (copia)
            svm = SVM(
                C=self.C,
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                tol=self.tol,
                seed=self.seed
            )
            svm.fit(X, y_bin)
            self.classifiers[c] = svm

        return self

    def decision_function(self, X):
        # Retorna un diccionario con el score (distancia al hiperplano) de cada clase para cada muestra.
        X = np.array(X, dtype=float)
        scores = {}
        for c in self.classes:
            scores[c] = self.classifiers[c].decision_function(X)
        return scores

    def predict(self, X):
        # Predice la clase asignando la muestra al clasificador con mayor score.
        X = np.array(X, dtype=float)
        scores = self.decision_function(X)

        # Si es una sola muestra
        if X.ndim == 1:
            return max(self.classes, key=lambda c: scores[c])

        # Si son varias muestras
        predictions = []
        for i in range(X.shape[0]):
            best_class = max(self.classes, key=lambda c: scores[c][i])
            predictions.append(best_class)
        return np.array(predictions)

    def predict_with_confidence(self, x):
        x = np.array(x, dtype=float).reshape(1, -1)
        scores = self.decision_function(x)
        scores = {c: float(scores[c][0]) for c in self.classes}

        # Softmax numéricamente estable sobre los scores
        max_score = max(scores.values())
        exp_vals = {c: np.exp(v - max_score) for c, v in scores.items()}
        total = sum(exp_vals.values())
        probs = {c: v / total for c, v in exp_vals.items()}

        label = max(probs, key=probs.get)
        confidence = probs[label]
        return label, confidence

    def predict_from_image(self, image, extractor, norm_stats=None, pca=None):
        """
        Recibe una imagen, extrae el vector de características, lo normaliza, 
        y devuelve la predicción con confianza.
        Guarda internamente el último vector procesado para depuración.
        """
        # 1. Extraer características de la imagen
        features = extractor.extract(image)
        if features is None:
            return None, 0.0, None

        features = np.asarray(features, dtype=float)
        self.last_features_ = features.copy()  # Guardar el vector resultante

        # 2. Normalizar usando las estadísticas del entrenamiento
        if norm_stats is not None:
            features = (features - norm_stats["medias"]) / norm_stats["stds"]

        # 3. Aplicar PCA si está disponible
        if pca is not None:
            features = pca.transform(features.reshape(1, -1))[0]

        # 4. Predecir
        label, confidence = self.predict_with_confidence(features)
        return label, confidence, features

# ============================================================
# Funciones auxiliares para integrar con el pipeline del proyecto
# ============================================================

def entrenar_svm(X_train, y_train, C=1.0, learning_rate=0.01, n_iterations=1000):
    modelo = SVM_OvR(C=C, learning_rate=learning_rate, n_iterations=n_iterations)
    modelo.fit(X_train, y_train)
    return modelo

if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.5)
    # Convertir etiquetas numéricas a strings (como en el proyecto)
    clases_str = ["Circulo", "Poligonos", "Fondo"]
    y_str = np.array([clases_str[yi] for yi in y])

    # Normalizar
    medias = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1e-9
    X_norm = (X - medias) / stds

    # Entrenar
    svm = SVM_OvR(C=1.0, learning_rate=0.01, n_iterations=500)
    svm.fit(X_norm, y_str)

    # Predecir
    predicciones = svm.predict(X_norm)
    exactitud = np.mean(predicciones == y_str)
    print(f"Exactitud en datos de entrenamiento: {exactitud:.2%}")

    # Probar predict_with_confidence
    muestra = X_norm[0]
    label, conf = svm.predict_with_confidence(muestra)
    print(f"Muestra 0 → Clase: {label}, Confianza: {conf:.3f}")