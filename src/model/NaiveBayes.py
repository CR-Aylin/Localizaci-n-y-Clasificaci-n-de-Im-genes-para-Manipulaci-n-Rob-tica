import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}
    
    def fit(self, X, y):
        X = np.array(X)
        self.classes = np.unique(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9
            self.prior[c] = len(X_c) / len(X)
    
    def predict(self, X):
        X = np.array(X)
        return [self._predict_single(x) for x in X]
    
    def _log_gaussian(self, x, c): # Log de la densidad gaussiana, calculado directamente (sin pasar por exp() y luego log()). 
        var = self.var[c]
        return -((x - self.mean[c]) ** 2) / (2 * var) - 0.5 * np.log(2 * np.pi * var)

    def _predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            posteriors[c] = np.sum(self._log_gaussian(x, c)) + np.log(self.prior[c])
        
        return max(posteriors, key=posteriors.get)


    def predict_with_confidence(self, x):                #Predice con confianza, donde por una probabilidad donde mayor es mejor, se utiliza para localizar por medio de ventana deslizante,
                                                         #solo acepta si la confianza supera el umbral definido
        x = np.array(x)
        log_posteriors = {}
        for c in self.classes:
            log_posteriors[c] = np.sum(self._log_gaussian(x, c)) + np.log(self.prior[c])

        #Vuelve logs futuros a probabilidades normalizadas numericamente estables, restando el maximo antes de exponenciar
        max_log = max(log_posteriors.values())
        exp_vals = {c: np.exp(v - max_log) for c, v in log_posteriors.items()}
        total = sum(exp_vals.values())
        probs = {c: v / total for c, v in exp_vals.items()}

        label = max(probs, key=probs.get)
        confidence = probs[label]
        return label, confidence
