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
    
    def _predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            exponent = np.exp(-((x - self.mean[c])**2) / (2 * self.var[c]))
            prob = (1 / np.sqrt(2 * np.pi * self.var[c])) * exponent
            posteriors[c] = np.sum(np.log(prob)) + np.log(self.prior[c])
        
        return max(posteriors, key=posteriors.get)


    def predict_with_confidence(self, x):                #Predice con confianza, donde por una probabilidad donde mayor es mejor, se utiliza para localizar por medio de ventana deslizante,
                                                         #solo acepta si la confianza supera el umbral definido
        x = np.array(x)
        log_posteriors = {}
        for c in self.classes:
            exponent = np.exp(-((x - self.mean[c])**2) / (2 * self.var[c]))
            prob = (1 / np.sqrt(2 * np.pi * self.var[c])) * exponent
            log_posteriors[c] = np.sum(np.log(prob)) + np.log(self.prior[c])

        #Vuelve logs futuros a probabilidades normalizadas numericamente estables, restando el maximo antes de exponenciar
        max_log = max(log_posteriors.values())
        exp_vals = {c: np.exp(v - max_log) for c, v in log_posteriors.items()}
        total = sum(exp_vals.values())
        probs = {c: v / total for c, v in exp_vals.items()}

        label = max(probs, key=probs.get)
        confidence = probs[label]
        return label, confidence
