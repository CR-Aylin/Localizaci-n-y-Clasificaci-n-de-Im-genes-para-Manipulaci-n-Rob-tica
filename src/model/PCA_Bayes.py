import numpy as np


class PCA:
    """
    PCA propio vía SVD. Se indica n_components o explained_variance_ratio
    (ej. 0.95 para conservar el 95% de la varianza).
    """

    def __init__(self, n_components=None, explained_variance_ratio=None):
        if n_components is None and explained_variance_ratio is None:
            raise ValueError("Debes indicar n_components o explained_variance_ratio.")
        self.n_components = n_components
        self.explained_variance_ratio_target = explained_variance_ratio

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centrado = X - self.mean_

        _, S, Vt = np.linalg.svd(X_centrado, full_matrices=False)

        varianza_total = np.sum(S ** 2)
        varianza_explicada = (S ** 2) / varianza_total

        if self.n_components is not None:
            k = min(self.n_components, X.shape[1])
        else:
            acumulada = np.cumsum(varianza_explicada)
            k = int(np.searchsorted(acumulada, self.explained_variance_ratio_target) + 1)
            k = min(k, X.shape[1])

        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = varianza_explicada[:k]
        self.n_components_ = k
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X_centrado = X - self.mean_
        return X_centrado @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
