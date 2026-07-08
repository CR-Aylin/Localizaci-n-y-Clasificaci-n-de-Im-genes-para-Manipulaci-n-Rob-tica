import numpy as np

from Entrenar_NaiveBayes import (
    cargar_dataset_desde_carpetas, train_test_split_manual, normalizar,
    matriz_confusion, precision_recall_f1
)
from Extractor_Seguro_Bayes import ExtractorSeguro
from NaiveBayes import NaiveBayes


def agregar_ruido_gaussiano(imagen, sigma):
    """Ruido gaussiano aditivo, sigma en escala de píxeles (0-255)."""
    ruido = np.random.normal(0, sigma, imagen.shape)
    return np.clip(imagen.astype(float) + ruido, 0, 255).astype(np.uint8)


def cambiar_iluminacion(imagen, factor_brillo):
    """factor_brillo > 1 aclara la imagen, < 1 la oscurece."""
    return np.clip(imagen.astype(float) * factor_brillo, 0, 255).astype(np.uint8)


def exactitud_del_modelo(modelo, X, y, medias, stds):
    X_norm = (X - medias) / stds
    y_pred = modelo.predict(X_norm)
    M = matriz_confusion(y, y_pred, sorted(set(y)))
    return np.trace(M) / M.sum()


def extraer_features_transformadas(rutas_e_imagenes, extractor, transformar, largo_esperado=None):
    """
    rutas_e_imagenes: lista de (imagen_bgr, clase).
    transformar: función que recibe una imagen y devuelve la imagen modificada
    (ruido o iluminación).
    largo_esperado: tamaño de vector que espera el modelo (medias/stds). Si
    no se indica, se usa el largo más frecuente del lote.
    """
    from collections import Counter

    X_bruto, y_bruto = [], []
    for img, clase in rutas_e_imagenes:
        img_transformada = transformar(img)
        try:
            features = extractor.extract(img_transformada)
        except Exception:
            continue
        if features is None:
            continue
        X_bruto.append(np.asarray(features, dtype=float))
        y_bruto.append(clase)

    if not X_bruto:
        return np.array([]), np.array([])

    if largo_esperado is None:
        largos = Counter(len(v) for v in X_bruto)
        largo_esperado = largos.most_common(1)[0][0]

    X, y = [], []
    for vec, clase in zip(X_bruto, y_bruto):
        if len(vec) == largo_esperado:
            X.append(vec)
            y.append(clase)

    return np.array(X, dtype=float), np.array(y)


def evaluar_robustez_ruido(imagenes_test, modelo, extractor, medias, stds,
                            niveles_sigma=(0, 5, 10, 20, 30, 50)):
    print("\n=== Robustez frente a ruido gaussiano ===")
    print(f"{'Sigma':>8} {'Exactitud':>12} {'Muestras usadas':>18}")
    resultados = []
    for sigma in niveles_sigma:
        transformar = lambda img, s=sigma: agregar_ruido_gaussiano(img, s)
        X, y = extraer_features_transformadas(imagenes_test, extractor, transformar,
                                               largo_esperado=len(medias))
        if len(X) == 0:
            print(f"{sigma:>8} {'sin datos':>12} {0:>18}")
            continue
        exactitud = exactitud_del_modelo(modelo, X, y, medias, stds)
        print(f"{sigma:>8} {exactitud:>11.2%} {len(X):>18}")
        resultados.append((sigma, exactitud))
    return resultados


def evaluar_sensibilidad_iluminacion(imagenes_test, modelo, extractor, medias, stds,
                                      factores_brillo=(0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)):
    print("\n=== Sensibilidad a cambios de iluminación ===")
    print(f"{'Factor':>8} {'Exactitud':>12} {'Muestras usadas':>18}")
    resultados = []
    for factor in factores_brillo:
        transformar = lambda img, f=factor: cambiar_iluminacion(img, f)
        X, y = extraer_features_transformadas(imagenes_test, extractor, transformar,
                                               largo_esperado=len(medias))
        if len(X) == 0:
            print(f"{factor:>8.2f} {'sin datos':>12} {0:>18}")
            continue
        exactitud = exactitud_del_modelo(modelo, X, y, medias, stds)
        print(f"{factor:>8.2f} {exactitud:>11.2%} {len(X):>18}")
        resultados.append((factor, exactitud))
    return resultados


def correr_experimentos_robustez(ruta_dataset, test_size=0.2):
    """
    Entrena Naive Bayes sobre el dataset original (sin ruido/iluminación
    alterada) y evalúa la exactitud sobre el conjunto de test cuando se le
    aplican distintos niveles de ruido y de cambio de brillo.
    """
    import os
    import cv2

    extractor = ExtractorSeguro()
    X, y = cargar_dataset_desde_carpetas(ruta_dataset, extractor)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=test_size)
    X_train_norm, _ = normalizar(X_train, X_test)
    medias = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1e-9

    modelo = NaiveBayes()
    modelo.fit(X_train_norm, y_train)

    # Releer las imágenes de test (en crudo) para poder alterarlas
    imagenes_test = []
    for clase in ["Circulo", "Poligonos", "Fondo"]:
        ruta_clase = os.path.join(ruta_dataset, clase)
        if not os.path.isdir(ruta_clase):
            continue
        for nombre in os.listdir(ruta_clase):
            if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(ruta_clase, nombre))
            if img is not None:
                imagenes_test.append((img, clase))

    resultados_ruido = evaluar_robustez_ruido(imagenes_test, modelo, extractor, medias, stds)
    resultados_iluminacion = evaluar_sensibilidad_iluminacion(imagenes_test, modelo, extractor, medias, stds)

    return resultados_ruido, resultados_iluminacion


if __name__ == "__main__":
    correr_experimentos_robustez(ruta_dataset="dataset_imagenes")
