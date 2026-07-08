import os
import time
from collections import Counter

import cv2
import numpy as np

from Extractor_Seguro_Bayes import ExtractorSeguro
from NaiveBayes import NaiveBayes
from PCA_Bayes import PCA

CLASES = ["Circulo", "Poligonos", "Fondo"]


def cargar_dataset_desde_carpetas(ruta_base, extractor=None):
    """
    Espera la estructura:
        ruta_base/
            Circulo/    *.jpg / *.png
            Poligonos/  *.jpg / *.png
            Fondo/      *.jpg / *.png

    Extrae el vector de características real de cada imagen. Si una imagen
    provoca un error o produce un vector de tamaño distinto al esperado,
    se descarta y se avisa por consola, sin romper el entrenamiento.
    """
    if extractor is None:
        extractor = ExtractorSeguro()

    X_bruto, y_bruto = [], []
    descartadas = 0

    for clase in CLASES:
        ruta_clase = os.path.join(ruta_base, clase)
        if not os.path.isdir(ruta_clase):
            print(f"Aviso: no existe la carpeta '{ruta_clase}', se omite esa clase.")
            continue

        for nombre in os.listdir(ruta_clase):
            if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            ruta_img = os.path.join(ruta_clase, nombre)
            img = cv2.imread(ruta_img)
            if img is None:
                print(f"  [omitida] no se pudo leer: {ruta_img}")
                descartadas += 1
                continue

            try:
                features = extractor.extract(img)
            except Exception as e:
                print(f"  [omitida] error extrayendo features de {ruta_img}: {e}")
                descartadas += 1
                continue

            if features is None:
                print(f"  [omitida] extractor devolvió None para {ruta_img}")
                descartadas += 1
                continue

            X_bruto.append(np.asarray(features, dtype=float))
            y_bruto.append(clase)

    if not X_bruto:
        raise RuntimeError("No se pudo extraer ninguna característica válida del dataset.")

    largos = Counter(len(v) for v in X_bruto)
    largo_esperado = largos.most_common(1)[0][0]

    X, y = [], []
    for vec, clase in zip(X_bruto, y_bruto):
        if len(vec) == largo_esperado:
            X.append(vec)
            y.append(clase)
        else:
            descartadas += 1

    print(f"\nVectores usados: {len(X)} (largo={largo_esperado}) | "
          f"Descartados por inconsistencia/errores: {descartadas}")

    return np.array(X, dtype=float), np.array(y)


def train_test_split_manual(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    n_test = int(n * test_size)
    idx_test = indices[:n_test]
    idx_train = indices[n_test:]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]


def k_fold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    particiones = []
    for i in range(k):
        idx_test = folds[i]
        idx_train = np.concatenate([folds[j] for j in range(k) if j != i])
        particiones.append((idx_train, idx_test))
    return particiones


def normalizar(X_train, X_test):
    medias = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1e-9
    return (X_train - medias) / stds, (X_test - medias) / stds


def matriz_confusion(y_real, y_pred, clases):
    idx = {c: i for i, c in enumerate(clases)}
    M = np.zeros((len(clases), len(clases)), dtype=int)
    for r, p in zip(y_real, y_pred):
        M[idx[r], idx[p]] += 1
    return M


def imprimir_matriz_confusion(M, clases):
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print("            " + "  ".join(f"{c:>10}" for c in clases))
    for i, c in enumerate(clases):
        print(f"{c:>10}  " + "  ".join(f"{v:>10}" for v in M[i]))


def precision_recall_f1(M, clases):
    resultados = {}
    for i, c in enumerate(clases):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        resultados[c] = {"precision": precision, "recall": recall, "f1": f1}
    return resultados


def imprimir_metricas(resultados):
    print("\nMétricas por clase:")
    print(f"{'Clase':>12} {'Precisión':>12} {'Recall':>10} {'F1':>10}")
    for c, m in resultados.items():
        print(f"{c:>12} {m['precision']:>12.2%} {m['recall']:>10.2%} {m['f1']:>10.2%}")


def preprocesar(X_train, X_test, usar_pca=False, varianza_objetivo=0.95):
    """Normaliza y, si usar_pca=True, aplica PCA (ajustado solo con train)."""
    X_train, X_test = normalizar(X_train, X_test)
    pca = None
    if usar_pca:
        pca = PCA(explained_variance_ratio=varianza_objetivo)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    return X_train, X_test, pca


def evaluar_holdout(X, y, test_size=0.2, usar_pca=False, varianza_objetivo=0.95):
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=test_size)
    X_train, X_test, pca = preprocesar(X_train, X_test, usar_pca, varianza_objetivo)

    modelo = NaiveBayes()

    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    t_entrenamiento = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = modelo.predict(X_test)
    t_inferencia_total = time.perf_counter() - t0
    t_inferencia_promedio = t_inferencia_total / len(X_test)

    clases = sorted(set(y))
    M = matriz_confusion(y_test, y_pred, clases)
    metricas = precision_recall_f1(M, clases)
    exactitud = np.trace(M) / M.sum()

    etiqueta_pca = f"CON PCA ({pca.n_components_} componentes)" if usar_pca else "SIN PCA"
    print(f"\n=== Naive Bayes {etiqueta_pca} — Hold-out (test_size={test_size}) ===")
    print(f"Dimensionalidad usada: {X_train.shape[1]}")
    print(f"Muestras train: {len(X_train)} | Muestras test: {len(X_test)}")
    print(f"Tiempo de entrenamiento: {t_entrenamiento*1000:.3f} ms")
    print(f"Tiempo de inferencia total: {t_inferencia_total*1000:.3f} ms "
          f"({t_inferencia_promedio*1000:.4f} ms por muestra)")
    print(f"Exactitud global: {exactitud:.2%}")

    imprimir_matriz_confusion(M, clases)
    imprimir_metricas(metricas)

    return exactitud, metricas, t_entrenamiento, t_inferencia_total


def evaluar_cross_validation(X, y, k=5, usar_pca=False, varianza_objetivo=0.95):
    particiones = k_fold_indices(len(X), k=k)
    exactitudes, t_ent_list, t_inf_list, n_comp_list = [], [], [], []

    etiqueta_pca = "CON PCA" if usar_pca else "SIN PCA"
    print(f"\n=== Naive Bayes {etiqueta_pca} — Validación cruzada (k={k}) ===")
    for i, (idx_train, idx_test) in enumerate(particiones, start=1):
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        X_train, X_test, pca = preprocesar(X_train, X_test, usar_pca, varianza_objetivo)
        if pca is not None:
            n_comp_list.append(pca.n_components_)

        modelo = NaiveBayes()

        t0 = time.perf_counter()
        modelo.fit(X_train, y_train)
        t_ent = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = modelo.predict(X_test)
        t_inf = time.perf_counter() - t0

        clases = sorted(set(y))
        M = matriz_confusion(y_test, y_pred, clases)
        exactitud = np.trace(M) / M.sum()

        exactitudes.append(exactitud)
        t_ent_list.append(t_ent)
        t_inf_list.append(t_inf)

        info_comp = f" | componentes = {pca.n_components_}" if pca is not None else ""
        print(f"  Fold {i}: exactitud = {exactitud:.2%}{info_comp} | "
              f"entrenamiento = {t_ent*1000:.3f} ms | inferencia = {t_inf*1000:.3f} ms")

    print(f"\nExactitud promedio: {np.mean(exactitudes):.2%} (+/- {np.std(exactitudes):.2%})")
    if n_comp_list:
        print(f"Componentes promedio usados: {np.mean(n_comp_list):.1f}")
    print(f"Tiempo de entrenamiento promedio: {np.mean(t_ent_list)*1000:.3f} ms")
    print(f"Tiempo de inferencia promedio: {np.mean(t_inf_list)*1000:.3f} ms")

    return exactitudes


def entrenar_y_evaluar(ruta_dataset, test_size=0.2, k=5, usar_pca=False, varianza_objetivo=0.95):
    extractor = ExtractorSeguro()
    X, y = cargar_dataset_desde_carpetas(ruta_dataset, extractor)

    print(f"\nDataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases presentes: {sorted(set(y))}")

    evaluar_holdout(X, y, test_size=test_size, usar_pca=usar_pca, varianza_objetivo=varianza_objetivo)
    evaluar_cross_validation(X, y, k=k, usar_pca=usar_pca, varianza_objetivo=varianza_objetivo)

    # Modelo final entrenado con todo el dataset, listo para usar en localización
    medias = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1e-9
    X_norm = (X - medias) / stds

    pca_final = None
    X_final = X_norm
    if usar_pca:
        pca_final = PCA(explained_variance_ratio=varianza_objetivo)
        X_final = pca_final.fit_transform(X_norm)

    modelo_final = NaiveBayes()
    modelo_final.fit(X_final, y)

    norm_stats = {"medias": medias, "stds": stds}
    return modelo_final, extractor, norm_stats, pca_final


if __name__ == "__main__":
    print("\n\n########## SIN PCA ##########")
    entrenar_y_evaluar(ruta_dataset="dataset_imagenes", usar_pca=False)

    print("\n\n########## CON PCA ##########")
    entrenar_y_evaluar(ruta_dataset="dataset_imagenes", usar_pca=True, varianza_objetivo=0.95)
