"""
Main de prueba — usa solo los 6 archivos mínimos:
    Extraccion_caracteristicas.py
    Extractor_Seguro_Bayes.py
    NaiveBayes.py
    PCA_Bayes.py
    Entrenar_Naive_Bayes.py
    Localizacion_Bayes.py

Entrena el modelo con y sin PCA, y expone dos funciones que reciben la
ruta de la imagen a evaluar y corren la detección por ventana deslizante.
"""

from Entrenar_Naive_Bayes import cargar_y_describir_dataset, entrenar_modelo_final
from Localizacion_Bayes import ejecutar_deteccion


# --- Ajusta esta ruta a tu proyecto ---
RUTA_DATASET = r"dataset\Entrenamiento"


def probar_sin_pca(ruta_imagen, modelo, extractor, norm_stats,
                    window_size=(100, 100), step=20, confidence_threshold=0.6):
    """
    Corre la detección por ventana deslizante sobre ruta_imagen usando el
    modelo entrenado SIN PCA. Devuelve la lista de detecciones (clase, (x, y)).
    """
    print("\n" + "=" * 60)
    print(f"Detección SIN PCA — imagen: {ruta_imagen}")
    print("=" * 60)

    detecciones = ejecutar_deteccion(
        ruta_imagen, modelo, extractor,
        norm_stats=norm_stats, pca=None,
        window_size=window_size, step=step, confidence_threshold=confidence_threshold
    )

    print(f"SIN PCA -> {len(detecciones)} objetos detectados: {detecciones}")
    return detecciones


def probar_con_pca(ruta_imagen, modelo, extractor, norm_stats, pca,
                    window_size=(100, 100), step=20, confidence_threshold=0.6):
    """
    Corre la detección por ventana deslizante sobre ruta_imagen usando el
    modelo entrenado CON PCA. Devuelve la lista de detecciones (clase, (x, y)).
    """
    print("\n" + "=" * 60)
    print(f"Detección CON PCA — imagen: {ruta_imagen}")
    print("=" * 60)

    detecciones = ejecutar_deteccion(
        ruta_imagen, modelo, extractor,
        norm_stats=norm_stats, pca=pca,
        window_size=window_size, step=step, confidence_threshold=confidence_threshold
    )

    print(f"CON PCA -> {len(detecciones)} objetos detectados: {detecciones}")
    return detecciones


def main():
    ruta_imagen = r"dataset\prueba1\ala.jpg"  # <-- ajusta la ruta de la imagen a probar

    # 1. Cargar el dataset UNA sola vez (se reutiliza para ambos modelos)
    X, y, extractor = cargar_y_describir_dataset(RUTA_DATASET)

    # 2. Entrenar SIN PCA
    modelo_sin_pca, extractor, norm_stats_sin_pca, _ = entrenar_modelo_final(
        X, y, extractor, usar_pca=False
    )

    # 3. Entrenar CON PCA
    modelo_con_pca, extractor, norm_stats_con_pca, pca = entrenar_modelo_final(
        X, y, extractor, usar_pca=True, varianza_objetivo=0.95
    )

    # 4. Probar ambos modelos sobre la misma imagen (o rutas distintas si quieres)
    probar_sin_pca(ruta_imagen, modelo_sin_pca, extractor, norm_stats_sin_pca)
    probar_con_pca(ruta_imagen, modelo_con_pca, extractor, norm_stats_con_pca, pca)


if __name__ == "__main__":
    main()
