import numpy as np

from Entrenar_NaiveBayes import cargar_dataset_desde_carpetas
from Extractor_Seguro_Bayes import ExtractorSeguro


def nombres_caracteristicas(hist_bins=16):
    """
    Nombres legibles para cada posición del vector, según el orden en que la
    Extraccion_caracteristicas.py arma el vector final:
    geom(5) + stat(3) + hu(7) + color(3*hist_bins) + medidas(4)
    """
    nombres = ["area", "perimetro", "circularidad", "compacidad", "aspect_ratio"]
    nombres += ["media_intensidad", "std_intensidad", "entropia"]
    nombres += [f"hu_{i+1}" for i in range(7)]
    nombres += [f"hist_h_{i}" for i in range(hist_bins)]
    nombres += [f"hist_s_{i}" for i in range(hist_bins)]
    nombres += [f"hist_v_{i}" for i in range(hist_bins)]
    nombres += ["x", "y", "ancho", "alto"]
    return nombres


def poder_discriminante(X, y):
    """
    Calcula, para cada característica, una razón de tipo Fisher:
        (varianza ENTRE clases) / (varianza PROMEDIO DENTRO de cada clase)
    Un valor alto significa que las clases están bien separadas en esa
    característica (por lo tanto sirve para diferenciar clases). Un valor
    cercano a 0 significa que todas las clases se superponen en esa
    característica y aporta poco al clasificador.
    """
    clases = np.unique(y)
    n_features = X.shape[1]
    media_global = X.mean(axis=0)

    varianza_entre = np.zeros(n_features)
    varianza_dentro = np.zeros(n_features)

    for c in clases:
        X_c = X[y == c]
        n_c = len(X_c)
        media_c = X_c.mean(axis=0)

        varianza_entre += n_c * (media_c - media_global) ** 2
        varianza_dentro += n_c * X_c.var(axis=0)

    varianza_entre /= len(X)
    varianza_dentro /= len(X)
    varianza_dentro[varianza_dentro == 0] = 1e-9

    return varianza_entre / varianza_dentro


def analizar_discriminancia(ruta_dataset, top_n=15, hist_bins=16):
    extractor = ExtractorSeguro(hist_bins=hist_bins)
    X, y = cargar_dataset_desde_carpetas(ruta_dataset, extractor)

    puntajes = poder_discriminante(X, y)
    nombres = nombres_caracteristicas(hist_bins=hist_bins)

    # Si el largo real del vector no coincide con lo esperado (por ejemplo,
    # el vector quedó truncado o extendido), se recorta al mínimo común
    n = min(len(nombres), len(puntajes))
    nombres, puntajes = nombres[:n], puntajes[:n]

    orden = np.argsort(puntajes)[::-1]

    print(f"\n=== Top {top_n} características más discriminantes ===")
    print(f"{'Característica':>20} {'Puntaje (Fisher)':>18}")
    for i in orden[:top_n]:
        print(f"{nombres[i]:>20} {puntajes[i]:>18.4f}")

    print(f"\n=== {top_n} características menos discriminantes ===")
    for i in orden[-top_n:]:
        print(f"{nombres[i]:>20} {puntajes[i]:>18.4f}")

    return list(zip(nombres, puntajes))


if __name__ == "__main__":
    analizar_discriminancia(ruta_dataset="dataset_imagenes")
