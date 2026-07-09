import os
import cv2
import pandas as pd
import Extraccion_caracteristicas as ec

carpetas = [
    r"\\dataset\\Entrenamiento\\Class_1",
    r"\\dataset\\Entrenamiento\\Class_2",
    r"\\dataset\\Entrenamiento\\Class_3"
]

Clases = ["Circulo", "Cuadrados", "Fondo"]


def crear_archivoEtiquetas():

    datos = []
    columnas = ["imagen"]

    # Características geométricas
    columnas += [
        "area",
        "perimetro",
        "circularidad",
        "compacidad",
        "aspect_ratio"
    ]

    # Características estadísticas
    columnas += [
        "media_intensidad",
        "std_intensidad",
        "entropia"
    ]

    # Momentos de Hu
    columnas += [f"hu_{i}" for i in range(1, 8)]

    # Histograma H
    columnas += [f"H_{i}" for i in range(16)]

    # Histograma S
    columnas += [f"S_{i}" for i in range(16)]

    # Histograma V
    columnas += [f"V_{i}" for i in range(16)]

    # Bounding Box
    columnas += [
        "x",
        "y",
        "ancho",
        "alto"
    ]

    # Etiqueta
    columnas += ["clase"]

    for i, carpeta in enumerate(carpetas):
        clase = Clases[i]
        print(f"\nProcesando carpeta: {carpeta}")
        if os.path.exists(carpeta):
            for carpeta_actual, _, archivos in os.walk(carpeta):
                for archivo in archivos:
                    ruta_completa = os.path.join(carpeta_actual, archivo)
                    print(f"Procesando: {ruta_completa}")
                    vector_caracteristicas = ec.proceso(ruta_completa)
                    datos.append(
                        [ruta_completa] +
                        list(vector_caracteristicas) +
                        [clase]
                    )

        else:
            print(f"La ruta no existe: {carpeta}")

    df = pd.DataFrame(datos, columns=columnas)

    # Guardar CSV
    df.to_csv(
        "dataset/dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Dataset guardado correctamente.")
    print(df.head())

    """ Se ve asi el vector de características final:
    VECTOR = np.concatenate([
    geom_features,
    stat_features,
    visual_features,
    medidas])
    
    """