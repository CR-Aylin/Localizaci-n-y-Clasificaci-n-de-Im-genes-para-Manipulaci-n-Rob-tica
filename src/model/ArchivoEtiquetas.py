import os
import cv2
import pandas as pd

import Extraccion_caracteristicas as ec


carpetas = [
    os.path.join("dataset", "Entrenamiento", "Class_1"),
    os.path.join("dataset", "Entrenamiento", "Class_2"),
    os.path.join("dataset", "Entrenamiento", "Class_3")
]

Clases = ["Cuadrados", "Circulo", "Fondo"]


def crear_archivoEtiquetas(hist_bins=16, chroma_threshold=12):

    datos = []

    columnas = [
        "imagen",

        # Características geométricas (solo invariantes a escala; se quitaron
        # area, perimetro, x, y, ancho, alto porque no son invariantes a escala
        # y confundían al clasificador entre fotos de entrenamiento y ventanas
        # de test de tamaño fijo)
        "circularidad",
        "compacidad",
        "aspect_ratio",

        # Características estadísticas
        "media_intensidad",
        "std_intensidad",
        "entropia",

        # Momentos de Hu
        *[f"hu_{i}" for i in range(1, 8)],

        # Histogramas HSV
        *[f"H_{i}" for i in range(hist_bins)],
        *[f"S_{i}" for i in range(hist_bins)],
        *[f"V_{i}" for i in range(hist_bins)],

        # Clase
        "clase"
    ]

    # Se crea UNA sola vez fuera del loop (antes se creaba por cada imagen,
    # lo cual era innecesario ya que el extractor no guarda estado entre llamadas
    # salvo su configuración).
    extractor = ec.features_extractor(
        hist_bins=hist_bins,
        chroma_threshold=chroma_threshold
    )

    for i, carpeta in enumerate(carpetas):

        clase = Clases[i]

        print("\nProcesando carpeta:", carpeta)

        if not os.path.exists(carpeta):
            print(" La ruta no existe")
            continue

        for carpeta_actual, _, archivos in os.walk(carpeta):

            for archivo in archivos:

                # Solo procesar imágenes
                if archivo.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):

                    ruta_completa = os.path.join(carpeta_actual, archivo)

                    print("Procesando:", ruta_completa)

                    image = cv2.imread(ruta_completa)

                    # Validar que la imagen se haya podido leer antes de procesarla
                    if image is None:
                        print("  ⚠ No se pudo leer la imagen, se omite")
                        continue

                    vector_caracteristicas = extractor.extract(image)

                    if vector_caracteristicas is None:
                        print("No se obtuvieron características")
                        continue

                    print("Características obtenidas:", len(vector_caracteristicas))

                    datos.append(
                        [ruta_completa] +
                        list(vector_caracteristicas) +
                        [clase]
                    )

    print("\nTotal imágenes procesadas:", len(datos))

    if len(datos) == 0:
        print(" No hay datos para crear el CSV")
        return

    # Crear DataFrame
    df = pd.DataFrame(datos, columns=columnas)

    print("\nColumnas:", len(columnas))
    print("Datos:", df.shape)

    os.makedirs("dataset", exist_ok=True)

    df.to_csv(
        os.path.join("dataset", "dataset.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset guardado correctamente")
    print(df.head())


if __name__ == "__main__":
    crear_archivoEtiquetas()