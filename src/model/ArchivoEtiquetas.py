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

def sliding_window(arr, k):
    n = len(arr)

    if k > n or k <= 0:
        return []

    ventanas = []

    for i in range(n - k + 1):
        ventanas.append(arr[i:i + k])

    return ventanas

def aplanar(self, img):
    """
    Recibe una imagen leída con cv2.imread() y devuelve un vector
    unidimensional con el promedio de los canales R y G.
    """

    # Convertir de BGR a RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Separar canales
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)

    # Promedio de R y G y B
    promedio = (R + G + B) / 3.0

    # Convertir a vector unidimensional
    return promedio.flatten()


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
                    #
                    P_imagen = aplanar(image)

                    ventanas = sliding_window(image, 64)
                    
                    for ventana  in ventanas:
                        
                        extractor = ec.features_extractor(
                            hist_bins=hist_bins,
                            chroma_threshold=chroma_threshold
                        )
                        vector_caracteristicas = extractor.extract(ventana)

                        if vector_caracteristicas is None:
                            print("No se obtuvieron características")
                            continue

                        print("Características obtenidas:", len(vector_caracteristicas))

                        datos.append(
                            [P_imagen] +
                            list(vector_caracteristicas) +
                            [clase]
                        )

                    # Validar que la imagen se haya podido leer antes de procesarla
                    if image is None:
                        print("  ⚠ No se pudo leer la imagen, se omite")
                        continue



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