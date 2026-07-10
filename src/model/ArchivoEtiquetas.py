import os
import cv2
import pandas as pd
import Extraccion_caracteristicas as ec

carpetas = [
    os.path.join("dataset", "Entrenamiento", "Class_1"),
    os.path.join("dataset", "Entrenamiento", "Class_2"),
    os.path.join("dataset", "Entrenamiento", "Class_3")
]

Clases = ["Cuadrados","Circulo", "Fondo"]


def crear_archivoEtiquetas():

    datos = []

    columnas = [
        "imagen",

        # Características geométricas
        "area",
        "perimetro",
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
        *[f"H_{i}" for i in range(16)],
        *[f"S_{i}" for i in range(16)],
        *[f"V_{i}" for i in range(16)],

        # Bounding Box
        "x",
        "y",
        "ancho",
        "alto",

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
                    
                    extractor = ec.features_extractor(hist_bins=16)
                    vector_caracteristicas = extractor.extract(image)


                    if vector_caracteristicas is None:
                        print("No se obtuvieron características")
                        continue


                    print("Características obtenidas:",len(vector_caracteristicas))


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


crear_archivoEtiquetas()