import os
import cv2
import numpy as np
import pandas as pd
import numpy as np

import Extraccion_caracteristicas as ec

# Carpetas de entrenamiento
carpetas = [
    os.path.join("dataset", "Entrenamiento", "Class_1"),
    os.path.join("dataset", "Entrenamiento", "Class_2"),
    os.path.join("dataset", "Entrenamiento", "Class_3"),
]


def ventana_deslizante(imagen, tamano_ventana=(100, 100), paso=20):
    """Genera únicamente los recortes de la ventana deslizante."""
    alto, ancho = imagen.shape[:2]
    win_w, win_h = tamano_ventana

    for y in range(0, alto - win_h + 1, paso):
        for x in range(0, ancho - win_w + 1, paso):
            yield imagen[y:y + win_h, x:x + win_w]

def sliding_window(arr, k):
    n = len(arr)

    if k > n or k <= 0:
        return []

    ventanas = []

    for i in range(n - k + 1):
        ventanas.append(arr[i:i + k])

    return ventanas

def aplanar(img):
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


def etiquetar_ventana_por_color(ventana):
    """
    Detecta si la ventana contiene un cuadrado azul o un círculo naranja/rojo.
    """
    hsv = cv2.cvtColor(ventana, cv2.COLOR_BGR2HSV)

<<<<<<< Updated upstream
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

=======
    # Cuadrado azul
    mascara_azul = cv2.inRange(
        hsv,
        np.array([90, 50, 50]),
        np.array([130, 255, 255])
    )
>>>>>>> Stashed changes

    # Círculo naranja/rojo
    mascara_naranja = cv2.inRange(
        hsv,
        np.array([0, 100, 100]),
        np.array([25, 255, 255])
    )

    pixeles_azul = cv2.countNonZero(mascara_azul)
    pixeles_naranja = cv2.countNonZero(mascara_naranja)
    umbral_presencia = 1000

    if pixeles_azul > umbral_presencia and pixeles_azul > pixeles_naranja:
        return "Cuadrados"

    if pixeles_naranja > umbral_presencia and pixeles_naranja > pixeles_azul:
        return "Circulo"

    return None


def generar_nombres_columnas(extractor, tamano_total_vector):
    """
    Genera las cabeceras dinámicamente. Descubre cuántos elementos corresponden 
    a la imagen aplanada restando las características fijas del tamaño total del vector.
    """
    columnas = ["clase"]
    
    # Calculamos cuántas posiciones fijas ocupan los descriptores tradicionales
    caracteristicas_fijas = extractor.get_feature_size()
    
    # La diferencia matemática son los píxeles automáticos de la imagen aplanada
    num_pixeles_img_P = tamano_total_vector - caracteristicas_fijas
    
    # 1. Píxeles automáticos del extractor
    columnas.extend([f"img_pixel_{i}" for i in range(num_pixeles_img_P)])
    
    # 2. Geométricas (3)
    columnas.extend(["geom_circularity", "geom_compactness", "geom_aspect_ratio"])
    
    # 3. Estadísticas (3)
    columnas.extend(["stat_mean_intensity", "stat_std_intensity", "stat_entropy"])
    
    # 4. Hu Moments (7)
    columnas.extend([f"hu_moment_{i+1}" for i in range(7)])
    
    # 5. Histogramas de Color (3 * hist_bins)
    columnas.extend([f"hist_h_bin_{i}" for i in range(extractor.hist_bins)])
    columnas.extend([f"hist_s_bin_{i}" for i in range(extractor.hist_bins)])
    columnas.extend([f"hist_v_bin_{i}" for i in range(extractor.hist_bins)])
    
    # 6. Componentes PCA (pca_components)
    columnas.extend([f"pca_component_{i+1}" for i in range(extractor.pca_components)])
    
    return columnas


def procesar_imagen_ventaneo(ruta_completa, extractor, paso_ventaneo=20):
    """
    Procesa cada ventana. Confía en el extractor para obtener la data visual 
    y numérica empaquetada.
    """
    img = cv2.imread(ruta_completa)

    if img is None:
        return []

    registros = []

    for ventana in ventana_deslizante(img, tamano_ventana=(100, 100), paso=paso_ventaneo):
        clase = etiquetar_ventana_por_color(ventana)

        if clase is None:
            continue

        vector_completo = np.asarray(extractor.extract(ventana)).reshape(-1)

        # PROTECCIÓN: Si el extractor no encontró contornos, devuelve un vector 
        # corto (sin la imagen aplanada). Saltamos esta ventana para no romper el DataFrame.
        if len(vector_completo) <= extractor.get_feature_size():
            continue

        # La fila ahora solo contiene la etiqueta de la clase y el vector entero de datos
        fila = [clase] + [float(val) for val in vector_completo]
        registros.append(fila)

<<<<<<< Updated upstream
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
                            list(P_imagen )+
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
=======
    return registros
>>>>>>> Stashed changes


if __name__ == "__main__":
    datos_consolidados = []
    paso_global = 20

    extractor = ec.features_extractor(debug=False)

    print("Procesando ventanas (el extractor incluye la imagen de forma automática)...")

    for carpeta in carpetas:
        print(f"\nProcesando carpeta: {carpeta}")

        if not os.path.exists(carpeta):
            print("La carpeta no existe, se omite.")
            continue

        for carpeta_actual, _, archivos in os.walk(carpeta):
            for archivo in archivos:

                if archivo.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    ruta_imagen = os.path.join(carpeta_actual, archivo)
                    print(f" -> {archivo}")

                    registros = procesar_imagen_ventaneo(
                        ruta_completa=ruta_imagen,
                        extractor=extractor,
                        paso_ventaneo=paso_global
                    )

                    datos_consolidados.extend(registros)

    print(f"\nTotal de ventanas procesadas válidas: {len(datos_consolidados)}")

    if datos_consolidados:
        # Medimos el tamaño real del vector de la primera muestra para calcular las columnas de la imagen
        tamano_vector_real = len(datos_consolidados[0]) - 1 # Restamos 1 porque la posición 0 es la 'clase'
        
        # Generamos los nombres correctos de las columnas adaptándonos al tamaño devuelto
        cabeceras = generar_nombres_columnas(extractor, tamano_vector_real)
        
        # Guardamos en el DataFrame
        df = pd.DataFrame(datos_consolidados, columns=cabeceras)

        os.makedirs("dataset", exist_ok=True)
        ruta_csv = os.path.join("dataset", "ventanas_caracteristicas.csv")

        df.to_csv(
            ruta_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nCSV guardado exitosamente con la imagen procesada por tu extractor en: {ruta_csv}")
        print(f"Forma de la matriz final guardada: {df.shape}")
        print("\nVista previa de los datos:")
        print(df.head(2))

    else:
        print("\nNo se extrajeron datos de ninguna ventana.")