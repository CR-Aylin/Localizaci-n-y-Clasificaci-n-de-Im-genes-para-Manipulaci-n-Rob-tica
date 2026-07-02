import os
import cv2
import pandas as pd

def crear_archivoEtiquetas(ruta ,imagenes):

    datos = []
    Clases = ["Circulo","Poligonos"]
    
    for clase in Clases:
        for img in os.listdir(ruta):
            if img.endswith(".jpg") or img.endswith(".png"):
                ruta_img = os.path.join(ruta, img) 
                img = cv2.imread(ruta_img)
                #Agregar Caracteristicas de la imagen para guardarlas en el archivo de etiquetas
                #Escala de grisees y tamaño ?

                datos.append([ruta_img, 0, 0, clase]) #guardamos todo 
#guardar CSV
    df = pd.DataFrame(datos, columns=['imagen', 'x', 'y', 'clase'])#nombre de las caracteriasticas
    df.to_csv("dataset\\dataset.csv", index=False , encoding='utf-8-sig') #guardar el archivo CSV en la ruta especificada  
    print(f"características guardadas: {len(datos)} imágenes")
