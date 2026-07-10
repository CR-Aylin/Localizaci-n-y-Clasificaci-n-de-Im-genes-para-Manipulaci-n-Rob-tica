import cv2
import time
import numpy as np
from pydobot import Dobot
import serial.tools.list_ports as list_ports 
import time # time.sleep(3)  # Espera 3 segundos
import os


import src.model.ArchivoEtiquetas as Arch
import src.robot.Calibracion as c
import src.robot.camara as cama
import src.model.Extraccion_caracteristicas as ec
import src.robot.dobot_movement as dm

#modelos
import src.model.KNN as knn

#import src.model.SVM as svm
import src.model.Test_svm as svm 
import src.model.Entrenar_Naive_Bayes as nb

import src.model.Test_knn as tk

ruta = r"dataset\prueba1\ala.jpg"


def obtener_centro(datos_clase):
    (x1, y1), (x2, y2) = datos_clase['coordenadas_cuadrado']
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def Ejemplo(ROBOTS, offsetx, offsety):
    
    Npixel_x = 169
    Npixel_y = 381

    X,Y = c.pixeles_a_mm(Npixel_x, Npixel_y, offsetx, offsety)
    print(X,Y)
    ROBOTS.move_to(X, Y,50,0, wait=True)

if __name__ == "__main__":
    #Arch.crear_archivoEtiquetas() #esta guardado no es necesario
    #descomentar con robot 
    ROBOTS = dm.conectar() #inicia
    dm.Home(ROBOTS)
    
    cam = cama.Cam()
    resultados = cam.sacar_foto("Prueba1")

    #print(resultados)
    
    pixel_x, pixel_y, = resultados["referencia"]
    pose = ROBOTS.pose()
    robot_x, robot_y = pose[0] , pose[1]
#hello
    coor , offsetx , offsety  = c.coordernadas_n(pixel_x, pixel_y, robot_x, robot_y)
    pose = ROBOTS.pose()

    print(f"Coordenadas Calculadas = {coor}")
    print(f"Coordenadas Actual = {pose} ")

    #Aqui Colocar Algoritmos una vez funcionen
    modelo = int(input("Seleccione Modelo: "))
    redu = bool(int(input("Reducción Dimensional (1/0): "))) #boleano 

    ruta = resultados['ruta'] #'ruta': 'dataset\\Pruebas\\Prueba1.png'
    ruta_dataset=r"dataset/Entrenamiento"

    match (modelo, redu):
        case (1, True):
            print("KNN - PCA")
            re = tk.ejecutar_knn_con_pca(ruta)

            Cuad = re[np.str_('Class_2')]
            Cir = re[np.str_('Class_1')]

            coorCuad = obtener_centro(Cuad)  
            coorCir = obtener_centro(Cir)    
            
        case (1, False):
            print("KNN")
            resultados = tk.ejecutar_knn_sin_pca(ruta)
            
            Cuad = resultados[np.str_('Class_2')]
            Cir = resultados[np.str_('Class_1')]

            coorCuad = obtener_centro(Cuad)  
            coorCir = obtener_centro(Cir)    

        case (2, True):
            print("NaivesBayes - PCA")
            X, y, extractor = nb.cargar_y_describir_dataset(ruta_dataset="dataset/Entrenamiento")
            nb.evaluar_modelo(
                X,
                y,
                usar_pca=True,
                varianza_objetivo=0.95
            )

            modelo_pca, extractor, norm_stats_pca, pca_modelo = nb.entrenar_modelo_final(
                X,
                y,
                extractor,
                usar_pca=True,
                varianza_objetivo=0.95
            )
            resultados =nb.evaluar_imagen(
                ruta,
                modelo_pca,
                extractor,
                norm_stats_pca,
                pca=pca_modelo,
                mostrar=True,
                ruta_salida="salida/resultado_con_pca.jpg"
            )
            #print(resultados)

            Cuad,Cir = resultados
            coorCir = Cir["centro"]
            coorCuad = Cuad['centro']
        case (2, False):
            print("NaivesBayes")
            X, y, extractor = nb.cargar_y_describir_dataset(ruta_dataset="dataset/Entrenamiento")
            nb.evaluar_modelo(X, y, usar_pca=False)

            modelo, extractor, norm_stats, pca = nb.entrenar_modelo_final(
                X,
                y,
                extractor,
                usar_pca=False
            )
            resultados =nb.evaluar_imagen(
                ruta,
                modelo,
                extractor,
                norm_stats,
                pca=None,
                mostrar=True,
                ruta_salida="salida/resultado_sin_pca.jpg"
            )
            
            coorCir = Cir["centro"]
            coorCuad = Cuad['centro']
            
        case (3, True):
            print("SVM - PCA")
            resultados = svm.ejecutar_deteccion_PCA(
                                ruta_dataset=r"dataset\Entrenamiento",
                                ruta_imagen_tablero=ruta,
                                confidence_threshold=0.65,
                                n_components=50
                            )
            
            coorCir = [Cir['x'], Cir['y']]
            coorCuad = [Cuad['x'], Cuad['y']]
        case (3, False):
            print("SVM")
            resultados = svm.ejecutar_deteccion(
                            ruta_dataset=r"dataset\Entrenamiento",
                            ruta_imagen_tablero=ruta,
                            confidence_threshold=0.65
                        )
            Cuad,Cir = resultados
            coorCir = [Cir['x'], Cir['y']]
            coorCuad = [Cuad['x'], Cuad['y']]

        case _:
            print("Combinación no válida")

print(f"Coordenadas Circulo = {coorCir}")
print(f"Coordenadas Cuadrado = {coorCuad}")

#una vez identificado el objeto , mover robot , coordenaa1 == objetos a mover ; coordenada 2 == donde dejar el objeto
#dm.mover_robot(ROBOTS,Cordenaa1,Cordenaa2) 135, 112 #posiciones en espejo

ROBOTS.move_to(135, 112,50,50, wait=True)
print("FFFF")