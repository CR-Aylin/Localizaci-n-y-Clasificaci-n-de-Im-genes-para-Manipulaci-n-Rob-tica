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
""" 
#Modelo Naive Bayes
import src.model.Entrenar_Naive_Bayes as enb
import src.model.Localizacion_Bayes as lb

"""

def Ejemplo(ROBOTS, offsetx, offsety):
    
    Npixel_x = 169
    Npixel_y = 381

    X,Y = c.pixeles_a_mm(Npixel_x, Npixel_y, offsetx, offsety)
    print(X,Y)
    ROBOTS.move_to(X, Y,50,0, wait=True)

if __name__ == "__main__":
    #Arch.crear_archivoEtiquetas() #esta guardado no es necesario
    """ #descomentar con robot 
    ROBOTS = dm.conectar() #inicia
    dm.Home(ROBOTS)

    cam = cama.Cam()
    resultados = cam.sacar_foto("Prueba1")

    #print(resultados)
    
    pixel_x, pixel_y, = resultados["referencia"]
    pose = ROBOTS.pose()
    robot_x, robot_y = pose[0] , pose[1]

    coor , offsetx , offsety  = c.coordernadas_n(pixel_x, pixel_y, robot_x, robot_y)
    pose = ROBOTS.pose()

    print(f"Coordenadas Calculadas = {coor}")
    print(f"Coordenadas Actual = {pose} ")
"""
    #Aqui Colocar Algoritmos una vez funcionen
    modelo = int(input("Seleccione Modelo: "))
    redu = bool(int(input("Reducción Dimensional (1/0): "))) #boleano
    
    match (modelo, redu):
        case (1, True):
            print("KNN - PCA")
        case (1, False):
            print("KNN")
        case (2, True):
            print("NaivesBayes - PCA")
        case (2, False):
            print("NaivesBayes")
        case (3, True):
            print("SVM - PCA")
            resultados = svm.ejecutar_deteccion_PCA()
            Cuad,Cir = resultados
            coorCir = [Cir['x'], Cir['y']]
            coorCuad = [Cuad['x'], Cuad['y']]
        case (3, False):
            print("SVM")
            resultados = svm.ejecutar_deteccion()
            Cuad,Cir = resultados
            coorCir = [Cir['x'], Cir['y']]
            coorCuad = [Cuad['x'], Cuad['y']]

        case _:
            print("Combinación no válida")

print(f"Coordenadas Circulo = {coorCir}")
print(f"Coordenadas Cuadrado = {coorCuad}")
#una vez identificado el objeto , mover robot , coordenaa1 == objetos a mover ; coordenada 2 == donde dejar el objeto
##dm.mover_robot(ROBOTS,Cordenaa1,Cordenaa2)
""" 
objeto_reconocido = None
    posicion_pixeles = None
 
    if modelo == 2:  # Naive Bayes (unico algoritmo integrado en este flujo por ahora)
        ruta_modelo_bayes = "modelo_bayes.pkl"
 
        # Si ya existe un modelo entrenado guardado, se reutiliza en vez de volver a entrenar cada vez que se ejecuta el robot
        if os.path.exists(ruta_modelo_bayes):
            with open(ruta_modelo_bayes, "rb") as f:
                modelo_bayes, extractor_bayes, norm_stats_bayes, pca_bayes = pickle.load(f)
        else:
            modelo_bayes, extractor_bayes, norm_stats_bayes, pca_bayes = enb.entrenar_y_evaluar(
                ruta_dataset="dataset_imagenes", usar_pca=redu
            )
            with open(ruta_modelo_bayes, "wb") as f:
                pickle.dump((modelo_bayes, extractor_bayes, norm_stats_bayes, pca_bayes), f)
 
        # Imagen completa del tablero recien capturada por la camara
        imagen_tablero = resultados.get("imagen")
        if imagen_tablero is None:
            ruta_imagen = resultados.get("ruta", "Prueba1.jpg")
            imagen_tablero = cv2.imread(ruta_imagen)
 
        detecciones = lb.sliding_window_localization_bayes(
            imagen_tablero, modelo_bayes, extractor_bayes,
            window_size=(100, 100), step=20, confidence_threshold=0.6,
            norm_stats=norm_stats_bayes, pca=pca_bayes if redu else None
        )
 
        if detecciones:
            mejor_deteccion = max(detecciones, key=lambda d: d['confidence'])
            objeto_reconocido = mejor_deteccion['class']
            posicion_pixeles = (mejor_deteccion['x'], mejor_deteccion['y'])
            print(f"Objeto reconocido: {objeto_reconocido} en pixel {posicion_pixeles} "
                  f"(confianza {mejor_deteccion['confidence']:.2%})")
        else:
            print("No se detecto ningun objeto en el tablero.")
 
    elif modelo in (1, 3):
        print("Ese modelo aun no esta integrado en este flujo (solo Naive Bayes por ahora).")
"""    
