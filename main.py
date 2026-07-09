import cv2
import time
import numpy as np
from pydobot import Dobot
import serial.tools.list_ports as list_ports 
import time # time.sleep(3)  # Espera 3 segundos
import os

import src.robot.Calibracion as c
import src.robot.camara as cama
import src.model.Extraccion_caracteristicas as ec
import src.robot.dobot_movement as dm

#modelos
import KNN as knn



def Ejemplo(ROBOTS):
    
    Npixel_x = 169
    Npixel_y = 381

    X,Y = c.pixeles_a_mm(Npixel_x, Npixel_y, offsetx, offsety)
    print(X,Y)
    ROBOTS.move_to(X, Y,50,0, wait=True)

if __name__ == "__main__":
        
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

    #Aqui Colocar Algoritmos una vez funcionen
    modelo = int(input("Seleccione Modelo: "))
    redu = bool(int(input("Reducción Dimensional (1/0): "))) #boleano

    match (modelo, redu):
        case (1, TRUE):
            print("KNN - PCA")
            
        case (1, False):
            print("KNN - PCA")
        case (2, TRUE):
            print("NaivesBayes - PCA")
        case (2, False):
            print("NaivesBayes - LDA")
        case (3, TRUE):
            print("SVM - PCA")
        case (3, False):
            print("SVM - LDA")
        case _:
            print("Combinación no válida")

#una vez identificado el objeto , mover robot , coordenaa1 == objetos a mover ; coordenada 2 == donde dejar el objeto
#   #dm.mover_robot(ROBOTS,Cordenaa1,Cordenaa2)



