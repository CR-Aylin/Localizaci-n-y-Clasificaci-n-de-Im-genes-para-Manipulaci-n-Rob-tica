import cv2
import numpy as np
from pydobot import Dobot
import serial.tools.list_ports as list_ports  # Cambio importante
import time
import os

import src.robot.Calibracion as c
import src.robot.camara as cama
import src.model.Extraccion_caracteristicas as ec
import src.robot.dobot_movement as dm

ROBOTS = dm.conectar()

dm.Home(ROBOTS)

cam = cama.Cam()
resultados = cam.sacar_foto("Prueba1")

print(resultados)

#Aqui Colocar Algoritmos una vez funcionen

#una vez identificado el objeto , mover robot
dm.mover_robot(ROBOTS,Cordenaa1, Cordenaa2)



