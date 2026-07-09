import cv2
import numpy as np
from pydobot import Dobot
import serial.tools.list_ports as list_ports  # Cambio importante
import time
import os

import Calibracion
import Extraccion_caracteristicas as ec
import dobot_movement as dm

ROBOTS =dm.conectar()
dm.Home(ROBOTS)
puntos, ruta = Calibracion.Calibracion()
Vector = ec.proceso(ruta)

#Aqui Colocar Algoritmos una vez funcionen

#una vez identificado el objeto , mover robot
dm.mover_robot(ROBOTS,Cordenaa1, Cordenaa2)



