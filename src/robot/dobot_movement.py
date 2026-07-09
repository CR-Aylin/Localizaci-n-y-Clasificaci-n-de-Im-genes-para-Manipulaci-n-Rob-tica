import cv2
import numpy as np
from pydobot import Dobot
import serial.tools.list_ports as list_ports  # Cambio importante
import time

ESCALA_X = 470.0 / 1920.0  # 0.2448 mm por píxel
ESCALA_Y = 265.0 / 1080.0  # 0.2454 mm por píxel

Velocidad = 50
Aceleracion = 50

LIMITES_DOBOT = {
    'X': {'min': -120, 'max': 120},
    'Y': {'min': -120, 'max': 120},
    'Z': {'min': 0, 'max': 150},
    'R': {'min': -90, 'max': 90}
}

def calcular_centro(x, y, ancho, alto):

    centro_x = x + ancho / 2
    centro_y = y + alto / 2

    return centro_x, centro_y

def conectar():
    puertos = list(list_ports.comports())
    
    if len(puertos) == 0:
        raise Exception("No se encontró ningún puerto serie.")
    
    print("Puertos encontrados:")
    for p in puertos:
        print(f"  {p.device} - {p.description}")
    
    puerto = puertos[-1].device
    print(f"\nConectando a {puerto}...\n")
    
    robot = Dobot(port=puerto, verbose=False)
    return robot

def Home(robot):  
    HOME_X = 250
    HOME_Y = 0
    HOME_Z = 50
    HOME_R = 0
    
    robot.move_to(HOME_X, HOME_Y, HOME_Z, HOME_R, wait=True)
    pose = robot.pose()
    print(f"Posición actual: {pose}")




def pixeles_a_mm(pix_x, pix_y, offsetx = 0, offsety = 0):
    # Convertimos los pixeles de la cámara a coordenadas X e Y del robot
    mm_x = pix_x * ESCALA_X + offsetx
    mm_y = pix_y * ESCALA_Y + offsety
    return mm_x, mm_y

def mover_robot(robot,Cordenaa1, Cordenaa2):

    # Obtener posición actual
    pose = robot.pose()
    print(f"Posición actual: {pose}")

    # Abrir pinza
    robot.grip(True)
    robot.suck(False)
    time.sleep(1)

    #print(f"Coordenadas base: X={base_pos_x}, Y={base_pos_y}")


    robot.move_to(Cordenaa1[0],Cordenaa1[1],Velocidad,Aceleracion, wait=True)

    #Cerrar pinza
    robot.grip(False)
    robot.suck(False)
    time.sleep(1)
    robot.move_to(Cordenaa1[0],Cordenaa1[1],40,Velocidad,Aceleracion, wait=True)
    robot.move_to(Cordenaa2[0],Cordenaa2[1],Velocidad,Aceleracion, wait=True)
"""
if __name__ == "__main__":

    # Conectar robot
    ROBOTS = conectar()  # Usar la función conectar() en lugar de hardcodear el puerto

    # Hacer Home
    Home(ROBOTS)  

    # Obtener posición actual
    pose = ROBOTS.pose()
    print(f"Posición actual: {pose}")

    # Calcular coordenadas
    base_pos_x, base_pos_y = pixeles_a_mm(1920, 1080)
    print(f"Coordenadas base: X={base_pos_x}, Y={base_pos_y}")

    # Mover robot (descomentar cuando estés listo)
    #ROBOTS.move_to(base_pos_x, base_pos_y,Velocidad,Aceleracion, wait=True)
"""
    
