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

def Home(robot):  # Ahora recibe el robot como parámetro
    HOME_X = 200
    HOME_Y = 0
    HOME_Z = 50
    HOME_R = 0
    
    robot.move_to(HOME_X, HOME_Y, HOME_Z, HOME_R, wait=True)

def pixeles_a_mm(pix_x, pix_y, offsetx = 0, offsety = 0):
    # Convertimos los pixeles de la cámara a coordenadas X e Y del robot
    mm_x = pix_x * ESCALA_X + offsetx
    mm_y = pix_y * ESCALA_Y + offsety
    return mm_x, mm_y

#funcion de trasformacion dentro del espacio pos pixel - pos dobot

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
    ROBOTS.move_to(base_pos_x, base_pos_y,Velocidad,Aceleracion, wait=True)

#561. 939 x , y
    #objeto está en el centro de la imagen (960, 540)
    pos_real_x, pos_real_y = pixeles_a_mm(960, 540)
    print(f"Posición real: X={pos_real_x:.2f} mm, Y={pos_real_y:.2f} mm")
    
    pos_x, pos_y ,z = pixeles_a_mm(561, 939)
    print(f"Coordenadas : X= {pos_x}, Y= {pos_y}")
    #    ROBOTS.move_to(x, y,z,r, Velocidad ,Aceleracion, wait=True)
    ROBOTS.move_to(pos_x, pos_y,Velocidad ,Aceleracion, wait=True)


    # Abrir pinza
    ROBOTS.grip(True)
    time.sleep(1)

    # Cerrar pinza
    ROBOTS.grip(False)
    time.sleep(1)

    #ROBOTS.close()
    
