import cv2
import numpy as np
import pygame
import time
import os 

#import camara as camara

L_X = 27*10
L_Y = 37.3*10

L_imgX = 1920
L_imgY = 1080

ESCALA_X = L_X/ L_imgX # mm por píxel
ESCALA_Y = L_Y/ L_imgY # mm por píxel

def calcular_offsets(pixel_x, pixel_y, robot_x, robot_y):

    # OffsetX:
    # offset_x = X_robot - pixel_x * ESCALA_X
    offset_x = robot_x - (pixel_x * ESCALA_X)

    # OffsetY:
    # offset_y = Y_robot - pixel_y * ESCALA_Y


    offset_y = robot_y - (pixel_y * ESCALA_Y)

    return offset_x, offset_y


def pixeles_a_mm(pix_x, pix_y, offsetx = 0, offsety = 0):
    # Convertimos los pixeles de la cámara a coordenadas X e Y del robot
    mm_x = pix_x * ESCALA_X + offsetx
    mm_y = pix_y * ESCALA_Y + offsety
    return mm_x, mm_y

def coordernadas_n(pixel_x, pixel_y, robot_x, robot_y):
    offsetx , offsety =  calcular_offsets(pixel_x, pixel_y, robot_x, robot_y)
    
    X,Y = pixeles_a_mm(pixel_x, pixel_y, offsetx, offsety)
    
    return [X,Y] , offsetx , offsety
    
"""
def Calibracion():
    cam = camara.Cam()
    #cam.configuracion_camara()
    resultado = cam.sacar_foto("calibracion")
    if puntos is None:
        print("No se pudo capturar la foto de calibración.")
        return None

    return resultado
"""

"""
resultado{
    "puntos": puntos,
    "referencia": (x_ref, y_ref),
    "ruta": ruta_completa

    }
"""