import cv2
import numpy as np
import pygame
import time
import os 

L_X = 27*10
L_Y = 37,3*10
L_imgX = 1920
L_imgY = 1080



ESCALA_X = L_X/ L_imgX # mm por píxel
ESCALA_Y = L_Y/ L_imgY # mm por píxel
#Delimitar el espasion a travez de 3 punto  seleccionados con le mouse

def Punto():
    # Inicializar Pygame
    pygame.init()

    # Abrir video
    cap = cv2.VideoCapture("video.mp4")  # Cambia por tu video

    ret, frame = cap.read()
    if not ret:
        print("No se pudo abrir el video.")
        sys.exit()

    alto, ancho = frame.shape[:2]
    pantalla = pygame.display.set_mode((ancho, alto))
    pygame.display.set_caption("Selecciona 3 puntos")

    fuente = pygame.font.SysFont(None, 24)
    puntos = []
    pausado = False

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                break

        # Convertir OpenCV (BGR) -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crear superficie de Pygame
        superficie = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        pantalla.blit(superficie, (0, 0))

        # Dibujar puntos seleccionados
        for i, (x, y) in enumerate(puntos):
            pygame.draw.circle(pantalla, (255, 0, 0), (x, y), 5)
            texto = fuente.render(f"{i+1}: ({x}, {y})", True, (255, 255, 0))
            pantalla.blit(texto, (x + 10, y - 10))

        pygame.display.flip()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

            elif evento.type == pygame.KEYDOWN:
                # Espacio para pausar/reanudar
                if evento.key == pygame.K_SPACE:
                    pausado = not pausado

            elif evento.type == pygame.MOUSEBUTTONDOWN:
                if pausado and len(puntos) < 3:
                    x, y = evento.pos
                    puntos.append((x, y))
                    print(f"Punto {len(puntos)}: ({x}, {y})")

                    if len(puntos) == 3:
                        print("\nCoordenadas finales:")
                        for i, p in enumerate(puntos):
                            print(f"Punto {i+1}: {p}")

    cap.release()
    pygame.quit()

    return puntos

def recortar_imagen(y_1,y_2, x_1,x_2 ):
    camara = cv2.VideoCapture(0)
    #fotograma
    ret, imagen = camara.read()

    if ret:
        # Guardar la imagen
        cv2.imwrite("captura.jpg", imagen)
        print("Imagen guardada como captura.jpg")
    else:
        print("No se pudo capturar la imagen.")

    recorte_img = imagen[y_1:y_2, x_1:x_2]
    
    cv2.imshow("Original Image", imagen)
    cv2.imshow("Cropped Image", recorte_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    camara.release()

def distancia():#calcular la diferencia para ajustar 

def pixeles_a_mm(pix_x, pix_y, offsetx = 0, offsety = 0):
    # Convertimos los pixeles de la cámara a coordenadas X e Y del robot
    mm_x = pix_x * ESCALA_X + offsetx
    mm_y = pix_y * ESCALA_Y + offsety
    return mm_x, mm_y

def Calibracion():
