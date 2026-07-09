import cv2
import pygame
import numpy as np
import os

# -----------------------------
# Función para guardar la imagen
# -----------------------------
def guardar_frame(frame, nombre="foto"):

    ruta = r"dataset\Pruebas"
    os.makedirs(ruta, exist_ok=True)

    ruta_completa = os.path.join(ruta, nombre + ".png")
    cv2.imwrite(ruta_completa, frame)

    print(f"Foto guardada en: {ruta_completa}")

    return ruta_completa


# -----------------------------
# Inicializar cámara
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

ret, frame = cap.read()

if not ret:
    print("No se pudo capturar imagen.")
    cap.release()
    exit()

alto, ancho = frame.shape[:2]

# -----------------------------
# Inicializar pygame
# -----------------------------
pygame.init()

pantalla = pygame.display.set_mode((ancho, alto))
pygame.display.set_caption("Seleccionar 3 puntos")

fuente = pygame.font.SysFont("Arial", 22)

puntos = []

ejecutando = True

while ejecutando:

    # Leer imagen de la cámara
    ret, frame = cap.read()

    if not ret:
        break

    # Convertir BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rotar para pygame
    frame_rgb = np.rot90(frame_rgb)

    superficie = pygame.surfarray.make_surface(frame_rgb)

    pantalla.blit(superficie, (0, 0))

    # Dibujar puntos
    for i, p in enumerate(puntos):

        pygame.draw.circle(pantalla, (255, 0, 0), p, 6)

        texto = fuente.render(
            f"P{i+1}: ({p[0]}, {p[1]})",
            True,
            (255, 255, 0)
        )

        pantalla.blit(texto, (10, 10 + i * 30))

    pygame.display.flip()

    # Eventos
    for evento in pygame.event.get():

        if evento.type == pygame.QUIT:
            ejecutando = False

        elif evento.type == pygame.MOUSEBUTTONDOWN:

            if len(puntos) < 3:

                x, y = pygame.mouse.get_pos()
                puntos.append((x, y))

                print(f"Punto {len(puntos)}: ({x}, {y})")

        elif evento.type == pygame.KEYDOWN:

            # Guardar foto
            if evento.key == pygame.K_s:

                guardar_frame(frame, "captura")

            # Reiniciar puntos
            elif evento.key == pygame.K_r:

                puntos.clear()

# Liberar recursos
cap.release()
pygame.quit()