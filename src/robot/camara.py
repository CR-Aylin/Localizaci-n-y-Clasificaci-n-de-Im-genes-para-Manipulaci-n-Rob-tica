import cv2
import time
import pygame
import os
import numpy as np


class Cam:

    def configuracion_camara(self):
        # Abrir la cámara
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)

        # ---------- CONFIGURACIÓN ----------
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 20)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_CONTRAST, 64)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        print("Configuración aplicada")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("PTZ Pro 2", frame)

            if cv2.waitKey(1) == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    # ===============================
    # MÉTODO PARA TOMAR Y RECORTAR FOTO
    # ===============================
    def sacar_foto(self, nombre="foto"):

        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)

        if not captura.isOpened():
            print("No se pudo abrir la cámara")
            return None

        ret, frame = captura.read()

        if not ret:
            print("No se pudo capturar imagen")
            captura.release()
            return None

        alto, ancho = frame.shape[:2]

        pygame.init()

        pantalla = pygame.display.set_mode((ancho, alto))
        pygame.display.set_caption("Seleccione 3 puntos")

        fuente = pygame.font.SysFont("Arial", 20)

        puntos = []

        ejecutando = True

        while ejecutando:

            ret, frame = captura.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.rot90(frame_rgb)

            superficie = pygame.surfarray.make_surface(frame_rgb)

            pantalla.blit(superficie, (0, 0))

            # Dibujar puntos
            for i, p in enumerate(puntos):

                pygame.draw.circle(pantalla, (255, 0, 0), p, 5)

                texto = fuente.render(
                    f"P{i+1}: ({p[0]}, {p[1]})",
                    True,
                    (255, 255, 0)
                )

                pantalla.blit(texto, (10, 10 + i * 25))

            # Dibujar rectángulo
            if len(puntos) == 3:

                xs = [p[0] for p in puntos]
                ys = [p[1] for p in puntos]

                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                pygame.draw.rect(
                    pantalla,
                    (0, 255, 0),
                    (xmin, ymin, xmax - xmin, ymax - ymin),
                    2
                )

            pygame.display.flip()

            for evento in pygame.event.get():

                if evento.type == pygame.QUIT:
                    ejecutando = False

                elif evento.type == pygame.KEYDOWN:

                    if evento.key == pygame.K_r:
                        puntos.clear()

                elif evento.type == pygame.MOUSEBUTTONDOWN:

                    if len(puntos) < 3:

                        x, y = pygame.mouse.get_pos()
                        puntos.append((x, y))

                        print(f"Punto {len(puntos)}: ({x}, {y})")

                        # Cuando ya existen 3 puntos
                        if len(puntos) == 3:

                            xs = [p[0] for p in puntos]
                            ys = [p[1] for p in puntos]

                            xmin = min(xs)
                            xmax = max(xs)
                            ymin = min(ys)
                            ymax = max(ys)

                            # Evitar recortes vacíos
                            if xmax > xmin and ymax > ymin:

                                recorte = frame[ymin:ymax, xmin:xmax]

                                ruta = r"dataset\Pruebas"
                                os.makedirs(ruta, exist_ok=True)

                                ruta_completa = os.path.join(
                                    ruta,
                                    nombre + ".png"
                                )

                                cv2.imwrite(ruta_completa, recorte)

                                print("Recorte guardado en:", ruta_completa)

                                captura.release()
                                pygame.quit()

                                return ruta_completa

        captura.release()
        pygame.quit()

        return None


if __name__ == "__main__":

    cam = Cam()

    ruta = cam.sacar_foto("Prueba1")

    if ruta:
        print("Imagen guardada en:", ruta)
    else:
        print("No se guardó ninguna imagen.")