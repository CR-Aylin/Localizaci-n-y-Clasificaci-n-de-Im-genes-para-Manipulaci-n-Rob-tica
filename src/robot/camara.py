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
        

    def sacar_fotor(self, nombre="foto"):

        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)

        if not captura.isOpened():
            print("No se pudo abrir la cámara")
            return None

        pygame.init()

        ret, frame = captura.read()
        if not ret:
            captura.release()
            pygame.quit()
            return None

        alto, ancho = frame.shape[:2]

        pantalla = pygame.display.set_mode((ancho, alto))
        pygame.display.set_caption("Seleccione 2 esquinas y 1 punto de referencia")

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

                color = (255, 0, 0)

                if i == 2:
                    color = (255, 255, 0)   # Punto de referencia

                pygame.draw.circle(pantalla, color, p, 5)

                texto = fuente.render(
                    f"P{i+1}: ({p[0]}, {p[1]})",
                    True,
                    (255, 255, 255)
                )

                pantalla.blit(texto, (10, 10 + i * 25))

            # Rectángulo usando SOLO los dos primeros puntos
            if len(puntos) >= 2:

                x1, y1 = puntos[0]
                x2, y2 = puntos[1]

                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

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

                    # Reiniciar puntos
                    if evento.key == pygame.K_r:
                        puntos.clear()
                        print("Puntos reiniciados.")

                    # Guardar imagen
                    elif evento.key == pygame.K_c:

                        if len(puntos) == 3:

                            x1, y1 = puntos[0]
                            x2, y2 = puntos[1]
                            x_ref, y_ref = puntos[2]

                            xmin = min(x1, x2)
                            xmax = max(x1, x2)
                            ymin = min(y1, y2)
                            ymax = max(y1, y2)
                            
                            ret, frame = captura.read()

                            if ret:

                                recorte = frame[ymin:ymax, xmin:xmax]
                                # Escalar la imagen a 1920x180 píxeles
                                ancho = 1920
                                alto = 180
                                frame_escalado = cv2.resize(recorte, (ancho, alto), interpolation=cv2.INTER_LINEAR)
                            

                                ruta = r"dataset\Pruebas"
                                os.makedirs(ruta, exist_ok=True)

                                ruta_completa = os.path.join(
                                    ruta,
                                    nombre + ".png"
                                )

                                cv2.imwrite(ruta_completa, frame_escalado)

                                print("Imagen guardada:", ruta_completa)
                                print("Punto de referencia:", (x_ref, y_ref))

                                captura.release()
                                pygame.quit()

                                return {
                                    "puntos": puntos,
                                    "referencia": (x_ref, y_ref),
                                    "ruta": ruta_completa
                                }

                    elif evento.key == pygame.K_ESCAPE:
                        ejecutando = False

                elif evento.type == pygame.MOUSEBUTTONDOWN:

                    if len(puntos) < 3:

                        x, y = pygame.mouse.get_pos()
                        puntos.append((x, y))

                        print(f"Punto {len(puntos)}: ({x}, {y})")

        captura.release()
        pygame.quit()

        return None
    
    def sacar_foto(self, nombre="foto"):

        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)

        if not captura.isOpened():
            print("No se pudo abrir la cámara")
            return None

        pygame.init()

        ret, frame = captura.read()
        if not ret:
            captura.release()
            pygame.quit()
            return None

        alto, ancho = frame.shape[:2]

        pantalla = pygame.display.set_mode((ancho, alto))
        pygame.display.set_caption("Seleccione 2 esquinas y 1 punto de referencia")

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

                color = (255, 0, 0)

                if i == 2:
                    color = (255, 255, 0)   # Punto de referencia

                pygame.draw.circle(pantalla, color, p, 5)

                texto = fuente.render(
                    f"P{i+1}: ({p[0]}, {p[1]})",
                    True,
                    (255, 255, 255)
                )

                pantalla.blit(texto, (10, 10 + i * 25))

            # Rectángulo usando SOLO los dos primeros puntos
            if len(puntos) >= 2:

                x1, y1 = puntos[0]
                x2, y2 = puntos[1]

                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

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

                    # Reiniciar puntos
                    if evento.key == pygame.K_r:
                        puntos.clear()
                        print("Puntos reiniciados.")

                    # Guardar imagen
                    elif evento.key == pygame.K_c:

                        if len(puntos) == 3:

                            x1, y1 = puntos[0]
                            x2, y2 = puntos[1]
                            x_ref, y_ref = puntos[2]

                            xmin = min(x1, x2)
                            xmax = max(x1, x2)
                            ymin = min(y1, y2)
                            ymax = max(y1, y2)
                            

                            ruta = r"dataset\Pruebas"
                            os.makedirs(ruta, exist_ok=True)

                            ruta_completa = os.path.join(
                                ruta,
                                nombre + ".png"
                            )

                            cv2.imwrite(ruta_completa, frame)

                            print("Imagen guardada:", ruta_completa)
                            print("Punto de referencia:", (x_ref, y_ref))

                            captura.release()
                            pygame.quit()

                            return {
                                "puntos": puntos,
                                "referencia": (x_ref, y_ref),
                                "ruta": ruta_completa
                            }

                    elif evento.key == pygame.K_ESCAPE:
                        ejecutando = False

                elif evento.type == pygame.MOUSEBUTTONDOWN:

                    if len(puntos) < 3:

                        x, y = pygame.mouse.get_pos()
                        puntos.append((x, y))

                        print(f"Punto {len(puntos)}: ({x}, {y})")

        captura.release()
        pygame.quit()

        return None
    
    
    def probar():

        cam = Cam()

        resultado = cam.sacar_foto("Prueba1")

        if resultado:

            print("Puntos:", resultado["puntos"])
            print("Referencia:", resultado["referencia"])
            print("Imagen:", resultado["ruta"])

        else:
            print("No se guardó ninguna imagen.")
            
        return resultado

"""


if __name__ == "__main__":

    cam = Cam()

    resultado = cam.sacar_foto("Prueba1")

    if resultado:

        puntos, ruta = resultado

        print("Puntos seleccionados:")
        print(puntos)

        print("Imagen guardada en:")
        print(ruta)

    else:
        print("No se guardó ninguna imagen.")
"""
