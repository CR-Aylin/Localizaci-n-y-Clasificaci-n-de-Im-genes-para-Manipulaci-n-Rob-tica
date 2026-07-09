import time
import ptzpro  # Asegúrate de que el nombre exacto de la clase sea el correcto


class PTZCamera:

    def __init__(self):
        # NOTA: Reemplaza 'Camera' por la clase real de tu librería si se llama distinto
        try:
            self.cam = ptzpro.Camera()
        except AttributeError:
            # Si ptzpro se instancia directamente (raro, pero posible en wrappers de C)
            self.cam =  ptzpro.PTZPro()
        except Exception as e:
            print(f"Error al inicializar el hardware de la cámara: {e}")
            self.cam = None

    def obtener_posicion(self):
        """Lee la posición actual de la cámara."""
        if not self.cam:
            return "Cámara no inicializada."
        try:
            posicion = self.cam.get_position()
            return posicion
        except Exception as e:
            return f"No se pudo leer posición: {e}"

    def mover(self, pan=0, tilt=0, zoom=0):
        """Mueve la cámara de manera secuencial o simultánea según la librería."""
        if not self.cam:
            print("Error: No hay conexión con la cámara.")
            return

        try:
            if pan != 0:
                self.cam.pan(pan)

            if tilt != 0:
                self.cam.tilt(tilt)

            if zoom != 0:
                self.cam.zoom(zoom)
        except Exception as e:
            print(f"Error durante el movimiento: {e}")


# --- Flujo de Ejecución ---

camara = PTZCamera()

# Posición inicial
print("Posición inicial:")
print(camara.obtener_posicion())
print("-" * 30)

# Movimiento
print("Moviendo cámara (Pan: 100, Tilt: -50, Zoom: 20)...")
camara.mover(pan=100, tilt=-50, zoom=20)

# Pausa para dar tiempo al hardware físico a completar el movimiento
print("Esperando a que el hardware finalice el movimiento...")
time.sleep(3)  # Aumentado a 3 segundos por seguridad

# Nueva posición
print("-" * 30)
print("Nueva posición alcanzada:")
print(camara.obtener_posicion())