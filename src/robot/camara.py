import cv2
import time
import os 

class Cam():

    def configuracion_camara(self): 
        # Abrir la cámara
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2)

        # ---------- CONFIGURACIÓN ----------

        # Desactivar enfoque automático
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # Ajustar enfoque manual
        cap.set(cv2.CAP_PROP_FOCUS, 20)

        # Brillo
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)

        # Contraste
        cap.set(cv2.CAP_PROP_CONTRAST, 64)

        # Saturación (Intensidad de color)
        cap.set(cv2.CAP_PROP_SATURATION, 128)

        # Balance de blancos automático
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # Exposición automática
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        print("Configuración aplicada")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("PTZ Pro 2", frame)

            if cv2.waitKey(1) == 27:  # ESC para salir
                break

        cap.release()
        cv2.destroyAllWindows()
        

    def sacar_foto(self, nombre="foto"):  # ← AGREGA self y valor por defecto
        captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # abrir camara
        time.sleep(2)
        
        # Capturar imagen
        ret, frame = captura.read()
        
        # Verificar si capturó correctamente
        if not ret or frame is None:
            print("Error: No se pudo capturar el frame")
            captura.release()
            return None  # ← CORREGIDO: retornar None en error
        
        # Crear el directorio si no existe
        ruta = r"dataset\Pruebas" 
        if not os.path.exists(ruta):
            os.makedirs(ruta)
            print(f"Directorio creado: {ruta}")

        ruta_completa = os.path.join(ruta, nombre + ".png")
        cv2.imwrite(ruta_completa, frame)
        print(f"Foto guardada en: {ruta_completa}")
        
        captura.release()
        
        return ruta_completa  
"""
if __name__ == "__main__":
    # Crear una instancia de la clase
    mi_camara = Cam()  # ← CREAR INSTANCIA
    
    # Llamar al método con la instancia
    ruta_guardada = mi_camara.sacar_foto("Prueba1")  # ← Usar instancia
    
    if ruta_guardada:
        print(f" Foto guardada exitosamente en: {ruta_guardada}")
    else:
        print(" Error al guardar la foto")
    
    # Opcional: abrir la cámara en vivo
    # mi_camara.configuracion_camara()  # ← Usar instancia"""