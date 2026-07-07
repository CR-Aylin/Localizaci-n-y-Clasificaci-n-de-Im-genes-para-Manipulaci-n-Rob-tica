import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


class features_extractor:

    def __init__(self, hist_bins=16 , debug=False):
        self.hist_bins = hist_bins
        self.debug = debug

    def extract(self, image):
        if self.debug:
            original = image.copy()
        # Asegurar que la imagen tiene 3 canales para histogramas de color
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Segmentación simple para obtener la máscara del objeto
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Características geométricas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si no hay contornos retornar ceros
        if not contours:
            return np.zeros(self.get_feature_size())

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * math.pi * area) / (perimeter**2 + 1e-6)
        compactness = (perimeter**2) / (area + 1e-6)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)

        geom_features = [area, perimeter, circularity, compactness, aspect_ratio]

        # Características etsadísticas
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        pixels = masked_gray[mask > 0]

        if len(pixels) == 0:
            mean_int, std_int, entropy = 0, 0, 0
        else:
            mean_int = np.mean(pixels)
            std_int = np.std(pixels)

            # Entropía
            hist_gray = cv2.calcHist([masked_gray], [0], mask, [256], [0, 256])
            hist_gray = hist_gray.flatten() / (hist_gray.sum() + 1e-6)
            entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-6))

        stat_features = [mean_int, std_int, entropy]

        # Características visuales
        # Hu Moments (7 momentos invariantes a escala, traslación y rotación)
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()

        # Histogramas de Color (Espacio HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], mask, [self.hist_bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], mask, [self.hist_bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], mask, [self.hist_bins], [0, 256]).flatten()

        # Normalizar histogramas
        hist_h /= (hist_h.sum() + 1e-6)
        hist_s /= (hist_s.sum() + 1e-6)
        hist_v /= (hist_v.sum() + 1e-6)

        medidas= self.largo_ancho(image)

        visual_features = np.concatenate([hu_moments, hist_h, hist_s, hist_v])
        
        

        # Concatenar todo el vector de características
        final_features = np.concatenate([medidas, geom_features, stat_features, visual_features])
        if self.debug:
            self.show_processed_images(original, gray, mask, cnt, image)
        return final_features

    def get_feature_size(self):
        # 5 (geom) + 3 (stat) + 7 (hu) + 3 * hist_bins (color)
        return 5 + 3 + 7 + (3 * self.hist_bins)
    
    def largo_ancho(self,img):
    
        img_original = img.copy()

        # Preprocesamiento (Grises y desenfoque para reducir ruido)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        suave = cv2.GaussianBlur(gris, (5, 5), 0)

        #Detección de Bordes (Canny)
        bordes = cv2.Canny(suave, 50, 150)

        #Detección de Esquinas (Harris)
        # Operamos sobre la imagen en gris (debe ser tipo float32)
        gris_float = np.float32(gris)
        dest_esquinas = cv2.cornerHarris(gris_float, blockSize=2, ksize=3, k=0.04)


        # Dilatamos el resultado para que las esquinas se vean más grandes al dibujar
        dest_esquinas = cv2.dilate(dest_esquinas, None)
        # Marcamos las esquinas en la imagen original en color ROJO
        img[dest_esquinas > 0.01 * dest_esquinas.max()] = [0, 0, 255]

        # Contornos para medir Ancho y Alto
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            if cv2.contourArea(c) < 100:
                continue
                
            # Obtener el rectángulo delimitador (Bounding Box)
            x, y, ancho, alto = cv2.boundingRect(c)
            # (Ancho x Alto) en la imagen en color AZUL

            texto = f"{ancho}x{alto} px"

            cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if(ancho == 383 and alto == 389):
                print("Cuadrado")
                print(x, y, ancho, alto)
                # Dibujar el rectángulo delimitador en color VERDE
                cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
            if(ancho == 112 and alto == 109):
                print("circulo")
                print(x, y, ancho, alto)
                cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

            if(self.debug):
                # OpenCV usa BGR, Matplotlib usa RGB; por eso convertimos los colores
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 6))
                plt.imshow(img_rgb)
                plt.axis('off')  # Oculta los ejes de coordenadas
                plt.show()
                
            vector=[x, y, ancho, alto]
            return vector
        
    def show_processed_images(self, original, gray, mask, contour, image_bgr):
            """Muestra las imágenes procesadas en diferentes etapas"""
            
            # Crear una figura con varias subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Imagen original
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Imagen Original')
            axes[0, 0].axis('off')
            
            # Imagen en gris
            axes[0, 1].imshow(gray, cmap='gray')
            axes[0, 1].set_title('Imagen en Gris')
            axes[0, 1].axis('off')
            
            # Máscara binaria
            axes[0, 2].imshow(mask, cmap='gray')
            axes[0, 2].set_title('Máscara Binaria')
            axes[0, 2].axis('off')
            
            # Imagen con contorno dibujado
            img_contour = original.copy()
            cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
            
            # Dibujar rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            axes[1, 0].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Contorno y Rectángulo')
            axes[1, 0].axis('off')
            
            # Máscara aplicada a la imagen en gris
            masked_img = cv2.bitwise_and(gray, gray, mask=mask)
            axes[1, 1].imshow(masked_img, cmap='gray')
            axes[1, 1].set_title('Máscara aplicada')
            axes[1, 1].axis('off')
            
            # Imagen con segmentación de color (mostrar en HSV)
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            # Mostrar canal de matiz con máscara
            hue_masked = cv2.bitwise_and(hsv[:,:,0], hsv[:,:,0], mask=mask)
            axes[1, 2].imshow(hue_masked, cmap='hsv', vmin=0, vmax=180)
            axes[1, 2].set_title('Canal de Matiz (HSV)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # También mostrar histogramas por separado
            self.show_histograms(original, mask)
        
    def show_histograms(self, image, mask):
            """Muestra los histogramas de color"""
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histograma del canal H
            hist_h = cv2.calcHist([hsv], [0], mask, [self.hist_bins], [0, 180])
            axes[0].bar(range(self.hist_bins), hist_h.flatten())
            axes[0].set_title('Histograma del Canal H (Matiz)')
            axes[0].set_xlabel('Bin')
            axes[0].set_ylabel('Frecuencia')
            
            # Histograma del canal S
            hist_s = cv2.calcHist([hsv], [1], mask, [self.hist_bins], [0, 256])
            axes[1].bar(range(self.hist_bins), hist_s.flatten())
            axes[1].set_title('Histograma del Canal S (Saturación)')
            axes[1].set_xlabel('Bin')
            axes[1].set_ylabel('Frecuencia')
            
            # Histograma del canal V
            hist_v = cv2.calcHist([hsv], [2], mask, [self.hist_bins], [0, 256])
            axes[2].bar(range(self.hist_bins), hist_v.flatten())
            axes[2].set_title('Histograma del Canal V (Valor)')
            axes[2].set_xlabel('Bin')
            axes[2].set_ylabel('Frecuencia')
            
            plt.tight_layout()
            plt.show()
            
            def proceso(Rutaimagen):
                #image = cv2.imread("C:\\Users\\User\\OneDrive\\Escritorio\\Blender_Trabajo\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\\dataset\\WIN_20260702_19_15_01_Pro.jpg")
                image = cv2.imread(Rutaimagen)
                
                # Imagen se cargó correctamente
                if image is None:
                    print("No se pudo cargar la imagen. Verifica la ruta.")
                else:
                    extractor = features_extractor(hist_bins=16, debug=True)
                    caracteristicas = extractor.extract(image)
                    print(f"Características extraídas: {caracteristicas.shape}")
                    print(f"Vector de características 10 : {caracteristicas[:10]}...")
                    print(f"Vector de características: {caracteristicas}")
                    print(f"Tamaño total del vector: {len(caracteristicas)}")
                
                return caracteristicas

"""
if __name__ == "__main__":
    
    #image = cv2.imread("C:\\Users\\User\\OneDrive\\Escritorio\\Blender_Trabajo\\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\\dataset\\WIN_20260702_19_15_01_Pro.jpg")
    image = cv2.imread("P:\IA - JAR\Localizaci-n-y-Clasificaci-n-de-Im-genes-para-Manipulaci-n-Rob-tica\dataset\Entrenamiento\Class_2\WIN_20260702_19_23_03_Pro.jpg")
    
    # Imagen se cargó correctamente
    if image is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
    else:
        extractor = features_extractor(hist_bins=16, debug=True)
        caracteristicas = extractor.extract(image)
        print(f"Características extraídas: {caracteristicas.shape}")
        print(f"Vector de características 10 : {caracteristicas[:10]}...")
        print(f"Vector de características: {caracteristicas}")
        print(f"Tamaño total del vector: {len(caracteristicas)}")
        
"""
# vector = [Características Geométricas, Características Estadísticas, Momentos de Hu, Histogramas de Color]