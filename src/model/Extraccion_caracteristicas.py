import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

class features_extractor:

    def __init__(self, hist_bins=16, debug=False):
        self.hist_bins = hist_bins
        self.debug = debug

    def get_feature_size(self):
        # 5 (geom) + 3 (stat) + 7 (hu) + (3 * hist_bins) (color) + 4 (medidas)
        return 5 + 3 + 7 + (3 * self.hist_bins) + 4

    def extract(self, image):
        if self.debug:
            original = image.copy()
            
        # Asegurar que la imagen tiene 3 canales para histogramas de color
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Segmentación para obtener la máscara del objeto
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Características geométricas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtro crítico: Si no hay contornos o son puro ruido menor a 50px, es fondo vacío
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        if not valid_contours:
            return np.zeros(self.get_feature_size())

        # Seleccionar el contorno dominante (objeto principal en la ventana)
        cnt = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * math.pi * area) / (perimeter**2 + 1e-6)
        compactness = (perimeter**2) / (area + 1e-6)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)

        geom_features = [area, perimeter, circularity, compactness, aspect_ratio]

        # Características estadísticas
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
        # Hu Moments
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

        visual_features = np.concatenate([hu_moments, hist_h, hist_s, hist_v])
        
        # CORREGIDO: Pasamos el contorno dominante directo para evitar recalcular bordes rotos
        medidas = self.largo_ancho(image, cnt)

        # Concatenar el VECTOR final de forma segura
        VECTOR = np.concatenate([geom_features, stat_features, visual_features, medidas])

        if self.debug:
            self.show_processed_images(original, gray, mask, cnt, image)
            
        return VECTOR

    def largo_ancho(self, img, cnt):
        """
        Calcula las medidas delimitadoras basadas en el contorno dominante detectado.
        Ya no recalcula Canny de cero, asegurando consistencia dimensional.
        """
        # Obtener el rectángulo delimitador directo del contorno
        x, y, ancho, alto = cv2.boundingRect(cnt)
        
        # Dibujar información si estás inspeccionando visualmente
        if self.debug:
            texto = f"{ancho}x{alto} px"
            cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Dinámico: Si es una ventana completa de entrenamiento (aprox > 300px) 
            # o una subventana de testeo (menor a 100px)
            if ancho > 300:
                cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        
        return np.array([x, y, ancho, alto], dtype=np.float32)

    def show_processed_images(self, original, gray, mask, contour, image_bgr):
        """Muestra las imágenes procesadas en diferentes etapas"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Imagen en Gris')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Máscara Binaria')
        axes[0, 2].axis('off')
        
        img_contour = original.copy()
        cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
        
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_contour, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        axes[1, 0].imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Contorno y Rectángulo')
        axes[1, 0].axis('off')
        
        masked_img = cv2.bitwise_and(gray, gray, mask=mask)
        axes[1, 1].imshow(masked_img, cmap='gray')
        axes[1, 1].set_title('Máscara aplicada')
        axes[1, 1].axis('off')
        
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hue_masked = cv2.bitwise_and(hsv[:,:,0], hsv[:,:,0], mask=mask)
        axes[1, 2].imshow(hue_masked, cmap='hsv', vmin=0, vmax=180)
        axes[1, 2].set_title('Canal de Matiz (HSV)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
