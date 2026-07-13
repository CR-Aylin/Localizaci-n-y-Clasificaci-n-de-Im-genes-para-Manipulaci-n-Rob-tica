import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

class features_extractor:

    def __init__(self, hist_bins=16, debug=False, chroma_threshold=12):
        self.hist_bins = hist_bins
        self.debug = debug
        self.chroma_threshold = chroma_threshold

    def comprimir_imagen(self, image, target_size=(64, 64)):
        """Prepara una imagen completa desde disco redimensionándola a tamaño fijo."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Redimensionar a tamaño fijo
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        img_ia = img_resized.astype(np.float32) / 255.0
        return img_ia.flatten()

    def get_feature_size(self):
<<<<<<< Updated upstream
        # 3 (geom, solo invariantes a escala) + 3 (stat) + 7 (hu) + (3 * hist_bins) (color)
        # area, perimetro, x, y, ancho, alto se calculan pero YA NO se incluyen en el
        # vector: son valores absolutos en píxeles que dependen de la resolución/distancia
        # de la foto, y en la ventana deslizante (fija en 100x100) nunca van a coincidir
        # con la escala de las fotos de entrenamiento, sin importar la clase real.
        return 3 + 3 + 7 + (3 * self.hist_bins)
=======
        # 3 (geom) + 3 (stat) + 7 (hu) + (3 * hist_bins) (color) + pca_components
        return 3 + 3 + 7 + (3 * self.hist_bins) + self.pca_components

    def _extract_pca_pixels(self, image, mask, cnt):
        """
        Extrae una representación comprimida de la imagen del objeto usando PCA.
        Mantiene un tamaño fijo e invariante a la posición.
        """
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Recortar la región del objeto usando el bounding box
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        # Aplicar máscara para obtener solo los píxeles del objeto
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
        
        # Convertir a escala de grises para PCA
        if len(masked_roi.shape) == 3:
            gray_roi = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = masked_roi
        
        # Redimensionar a un tamaño fijo para consistencia
        fixed_size = 32
        resized = cv2.resize(gray_roi, (fixed_size, fixed_size), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(roi_mask, (fixed_size, fixed_size), interpolation=cv2.INTER_NEAREST)
        
        # Aplicar máscara al redimensionado
        resized_masked = cv2.bitwise_and(resized, resized, mask=resized_mask)
        
        # Aplanar y aplicar PCA
        flat = resized_masked.flatten().astype(np.float32)
        
        # Centrar datos
        mean_val = np.mean(flat[flat > 0]) if np.any(flat > 0) else 0
        centered = flat - mean_val
        
        # Calcular componentes principales de la ventana actual
        cov = np.cov(centered.reshape(1, -1)) if len(centered) > 1 else np.array([[0]])
        
        if cov.size == 1:
            return np.zeros(self.pca_components)
            
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx[:self.pca_components]]
            
            compressed = np.dot(centered, eigenvectors)
            return compressed.flatten()
        except:
            return np.zeros(self.pca_components)

    def sliding_window(self, image, window_size=(100, 100), step_size=50):
        h, w = image.shape[:2]
        win_h, win_w = window_size
        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                window = image[y:y+win_h, x:x+win_w]
                yield (x, y, window)
>>>>>>> Stashed changes

    def extract(self, image, return_bbox=False):
        if self.debug:
            original = image.copy()

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 1. Extraemos la imagen aplanada desde el principio usando self
        imagen_flaten = self.comprimir_imagen(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # --- SEGMENTACIÓN POR CHROMA ---
        a_channel = lab[:, :, 1].astype(np.float32) - 128.0
        b_channel = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a_channel**2 + b_channel**2)
        mask = np.where(chroma > self.chroma_threshold, 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        # 2. CORRECCIÓN: Si no hay contornos, devolver vector vacío sumando la imagen aplanada
        if not valid_contours:
            ceros_features = np.zeros(self.get_feature_size())
            vector_vacio = np.concatenate([imagen_flaten, ceros_features])
            
            if return_bbox:
                return vector_vacio, (0, 0, 0, 0)
            return vector_vacio

        cnt = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * math.pi * area) / (perimeter**2 + 1e-6)
        compactness = (perimeter**2) / (area + 1e-6)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)
        geom_features = [circularity, compactness, aspect_ratio]

        # Estadísticas
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        pixels = masked_gray[mask > 0]

        if len(pixels) == 0:
            mean_int, std_int, entropy = 0, 0, 0
        else:
            mean_int = np.mean(pixels)
            std_int = np.std(pixels)
            hist_gray = cv2.calcHist([masked_gray], [0], mask, [256], [0, 256]).flatten()
            hist_gray = hist_gray / (hist_gray.sum() + 1e-6)
            entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-6))

        stat_features = [mean_int, std_int, entropy]

        # Hu Moments & Histogramas HSV
        hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
        hist_h = cv2.calcHist([hsv], [0], mask, [self.hist_bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], mask, [self.hist_bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], mask, [self.hist_bins], [0, 256]).flatten()

        hist_h /= (hist_h.sum() + 1e-6)
        hist_s /= (hist_s.sum() + 1e-6)
        hist_v /= (hist_v.sum() + 1e-6)
        
        

        visual_features = np.concatenate([hu_moments, hist_h, hist_s, hist_v])
<<<<<<< Updated upstream

        # Concatenar el VECTOR final de forma segura.
        # NOTA: x, y, ancho, alto (bounding box absoluto) ya NO se incluyen aquí.
        # No aportan información de clase real (dependen de dónde cae la ventana
        # o la resolución de la foto) y solo confundían al clasificador. Si se
        # necesitan para dibujar o depurar, usa self.largo_ancho(image, cnt) aparte.
        VECTOR = np.concatenate([geom_features, stat_features, visual_features])
=======
        pca_features = self._extract_pca_pixels(image, mask, cnt)

        # 3. CORRECCIÓN: Concatenación segura de todos los vectores 1D
        VECTOR = np.concatenate([imagen_flaten, geom_features, stat_features, visual_features, pca_features])
>>>>>>> Stashed changes

        if self.debug:
            self.largo_ancho(image, cnt)
            self.show_processed_images(original, gray, mask, cnt, image)

        if return_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
            return VECTOR, (bbox_x, bbox_y, bbox_w, bbox_h)

<<<<<<< Updated upstream
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
    
    def extraer_caracteristicas(img):
        """
        Recibe una imagen leída con cv2.imread() y devuelve un vector
        unidimensional con el promedio de los canales R y G.
        """

        # Convertir de BGR a RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Separar canales
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)

        # Promedio de R y G
        promedio = (R + G) / 2.0

        # Convertir a vector unidimensional
        return promedio.flatten()

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
        axes[0, 2].set_title('Máscara Binaria (saturación)')
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
=======
        return VECTOR
>>>>>>> Stashed changes
