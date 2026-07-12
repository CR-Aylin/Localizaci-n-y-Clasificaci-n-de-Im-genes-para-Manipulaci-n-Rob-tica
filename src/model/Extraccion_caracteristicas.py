import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

class features_extractor:

    def __init__(self, hist_bins=16, debug=False, chroma_threshold=12):
        self.hist_bins = hist_bins
        self.debug = debug
        # Umbral de "chroma" (distancia al gris neutro en espacio LAB) para separar
        # objeto (con color) de fondo (gris/acromático). Se usa LAB en vez de la
        # saturación HSV porque la saturación HSV cae mucho en colores claros/pastel
        # (ej. el celeste claro del cubo), aunque el color siga siendo visible a simple
        # vista. LAB no tiene ese problema porque separa luminosidad (L) de color (a, b).
        self.chroma_threshold = chroma_threshold

    def get_feature_size(self):
        # 3 (geom, solo invariantes a escala) + 3 (stat) + 7 (hu) + (3 * hist_bins) (color)
        # area, perimetro, x, y, ancho, alto se calculan pero YA NO se incluyen en el
        # vector: son valores absolutos en píxeles que dependen de la resolución/distancia
        # de la foto, y en la ventana deslizante (fija en 100x100) nunca van a coincidir
        # con la escala de las fotos de entrenamiento, sin importar la clase real.
        return 3 + 3 + 7 + (3 * self.hist_bins)

    # nuevo: Método para extraer la imagen comprimida mediante PCA
    def _extract_pca_pixels(self, image, mask, cnt):
        """
        Extrae una representación comprimida de la imagen del objeto usando PCA.
        Esto permite incluir información visual espacial en el vector de características
        manteniendo un tamaño fijo e invariante a la posición dentro de la ventana.
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
        
        # Obtener píxeles no nulos del objeto
        pixels = gray_roi[roi_mask > 0]
        
        if len(pixels) < self.pca_components:
            # Si hay muy pocos píxeles, retornar ceros
            return np.zeros(self.pca_components)
        
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
        
        # Calcular componentes principales manualmente para evitar dependencia externa en extracción
        cov = np.cov(centered.reshape(1, -1)) if len(centered) > 1 else np.array([[0]])
        
        if cov.size == 1:
            return np.zeros(self.pca_components)
            
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Ordenar por valor propio descendente
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx[:self.pca_components]]
            
            # Proyectar datos
            compressed = np.dot(centered, eigenvectors)
            return compressed.flatten()
        except:
            return np.zeros(self.pca_components)
    # nuevo fin

    def sliding_window(self, image, window_size=(100, 100), step_size=50):
        h, w = image.shape[:2]
        win_h, win_w = window_size
        
        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                window = image[y:y+win_h, x:x+win_w]
                yield (x, y, window)

    def extract(self, image, return_bbox=False):
        if self.debug:
            original = image.copy()

        # Asegurar que la imagen tiene 3 canales para histogramas de color
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # --- SEGMENTACIÓN CORREGIDA ---
        # El fondo es GRIS (acromático) y los objetos tienen color. Segmentar por
        # intensidad (gray > 15) no sirve porque el fondo gris también supera ese
        # umbral (todo el mask se llenaba). Segmentar por saturación HSV tampoco es
        # suficiente porque colores CLAROS/pastel (ej. celeste pálido del cubo) tienen
        # saturación HSV baja aunque sí tengan color.
        # Solución: usar el "chroma" en espacio LAB, es decir, qué tan lejos está cada
        # píxel del gris neutro (a=128, b=128), sin que la luminosidad (L) afecte el
        # cálculo. Esto detecta color tanto en tonos saturados (rojo) como pastel (celeste).
        a_channel = lab[:, :, 1].astype(np.float32) - 128.0
        b_channel = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a_channel**2 + b_channel**2)
        mask = np.where(chroma > self.chroma_threshold, 255, 0).astype(np.uint8)

        # Limpieza morfológica para quitar ruido de píxeles sueltos y cerrar huecos
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Características geométricas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtro crítico: Si no hay contornos o son puro ruido menor a 50px, es fondo vacío
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        if not valid_contours:
            if return_bbox:
                return np.zeros(self.get_feature_size()), (0, 0, 0, 0)
            return np.zeros(self.get_feature_size())

        # Seleccionar el contorno dominante (objeto principal en la ventana)
        cnt = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * math.pi * area) / (perimeter**2 + 1e-6)
        compactness = (perimeter**2) / (area + 1e-6)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)

        # area y perimeter se calculan (se necesitan para circularidad/compacidad)
        # pero NO se incluyen crudos en el vector: no son invariantes a escala y
        # varían muchísimo entre fotos de entrenamiento y ventanas de test 100x100.
        geom_features = [circularity, compactness, aspect_ratio]

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
        hist_h = cv2.calcHist([hsv], [0], mask, [self.hist_bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], mask, [self.hist_bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], mask, [self.hist_bins], [0, 256]).flatten()

        # Normalizar histogramas
        hist_h /= (hist_h.sum() + 1e-6)
        hist_s /= (hist_s.sum() + 1e-6)
        hist_v /= (hist_v.sum() + 1e-6)

        visual_features = np.concatenate([hu_moments, hist_h, hist_s, hist_v])

        # Concatenar el VECTOR final de forma segura.
        # NOTA: x, y, ancho, alto (bounding box absoluto) ya NO se incluyen aquí.
        # No aportan información de clase real (dependen de dónde cae la ventana
        # o la resolución de la foto) y solo confundían al clasificador. Si se
        # necesitan para dibujar o depurar, usa self.largo_ancho(image, cnt) aparte.
        #VECTOR = np.concatenate([geom_features, stat_features, visual_features])

        # nuevo: Extraer imagen comprimida mediante PCA
        pca_features = self._extract_pca_pixels(image, mask, cnt)
        # nuevo fin

        # Concatenar el VECTOR final incluyendo la imagen comprimida
        # nuevo: Se añade pca_features al vector final
        VECTOR = np.concatenate([geom_features, stat_features, visual_features, pca_features])
        # nuevo fin

        if self.debug:
            self.largo_ancho(image, cnt)
            self.show_processed_images(original, gray, mask, cnt, image)

        if return_bbox:
            # Posición y tamaño del objeto (metadata útil para localización/registro,
            # pero NO se incluye en VECTOR: no es invariante a escala y confunde al
            # clasificador, como vimos con el bug anterior)
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(cnt)
            return VECTOR, (bbox_x, bbox_y, bbox_w, bbox_h)

        return VECTOR

    def extract_d(self, image, window_size=(100, 100), step_size=50, min_object_area=50):
        detections = []
        
        for x, y, window in self.sliding_window(image, window_size, step_size):
            # Extraer características de esta ventana
            features, bbox = self.extract(window, return_bbox=True)
            
            # Verificar si se detectó un objeto (features no son todos ceros)
            if not np.all(features == 0):
                # Ajustar bbox a coordenadas globales de la imagen
                global_bbox = (x + bbox[0], y + bbox[1], bbox[2], bbox[3])
                detections.append((x, y, features, global_bbox))
        
        return detections

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