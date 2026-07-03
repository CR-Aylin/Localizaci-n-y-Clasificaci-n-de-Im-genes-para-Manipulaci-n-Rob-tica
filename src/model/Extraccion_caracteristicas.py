class features_extractor:
    def __init__(self, hist_bins=16):
        self.hist_bins = hist_bins

    def extract(self, image):
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

        visual_features = np.concatenate([hu_moments, hist_h, hist_s, hist_v])

        # Concatenar todo el vector de características
        final_features = np.concatenate([geom_features, stat_features, visual_features])
        return final_features

    def get_feature_size(self):
        # 5 (geom) + 3 (stat) + 7 (hu) + 3 * hist_bins (color)
        return 5 + 3 + 7 + (3 * self.hist_bins)
