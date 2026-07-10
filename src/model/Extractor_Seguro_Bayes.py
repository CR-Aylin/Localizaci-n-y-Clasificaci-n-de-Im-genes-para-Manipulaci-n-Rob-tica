import cv2
import numpy as np

class ExtractorSeguro:
    def __init__(self, chroma_threshold=10, area_minima=30):
        """
        Extractor de características con parámetros ajustables.
        chroma_threshold: umbral para segmentación por saturación (más bajo = más sensible)
        area_minima: área mínima del contorno a considerar
        """
        self.chroma_threshold = chroma_threshold
        self.area_minima = area_minima
        self.hist_bins = 16  # Bins para el histograma
        self.hist_range = (0, 256)
        
    def extract(self, imagen):
        """
        Extrae características de la imagen:
        - Histograma de color en HSV (16 bins por canal)
        - 7 momentos de Hu invariantes
        """
        if imagen is None:
            return None
            
        # Intentar encontrar el contorno principal
        contorno = self._encontrar_contorno_principal(imagen)
        
        if contorno is None:
            return None
            
        # Extraer características del contorno
        try:
            # 1. Máscara del contorno
            mask = np.zeros(imagen.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contorno], -1, 255, -1)
            
            # 2. Histograma en HSV
            hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], mask, [self.hist_bins], self.hist_range).flatten()
            hist_s = cv2.calcHist([hsv], [1], mask, [self.hist_bins], self.hist_range).flatten()
            hist_v = cv2.calcHist([hsv], [2], mask, [self.hist_bins], self.hist_range).flatten()
            
            # Normalizar histogramas
            hist_h = hist_h / (np.sum(hist_h) + 1e-7)
            hist_s = hist_s / (np.sum(hist_s) + 1e-7)
            hist_v = hist_v / (np.sum(hist_v) + 1e-7)
            
            # 3. Momentos de Hu
            moments = cv2.moments(contorno)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
            
            # Combinar todas las características
            features = np.concatenate([hist_h, hist_s, hist_v, hu_moments])
            
            return features
            
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return None
    
    def _encontrar_contorno_principal(self, imagen):
        """
        Encuentra el contorno principal usando segmentación por chroma en espacio LAB
        """
        try:
            # Convertir a LAB
            lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
            
            # Calcular chroma (saturación en LAB)
            a_channel = lab[:, :, 1].astype(np.float32) - 128.0
            b_channel = lab[:, :, 2].astype(np.float32) - 128.0
            chroma = np.sqrt(a_channel**2 + b_channel**2)
            
            # Crear máscara por umbral
            mask = np.where(chroma > self.chroma_threshold, 255, 0).astype(np.uint8)
            
            # Limpieza morfológica
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar por área mínima
            validos = [c for c in contornos if cv2.contourArea(c) > self.area_minima]
            
            if not validos:
                return None
                
            # Devolver el contorno de mayor área
            return max(validos, key=cv2.contourArea)
            
        except Exception as e:
            print(f"Error encontrando contorno: {e}")
            return None


    def aplanar(self, img):
        """
        Recibe una imagen leída con cv2.imread() y devuelve un vector
        unidimensional con el promedio de los canales R y G.
        """

        # Convertir de BGR a RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Separar canales
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)

        # Promedio de R y G y B
        promedio = (R + G + B) / 3.0

        # Convertir a vector unidimensional
        return promedio.flatten()


    def extraer_objetos(self, imagen):
        """
        Devuelve todos los objetos encontrados con sus características.
        """
        resultados = []

        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        aplanada = self.aplanar(imagen)
        #lab = imagen

        a = lab[:, :, 1].astype(np.float32) - 128
        b = lab[:, :, 2].astype(np.float32) - 128

        chroma = np.sqrt(a*a + b*b)

        mask = np.where(chroma > self.chroma_threshold,255,0).astype(np.uint8)

        kernel = np.ones((3,3),np.uint8)

        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

        contornos,_ = cv2.findContours(mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:

            if cv2.contourArea(contorno) < self.area_minima:
                continue

            mascara = np.zeros(imagen.shape[:2],dtype=np.uint8)
            cv2.drawContours(mascara,[contorno],-1,255,-1)

            hsv = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)

            hist_h = cv2.calcHist([hsv],[0],mascara,[self.hist_bins],self.hist_range).flatten()
            hist_s = cv2.calcHist([hsv],[1],mascara,[self.hist_bins],self.hist_range).flatten()
            hist_v = cv2.calcHist([hsv],[2],mascara,[self.hist_bins],self.hist_range).flatten()

            hist_h /= np.sum(hist_h)+1e-7
            hist_s /= np.sum(hist_s)+1e-7
            hist_v /= np.sum(hist_v)+1e-7

            momentos = cv2.moments(contorno)
            hu = cv2.HuMoments(momentos).flatten()
            hu = -np.sign(hu)*np.log10(np.abs(hu)+1e-7)

            features = np.concatenate((aplanada, hist_h,hist_s,hist_v,hu)) #Asignar imagen como ramo

            resultados.append((contorno,features))

        return resultados