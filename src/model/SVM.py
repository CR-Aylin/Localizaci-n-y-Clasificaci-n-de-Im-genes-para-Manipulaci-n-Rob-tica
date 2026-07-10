import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):

        self.scaler = StandardScaler()

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=False
        )

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)

        X = self.scaler.fit_transform(X)

        self.model.fit(X, y)

    def predict(self, X):

        X = np.asarray(X, dtype=np.float32)

        X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_with_distance(self, x):

        x = np.asarray(x, dtype=np.float32).reshape(1, -1)

        x = self.scaler.transform(x)

        label = self.model.predict(x)[0]

        scores = self.model.decision_function(x)

        if scores.ndim == 1:
            confidence = abs(scores[0])
        else:
            confidence = np.max(np.abs(scores[0]))

        return label, float(confidence)


class SVM_PCA:

    def __init__(self,
                 n_components=50,
                 kernel='rbf',
                 C=1.0,
                 gamma='scale'):

        self.scaler = StandardScaler()

        self.n_components = n_components

        self.pca = None

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=False
        )

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)

        X = self.scaler.fit_transform(X)

        n_components = min(
            self.n_components,
            X.shape[0],
            X.shape[1]
        )

        self.pca = PCA(n_components=n_components)

        X = self.pca.fit_transform(X)

        self.model.fit(X, y)

    def predict(self, X):

        X = np.asarray(X, dtype=np.float32)

        X = self.scaler.transform(X)

        X = self.pca.transform(X)

        return self.model.predict(X)

    def predict_with_distance(self, x):

        x = np.asarray(x, dtype=np.float32).reshape(1, -1)

        x = self.scaler.transform(x)

        x = self.pca.transform(x)

        label = self.model.predict(x)[0]

        scores = self.model.decision_function(x)

        if scores.ndim == 1:
            confidence = abs(scores[0])
        else:
            confidence = np.max(np.abs(scores[0]))

        return label, float(confidence)


def sliding_window_localization_svm(
        board_image,
        model,
        extractor,
        window_size=(100, 100),
        step=20,
        confidence_threshold=0.5):

    h, w = board_image.shape[:2]

    detections = []

    for y in range(0, h - window_size[1] + 1, step):

        for x in range(0, w - window_size[0] + 1, step):

            window = board_image[
                y:y + window_size[1],
                x:x + window_size[0]
            ]

            # return_bbox=True: además del vector de features (que ya NO incluye
            # posición/tamaño crudos, para no confundir al clasificador), obtenemos
            # el bounding box real del objeto DENTRO de la ventana, para poder
            # dibujar un rectángulo ajustado a la forma en vez de un cuadrado fijo
            # de 100x100 centrado a ciegas.
            features, bbox_local = extractor.extract(window, return_bbox=True)

            features = np.asarray(features, dtype=np.float32)

            # Ignorar ventanas vacías
            if np.linalg.norm(features) < 1e-6:
                continue

            label, confidence = model.predict_with_distance(features)

            if confidence >= confidence_threshold:

                bbox_x, bbox_y, bbox_w, bbox_h = bbox_local

                # Convertir el bounding box de coordenadas LOCALES (dentro de la
                # ventana) a coordenadas GLOBALES (dentro de la imagen completa)
                x1 = x + bbox_x
                y1 = y + bbox_y
                x2 = x1 + bbox_w
                y2 = y1 + bbox_h

                detections.append({
                    "x": x + window_size[0] // 2,
                    "y": y + window_size[1] // 2,
                    "bbox": (x1, y1, x2, y2),
                    "class": label,
                    "confidence": confidence
                })

    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # margen de tolerancia: si dos cajas del mismo objeto grande quedaron a una
    # distancia menor al "step" del sliding window, igual se consideran parte
    # del mismo objeto (no hace falta que se toquen exactamente)
    final_detections = _fusionar_detecciones(detections, margin=step)

    return final_detections


def _rects_overlap(bbox_a, bbox_b, margin=0):
    """
    True si dos rectángulos (x1,y1,x2,y2) se solapan o están a una distancia
    <= margin uno del otro (expandiendo bbox_b por 'margin' antes de comparar).
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    bx1 -= margin
    by1 -= margin
    bx2 += margin
    by2 += margin

    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _fusionar_detecciones(detections, margin=20):
    """
    Fusiona detecciones de la MISMA clase en objetos completos usando componentes
    conexas: si la caja A se solapa (o está cerca, según 'margin') de la caja B,
    y B con C, entonces A, B y C se fusionan juntas en UNA sola caja aunque A y C
    no se toquen directamente entre sí. Esto es necesario para objetos más
    grandes que la ventana deslizante, que quedan fragmentados en varias cajas
    parciales repartidas por distintas zonas del mismo objeto.
    """
    n = len(detections)
    padre = list(range(n))

    def encontrar(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    def unir(i, j):
        ri, rj = encontrar(i), encontrar(j)
        if ri != rj:
            padre[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if detections[i]["class"] != detections[j]["class"]:
                continue
            if _rects_overlap(detections[i]["bbox"], detections[j]["bbox"], margin=margin):
                unir(i, j)

    grupos = {}
    for i in range(n):
        raiz = encontrar(i)
        grupos.setdefault(raiz, []).append(detections[i])

    resultado = []
    for grupo in grupos.values():
        x1 = min(d["bbox"][0] for d in grupo)
        y1 = min(d["bbox"][1] for d in grupo)
        x2 = max(d["bbox"][2] for d in grupo)
        y2 = max(d["bbox"][3] for d in grupo)

        confianza_maxima = max(d["confidence"] for d in grupo)
        clase = grupo[0]["class"]

        resultado.append({
            "class": clase,
            "confidence": confianza_maxima,
            "bbox": (x1, y1, x2, y2),
            # Posición final del objeto = centro de la caja fusionada
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2
        })

    return resultado