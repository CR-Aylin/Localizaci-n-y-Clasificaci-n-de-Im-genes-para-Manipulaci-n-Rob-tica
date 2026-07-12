import os
import time
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, f1_score)
import seaborn as sns

# Importar tus módulos propios
from src.model.Extraccion_caracteristicas import features_extractor
from src.model.KNN import KNN, sliding_window_localization

# ============================================================
# 1. FUNCIÓN PARA CARGAR EL DATASET
# ============================================================
def cargar_dataset(ruta_base, extractor):
    X = []
    y = []
    clases = sorted(os.listdir(ruta_base))
    print(f"Clases detectadas: {clases}")
    
    for clase in clases:
        ruta_clase = os.path.join(ruta_base, clase)
        if not os.path.isdir(ruta_clase):
            continue
            
        archivos = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Procesando {clase}: {len(archivos)} imágenes")
        
        for archivo in archivos:
            ruta_img = os.path.join(ruta_clase, archivo)
            img = cv2.imread(ruta_img)
            if img is None: continue
                
            features = extractor.extract(img)
            if np.sum(features) == 0 or np.isnan(features).any(): continue
                
            X.append(features)
            y.append(clase)
    
    return np.array(X), np.array(y), sorted(list(set(y)))

def calcular_dist_threshold(X_ref, model, percentil=95, factor=1.5):
    distancias = [model.predict_with_distance(x)[1] for x in X_ref]
    base = np.percentile(distancias, percentil)
    return base * factor

# ============================================================
# 2. FUNCIÓN DE EVALUACIÓN (MÉTRICAS)
# ============================================================
def evaluar_knn(X_train, y_train, X_test, y_test, clases, 
                usar_pca=False, n_components=15, k=5, 
                guardar_modelos=False, ruta_guardado="modelos_knn"):
    
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN KNN {'CON PCA' if usar_pca else 'SIN PCA'} (k={k})")
    print(f"{'='*60}")
    
    # --- ESCALADO ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- PCA ---
    pca = None
    if usar_pca:
        pca = PCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"PCA: {X_train.shape[1]} -> {n_components} componentes")
    
    # --- ENTRENAMIENTO ---
    knn = KNN(k=k)
    t0 = time.time()
    knn.fit(X_train_scaled, y_train)
    tiempo_entrenamiento = time.time() - t0
    
    # --- INFERENCIA ---
    t0 = time.time()
    predicciones = knn.predict(X_test_scaled)
    tiempo_inferencia = time.time() - t0
    
    # --- MÉTRICAS ---
    accuracy = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(y_test, predicciones, average='macro', labels=clases, zero_division=0)
    
    print(f"Accuracy: {accuracy*100:.2f}% | F1: {f1_macro:.4f}")
    print(classification_report(y_test, predicciones, labels=clases, target_names=clases, zero_division=0))
    
    # --- MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_test, predicciones, labels=clases)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz Confusión KNN {'(PCA)' if usar_pca else '(Sin PCA)'}")
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout(); plt.show()
    
    dist_threshold = calcular_dist_threshold(X_train_scaled, knn)
    print(f"Umbral de distancia calibrado: {dist_threshold:.4f}")

    # --- GUARDAR MODELOS ---
    if guardar_modelos:
        os.makedirs(ruta_guardado, exist_ok=True)
        sufijo = "pca" if usar_pca else "sin_pca"
        joblib.dump(scaler, os.path.join(ruta_guardado, f"scaler_{sufijo}.pkl"))
        joblib.dump(knn, os.path.join(ruta_guardado, f"knn_{sufijo}.pkl"))
        if pca is not None:
            joblib.dump(pca, os.path.join(ruta_guardado, f"pca_{sufijo}.pkl"))
        print(f"Modelos guardados en: {ruta_guardado}/")
    
    return {'knn': knn, 'scaler': scaler, 'pca': pca, 'accuracy': accuracy, 'f1': f1_macro, 'dist_threshold': dist_threshold}

# ============================================================
# 3. FUNCIÓN DE PRUEBA VISUAL EN TABLERO
# ============================================================
def probar_en_tablero(ruta_imagen, knn, scaler, pca, extractor, window_size=(100, 100), dist_threshold=10):
    """
    Aplica la ventana deslizante, agrupa detecciones por clase y retorna el bounding box por clase.
    """
    print(f"\nAnalizando imagen: {ruta_imagen}")
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("Error al cargar la imagen.")
        return {}

    detections_raw = sliding_window_localization(
        board_image=img,
        model=knn,
        extractor=extractor,
        scaler=scaler,
        pca=pca,
        window_size=window_size,
        step=20,
        dist_threshold=dist_threshold
    )
    
    print(f"--- Resultados de Detección ({len(detections_raw)} objetos encontrados) ---")
    
    detecciones_por_clase = {}
    
    if not detections_raw:
        print("No se detectaron objetos con la confianza suficiente.")
    else:
        for i, det in enumerate(detections_raw):
            x, y = int(det['x']), int(det['y'])
            label = det['class']
            conf = det['confidence']
            
            print(f"  Deteccion #{i+1}: Clase='{label}' | Coordenadas(x,y)=({x}, {y}) | Confianza={conf:.4f}")
            
            if label not in detecciones_por_clase:
                detecciones_por_clase[label] = []
            detecciones_por_clase[label].append((x, y))
            
    print("-------------------------------------------------------")

    # Dibujar resultados (Visualización)
    img_result = img.copy()
    resultados_retorno = {} 

    for label, puntos in detecciones_por_clase.items():
        if not puntos:
            continue
        
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Ajustar para centrar el box alrededor de los puntos detectados (centros de ventanas)
        # Opcional: Añadir un margen basado en la mitad del tamaño de la ventana
        margin_w = window_size[0] // 2
        margin_h = window_size[1] // 2
        
        box_x1 = max(0, min_x - margin_w)
        box_y1 = max(0, min_y - margin_h)
        box_x2 = min(img.shape[1], max_x + margin_w)
        box_y2 = min(img.shape[0], max_y + margin_h)
        
        # Guardar en resultados de retorno
        resultados_retorno[label] = {
            'clase': label,
            'coordenadas_cuadrado': ((box_x1, box_y1), (box_x2, box_y2)), # Esquina sup-izq e inf-der
            'puntos_detectados': puntos
        }
        
        # Dibujar el rectángulo agrupado
        color = (0, 255, 0) if label == "Class_1" else (255, 0, 0)
        cv2.rectangle(img_result, (box_x1, box_y1), (box_x2, box_y2), color, 3)
        cv2.putText(img_result, f"{label} ({len(puntos)} objs)", (box_x1, box_y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecciones Agrupadas por Clase")
    plt.axis('off')
    plt.show()

    # nuevo: Imprimir resumen de cuadrados generados
    print("\n=== CUADRADOS GENERADOS POR CLASE ===")
    for label, info in resultados_retorno.items():
        coords = info['coordenadas_cuadrado']
        print(f"Clase: {label} | Cuadrado: TopLeft{coords[0]} - BottomRight{coords[1]}")

    return resultados_retorno 

def ejecutar_knn_con_pca(RUTA_TABLERO):
    RUTA_DATASET = ("dataset\\Entrenamiento")
    
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    K_VECINOS = 5
    N_COMPONENTES_PCA = 15
    
    print("Cargando dataset...")
    extractor = features_extractor(hist_bins=16, debug=False)
    X, y, clases = cargar_dataset(RUTA_DATASET, extractor)
    
    conteo = Counter(y)
    for c, n in conteo.items():
        if n < 2: raise ValueError(f"Clase {c} tiene solo {n} muestras. Añade más imágenes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    res_con_pca = evaluar_knn(X_train, y_train, X_test, y_test, clases, usar_pca=True, n_components=N_COMPONENTES_PCA, k=K_VECINOS, guardar_modelos=True)
    
    detecciones_agrupadas = probar_en_tablero(
        ruta_imagen=RUTA_TABLERO,
        knn=res_con_pca['knn'],
        scaler=res_con_pca['scaler'],
        pca=res_con_pca['pca'],
        extractor=extractor
        dist_threshold=res_con_pca['dist_threshold']
    )
    
    return detecciones_agrupadas

def ejecutar_knn_sin_pca(RUTA_TABLERO):
    RUTA_DATASET = ("dataset\\Entrenamiento")

    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    K_VECINOS = 5
    N_COMPONENTES_PCA = 15
    
    print("Cargando dataset...")
    extractor = features_extractor(hist_bins=16, debug=False)
    X, y, clases = cargar_dataset(RUTA_DATASET, extractor)
    
    conteo = Counter(y)
    for c, n in conteo.items():
        if n < 2: raise ValueError(f"Clase {c} tiene solo {n} muestras. Añade más imágenes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    res_sin_pca = evaluar_knn(X_train, y_train, X_test, y_test, clases, usar_pca=False, k=K_VECINOS, guardar_modelos=True)

    detecciones_agrupadas = probar_en_tablero(
        ruta_imagen=RUTA_TABLERO,
        knn=res_sin_pca['knn'],
        scaler=res_sin_pca['scaler'],
        pca=res_sin_pca['pca'],
        extractor=extractor
        dist_threshold=res_sin_pca['dist_threshold']
    )
    
    return detecciones_agrupadas

# ============================================================
# 4. SCRIPT PRINCIPAL
# ============================================================
if __name__ == "__main__":
    RUTA_DATASET = ("dataset\\Entrenamiento")
    RUTA_TABLERO = ("dataset\\Test\WIN_20260702_17_07_52_Pro.jpg")
    
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    K_VECINOS = 5
    N_COMPONENTES_PCA = 15
    
    print("Cargando dataset...")
    extractor = features_extractor(hist_bins=16, debug=False)
    X, y, clases = cargar_dataset(RUTA_DATASET, extractor)
    
    conteo = Counter(y)
    for c, n in conteo.items():
        if n < 2: raise ValueError(f"Clase {c} tiene solo {n} muestras. Añade más imágenes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    res_sin_pca = evaluar_knn(X_train, y_train, X_test, y_test, clases, usar_pca=False, k=K_VECINOS, guardar_modelos=True)
    res_con_pca = evaluar_knn(X_train, y_train, X_test, y_test, clases, usar_pca=True, n_components=N_COMPONENTES_PCA, k=K_VECINOS, guardar_modelos=True)
    
    mejor_modelo = res_sin_pca if res_sin_pca['accuracy'] >= res_con_pca['accuracy'] else res_con_pca
    tipo_modelo = "SIN PCA" if res_sin_pca['accuracy'] >= res_con_pca['accuracy'] else "CON PCA"
    
    print(f"\nUsando modelo {tipo_modelo} para la prueba visual (Accuracy: {mejor_modelo['accuracy']*100:.2f}%)")
    
    resultados_finales = probar_en_tablero(
        ruta_imagen=RUTA_TABLERO,
        knn=mejor_modelo['knn'],
        scaler=mejor_modelo['scaler'],
        pca=mejor_modelo['pca'],
        extractor=extractor
        dist_threshold=mejor_modelo['dist_threshold']
    )