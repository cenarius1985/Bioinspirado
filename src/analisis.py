import cv2
import numpy as np
import pandas as pd
import os
import ast
from src.configuracion import Config
from src.utils import segment_image, generate_single_channel_image

def run_visual_analysis(run_dir):
    """
    Recorre el directorio de resultados (run_dir), busca archivos Excel generados,
    y genera visualizaciones (matrices de imágenes segmentadas) para cada uno.
    """
    print(f"\n{'='*50}")
    print(f"Iniciando Análisis Visual en: {run_dir}")
    print(f"{'='*50}")

    # Recorrer recursivamente buscando archivos Excel
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.endswith(".xlsx") and file.startswith("Resultados_"):
                excel_path = os.path.join(root, file)
                print(f"Procesando: {excel_path}")
                try:
                    process_excel_results(excel_path, run_dir)
                except Exception as e:
                    print(f"Error procesando {file}: {e}")

def process_excel_results(excel_path, base_output_dir):
    # Cargar datos
    df = pd.read_excel(excel_path)
    
    # Identificar algoritmos presentes en el Excel basándose en columnas que terminan en '_Thresholds'
    # Ejemplo columna: 'TSO_Thresholds'
    algoritmos_presentes = []
    for col in df.columns:
        if col.endswith('_Thresholds'):
            algo_name = col.replace('_Thresholds', '')
            algoritmos_presentes.append(algo_name)
    
    if not algoritmos_presentes:
        print("  No se encontraron columnas de umbrales en el archivo.")
        return

    # Crear carpeta para guardar visualizaciones de este Excel
    # Usaremos el nombre del archivo (sin extensión) para crear una subcarpeta
    excel_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_vis_dir = os.path.join(os.path.dirname(excel_path), "Visualizacion")
    os.makedirs(output_vis_dir, exist_ok=True)

    # Lista para almacenar todas las imágenes procesadas y crear una gran matriz final
    # Estructura: Lista de filas (cada fila es una imagen original + sus segmentaciones)
    all_images_rows = []

    for index, row in df.iterrows():
        img_name = row['Imagen']
        img_path = os.path.join(Config.IMG_DIR, img_name)
        
        # Cargar Imagen Original
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"  Error: No se pudo cargar la imagen original {img_name}")
            continue
            
        # Redimensionar para visualización estándar (ej. 200x200)
        target_size = (200, 200)
        original_resized = cv2.resize(original_img, target_size)
        
        # Iniciar la fila de imágenes con la original
        row_images = [original_resized]
        
        # Procesar cada algoritmo
        for algo in algoritmos_presentes:
            thresh_col = f'{algo}_Thresholds'
            thresholds_str = row[thresh_col]
            
            try:
                # Convertir string "[1, 2, 3]" a lista [1, 2, 3]
                thresholds = ast.literal_eval(thresholds_str)
                
                # Segmentar imagen original (usando grayscale para segmentación)
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                
                # --- NUEVO MÉTODO DE VISUALIZACIÓN ---
                # Usar la misma lógica de "aplicar_umbrales" del notebook original
                # para generar una imagen segmentada visualmente distinguible (colores aleatorios)
                
                # Establecer semilla para consistencia de colores por algoritmo/umbral
                np.random.seed(42)
                
                imagen_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                
                # Ordenar umbrales por si acaso
                thresholds = sorted(thresholds)
                
                # Crear máscara acumulativa para ir pintando regiones
                # Pero la lógica del notebook original era:
                # for i, umbral in enumerate(umbrales):
                #    _, img_umbral = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
                #    color = ...
                #    imagen_color[img_umbral == 255] = color
                
                # Esta lógica del notebook "pinta encima" sucesivamente.
                # Si umbral[0]=50, pinta todo > 50.
                # Si umbral[1]=100, pinta todo > 100 (sobreescribiendo lo anterior).
                # Esto efectivamente colorea las regiones por capas.
                
                for umbral in thresholds:
                    _, img_umbral = cv2.threshold(gray_img, umbral, 255, cv2.THRESH_BINARY)
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    imagen_color[img_umbral == 255] = color
                
                img_seg_bgr = imagen_color
                # -------------------------------------
                
                # Redimensionar
                img_seg_resized = cv2.resize(img_seg_bgr, target_size)
                
                # Agregar texto con nombre del algoritmo
                cv2.putText(img_seg_resized, algo, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                row_images.append(img_seg_resized)
                
            except Exception as e:
                print(f"  Error procesando {algo} para {img_name}: {e}")
                # Agregar imagen negra en caso de error
                black_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                row_images.append(black_img)

        # Concatenar imágenes de la fila horizontalmente
        row_concat = np.hstack(row_images)
        all_images_rows.append(row_concat)
        
        # Guardar imagen individual de comparación (Original + Algos)
        output_filename = os.path.join(output_vis_dir, f"Comp_{img_name}")
        cv2.imwrite(output_filename, row_concat)

    # Crear Gran Matriz (Concatenar todas las filas verticalmente)
    if all_images_rows:
        try:
            full_matrix = np.vstack(all_images_rows)
            matrix_filename = os.path.join(output_vis_dir, f"Matriz_Completa_{excel_name}.png")
            cv2.imwrite(matrix_filename, full_matrix)
            print(f"  Matriz visual guardada en: {matrix_filename}")
        except Exception as e:
            print(f"  Error creando matriz completa: {e}")
