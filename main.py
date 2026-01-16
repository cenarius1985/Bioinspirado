import os
import cv2
import numpy as np
import pandas as pd
from src.configuracion import Config
from src.segmentacion import run_algorithm, run_algorithm_test
from src.utils import calculate_psnr, calculate_ssim, generate_single_channel_image, segment_image
from src.dicom_converter import ConvertidorDICOM, verificar_tipo_archivos

# Import objective functions
from src.funcion_objetivo_kapur import kapur_objective_function
from src.funcion_objetivo_tsallis import tsallis_objective_function
from src.funcion_objetivo_otsu import otsu_objective_function

from src.analisis import run_visual_analysis
from src.unir_resultado_excel import unir_resultados

def get_user_mode_selection():
    print("\n--- SELECCIÓN DE MODO ---")
    print("1. TEST (Rápido: 2 imágenes, pocas iteraciones, dims bajas)")
    print("2. PRODUCCION (Completo: Todas las imágenes, configuración full)")
    
    while True:
        try:
            selection = input("Seleccione una opción (1 o 2): ").strip()
            if selection == '1':
                return 'TEST'
            elif selection == '2':
                return 'PROD'
            else:
                print("Opción inválida. Por favor ingrese 1 o 2.")
        except Exception as e:
            print(f"Error en la entrada: {e}")

def main():
    # 0. Configurar Modo (Test vs Producción)
    execution_mode = get_user_mode_selection()
    Config.setup_mode(execution_mode)

    # 1. Setup Environment
    if not os.path.exists(Config.IMG_DIR):
        print(f"Error: Directory {Config.IMG_DIR} does not exist.")
        return

    # --- NUEVA LÓGICA DICOM ---
    # Verificar si hay archivos DICOM en la carpeta de imágenes
    tiene_png, tiene_dicom, count_png, count_dicom = verificar_tipo_archivos(Config.IMG_DIR)
    
    if tiene_dicom:
        print(f"\n{'='*50}")
        print(f"DETECTADOS {count_dicom} ARCHIVOS DICOM")
        print(f"{'='*50}")
        print("Iniciando conversión automática a PNG de 16-bits...")
        
        # Inicializar convertidor apuntando a la MISMA carpeta de origen para dejar los PNG ahí
        # Opcionalmente podríamos usar una subcarpeta, pero el usuario pidió "trabajar con todas las imagenes"
        # y que el análisis pase por todas. Si las ponemos en la misma carpeta, el flujo normal las tomará.
        convertidor = ConvertidorDICOM(Config.IMG_DIR, Config.IMG_DIR) 
        
        # Convertir todos los DICOM
        nuevos_pngs = convertidor.convertir_carpeta_completa(conservar_metadatos=True)
        
        print(f"Conversión completada. Se generaron {len(nuevos_pngs)} nuevas imágenes PNG.")
        print(f"{'='*50}\n")
    # --------------------------

    images_list = [f for f in os.listdir(Config.IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images_list:
        print("No images found.")
        return
        
    # Si estamos en modo TEST, limitamos a las 2 primeras imágenes
    if Config.MODE == 'TEST':
        print(f"MODO TEST ACTIVADO: Procesando solo las primeras 2 imágenes de {len(images_list)} disponibles.")
        images_list = images_list[:2]

    # Master Loop: Objective Functions
    for obj_func_name in Config.OBJ_FUNCTIONS_LIST:
        print(f"\n{'='*50}")
        print(f"Starting Analysis for Objective Function: {obj_func_name}")
        print(f"{'='*50}")
        
        # Select Objective Function
        if obj_func_name == 'Kapur':
            current_obj_func = kapur_objective_function
        elif obj_func_name == 'Tsallis':
            current_obj_func = tsallis_objective_function
        elif obj_func_name == 'Otsu':
            current_obj_func = otsu_objective_function
        else:
            print(f"Skipping unknown objective function: {obj_func_name}")
            continue

        # Master Loop: Dimensions
        for dim in Config.DIMENSIONS_LIST:
            print(f"\n{'-'*30}")
            print(f"Running for Dimension: {dim}")
            print(f"{'-'*30}")
            
            # Update Config dynamically for this run
            Config.DIM = dim
            Config.OBJ_FUNC = obj_func_name # Update current obj func name in Config too
            
            # Data Structures for Results (Reset for each Dim/ObjFunc combination)
            all_results = []
            thresholds_data = {algo: [] for algo in Config.ALGORITHMS}
            convergence_data = [] 

            # Loop over Images
            for img_idx, img_name in enumerate(images_list):
                print(f"Processing Image {img_idx + 1}/{len(images_list)}: {img_name}")
                
                img_path = os.path.join(Config.IMG_DIR, img_name)
                # Cargar imagen con cualquier profundidad (permite 16-bit, etc.)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                
                if image is None:
                    print(f"Error loading {img_name}")
                    continue

                # Preprocessing
                if image.shape[0] != 96 or image.shape[1] != 96:
                    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
                
                # Detectar y configurar límites automáticamente antes de cualquier procesamiento
                Config.set_bounds_from_image(image)
                
                # Normalización opcional (comentada para permitir detección automática real)
                # if image.dtype != np.uint8:
                #     image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Metrics storage for this image
                img_metrics = {algo: {'fitness': [], 'psnr': [], 'ssim': [], 'thresholds': [], 'convergence': []} for algo in Config.ALGORITHMS}
                
                # Loop Executions
                for execution in range(Config.EJECUCIONES):
                    print(f"  > Imagen {img_idx + 1}/{len(images_list)} | Ejecución {execution + 1}/{Config.EJECUCIONES}", end='\r')
                    
                    for algo in Config.ALGORITHMS:
                        # Run Algorithm
                        if Config.MODE == 'TEST':
                             best_fitness, best_thresholds, convergence = run_algorithm_test(algo, image, current_obj_func, Config)
                        else:
                             best_fitness, best_thresholds, convergence = run_algorithm(algo, image, current_obj_func, Config)
                        
                        # Calculate Metrics
                        img_seg = segment_image(image, best_thresholds)
                        img_res = generate_single_channel_image(img_seg)
                        psnr = calculate_psnr(image, img_res)
                        ssim_val = calculate_ssim(image, img_res)
                        
                        # Store
                        img_metrics[algo]['fitness'].append(best_fitness)
                        img_metrics[algo]['psnr'].append(psnr)
                        img_metrics[algo]['ssim'].append(ssim_val)
                        img_metrics[algo]['thresholds'].append(best_thresholds)
                        img_metrics[algo]['convergence'].append(convergence)
                    
                # print() # Newline after executions
                
                # Aggregate Metrics for DataFrame
                row = {'Imagen': img_name}
                for algo in Config.ALGORITHMS:
                    # Calculate mean metrics
                    row[f'{algo}_Fitness'] = np.mean(img_metrics[algo]['fitness'])
                    row[f'{algo}_PSNR'] = np.mean(img_metrics[algo]['psnr'])
                    row[f'{algo}_SSIM'] = np.mean(img_metrics[algo]['ssim'])
                    
                    # Find best thresholds (Max Fitness)
                    best_idx = np.argmax(img_metrics[algo]['fitness'])
                    best_run_thresholds = img_metrics[algo]['thresholds'][best_idx]
                    row[f'{algo}_Thresholds'] = str(best_run_thresholds) # Save as string for Excel

                    # Append thresholds to global list
                    thresholds_data[algo].append(img_metrics[algo]['thresholds'])
                
                all_results.append(row)
                
                # Convergence Structure
                convergencia_img = []
                for i in range(Config.EJECUCIONES):
                    exec_convs = []
                    for algo in Config.ALGORITHMS:
                        exec_convs.append(img_metrics[algo]['convergence'][i])
                    convergencia_img.append(exec_convs)
                convergence_data.append(convergencia_img)

            # Save Results for this Dim/ObjFunc
            # DataFrame
            df_metrics = pd.DataFrame(all_results)
            output_excel = Config.get_output_excel(obj_func_name, dim)
            df_metrics.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Numpy Arrays
            for algo in Config.ALGORITHMS:
                filename = Config.get_thresholds_filename(algo, dim)
                np.save(filename, thresholds_data[algo])
                
            conv_filename = Config.get_convergence_filename(dim)
            np.save(conv_filename, convergence_data)
            print(f"Saved numpy arrays for Dim {dim}")

    print("\nAll Analysis Complete.")
    
    # 2. Run Visual Analysis
    run_visual_analysis(Config.RUN_DIR)

    # 3. Consolidar Resultados
    unir_resultados(Config.RUN_DIR)

if __name__ == "__main__":
    main()
