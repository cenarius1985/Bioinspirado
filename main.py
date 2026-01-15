import os
import cv2
import numpy as np
import pandas as pd
from src.configuracion import Config
from src.segmentacion import run_algorithm
from src.utils import segment_image, generate_single_channel_image, calculate_psnr, calculate_ssim

# Import objective functions
from src.funcion_obtjetivo_Kapur import kapur_objective_function
from src.funcion_objetivo_Tsallis import tsallis_objective_function
from src.funcion_objetivo_otsu import otsu_objective_function

def main():
    # 1. Setup Environment
    if not os.path.exists(Config.IMG_DIR):
        print(f"Error: Directory {Config.IMG_DIR} does not exist.")
        return

    images_list = [f for f in os.listdir(Config.IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images_list:
        print("No images found.")
        return

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
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error loading {img_name}")
                    continue

                # Preprocessing
                if image.shape[0] != 96 or image.shape[1] != 96:
                    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Metrics storage for this image
                img_metrics = {algo: {'fitness': [], 'psnr': [], 'ssim': [], 'thresholds': [], 'convergence': []} for algo in Config.ALGORITHMS}
                
                # Loop Executions
                for execution in range(Config.EJECUCIONES):
                    # print(f"  Execution {execution + 1}/{Config.EJECUCIONES}", end='\r')
                    
                    for algo in Config.ALGORITHMS:
                        # Run Algorithm
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
                    row[f'{algo}_Fitness'] = np.mean(img_metrics[algo]['fitness'])
                    row[f'{algo}_PSNR'] = np.mean(img_metrics[algo]['psnr'])
                    row[f'{algo}_SSIM'] = np.mean(img_metrics[algo]['ssim'])
                    
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

if __name__ == "__main__":
    main()
