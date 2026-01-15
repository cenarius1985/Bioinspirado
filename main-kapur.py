import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio src al path para poder importar el módulo
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

try:
    from segmentacion_kapur import HHO, segment_image, generate_single_channel_image
except ImportError as e:
    print(f"Error al importar el módulo: {e}")
    sys.exit(1)

def main():
    # Directorio de imágenes
    img_dir = os.path.join(current_dir, 'img')
    if not os.path.exists(img_dir):
        print(f"El directorio {img_dir} no existe.")
        return

    # Obtener lista de imágenes
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print("No se encontraron imágenes en la carpeta img.")
        return

    # Seleccionar la primera imagen para la demostración
    # Puedes cambiar esto para iterar sobre todas las imágenes
    img_name = images[0]
    img_path = os.path.join(img_dir, img_name)
    print(f"Procesando imagen: {img_path}")
    
    # Leer imagen en escala de grises
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"No se pudo cargar la imagen {img_path}")
        return

    # Parámetros del algoritmo
    N = 30   # Número de agentes de búsqueda
    T = 100  # Número máximo de iteraciones
    dim = 4  # Dimensión (número de umbrales)
    lb = 0   # Límite inferior
    ub = 255 # Límite superior
    
    # Inicializar el algoritmo (HHO en este caso)
    print("Inicializando HHO (Harris Hawks Optimization)...")
    # Nota: Asegúrate de que los argumentos coincidan con el __init__ de HHO
    # HHO(image, N, T, lb, ub, dim)
    optimizer = HHO(image, N, T, lb, ub, dim)
    
    print("Optimizando...")
    best_fitness, best_thresholds, convergence = optimizer.optimize()
    
    print(f"Mejor Aptitud (Fitness): {best_fitness}")
    print(f"Mejores Umbrales: {best_thresholds}")
    
    # Segmentar la imagen con los mejores umbrales encontrados
    segmented_layers = segment_image(image, best_thresholds)
    result_image = generate_single_channel_image(segmented_layers)
    
    # Mostrar resultados
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result_image, cmap='gray')
    plt.title(f'Segmentada (Kapur, HHO, k={dim})')
    plt.axis('off')
    
    plt.tight_layout()
    print("Mostrando resultado...")
    plt.show()

if __name__ == "__main__":
    main()
