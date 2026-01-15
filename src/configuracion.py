import numpy as np
import os
import datetime

class Config:
    # Directorios
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR = os.path.join(BASE_DIR, 'img')
    OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'Resultados') # Carpeta base para salidas
    RUN_DIR = None # Se definirá al configurar el modo

    # Parámetros del Algoritmo
    N = 30          # Número de agentes/población
    T = 100         # Número máximo de iteraciones
    
    # Listas para ejecución automática
    DIMENSIONS_LIST = [5, 6, 7, 8, 9] # Dimensiones a probar
    OBJ_FUNCTIONS_LIST = ['Kapur', 'Tsallis', 'Otsu'] # Funciones objetivo a probar
    
    # Valores por defecto (se sobrescriben en main.py durante la ejecución automática)
    DIM = 7         
    LB = 0          # Límite inferior (Dinámico)
    UB = 255        # Límite superior (Dinámico)
    EJECUCIONES = 30 # Número de ejecuciones por imagen

    @classmethod
    def set_bounds_from_image(cls, image):
        """
        Determina automáticamente LB y UB basándose en el tipo de datos de la imagen.
        """
        if image.dtype == np.uint8:
            cls.LB = 0
            cls.UB = 255
        elif image.dtype == np.uint16:
            cls.LB = 0
            cls.UB = 65535
        elif np.issubdtype(image.dtype, np.floating):
            # Asumimos rango normalizado 0.0-1.0 o detectamos max
            cls.LB = 0.0
            cls.UB = 1.0 if image.max() <= 1.0 else 255.0
        else:
            # Fallback para otros tipos
            cls.LB = float(image.min())
            cls.UB = float(image.max())
        
        # print(f"  -> Configurados límites dinámicos: [{cls.LB}, {cls.UB}] para dtype={image.dtype}")

    # Algoritmos a ejecutar
    ALGORITHMS = ['TSO', 'HBA', 'BES', 'GWO', 'HHO', 'CSA', 'WOA', 'SCSO']
    
    # Modo de ejecución
    MODE = 'PROD' # 'TEST' o 'PROD'

    @classmethod
    def setup_mode(cls, mode='PROD'):
        cls.MODE = mode
        
        # Configurar directorio de salida único para esta ejecución
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.RUN_DIR = os.path.join(cls.OUTPUT_BASE_DIR, f"{timestamp}_{mode}")
        os.makedirs(cls.RUN_DIR, exist_ok=True)
        print(f">>> Directorio de resultados creado: {cls.RUN_DIR} <<<")

        if mode == 'TEST':
            print(">>> CONFIGURANDO MODO TEST (Rápido) <<<")
            cls.N = 10              # Población reducida
            cls.T = 10              # Pocas iteraciones
            cls.DIMENSIONS_LIST = [3, 4] # Solo dimensiones bajas
            cls.EJECUCIONES = 2     # Pocas ejecuciones
        else:
            print(">>> CONFIGURANDO MODO PRODUCCION (Completo) <<<")
            cls.N = 30
            cls.T = 100
            cls.DIMENSIONS_LIST = [5, 6, 7, 8, 9]
            cls.EJECUCIONES = 30

    # Función Objetivo actual
    OBJ_FUNC = 'Kapur'

    # Configuración de salida
    @classmethod
    def get_output_excel(cls, obj_func, dim):
        # Estructura: RUN_DIR / ObjFunc / {dim}D / Resultados.xlsx
        folder = os.path.join(cls.RUN_DIR, obj_func, f"{dim}D")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f'Resultados_{obj_func}_{dim}dim.xlsx')
    
    @classmethod
    def get_thresholds_filename(cls, algo_name, dim):
        # Estructura: RUN_DIR / ObjFunc / {dim}D / Thresholds / {algo}.npy
        folder = os.path.join(cls.RUN_DIR, cls.OBJ_FUNC, f"{dim}D", "Thresholds")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f'Thresholds_{algo_name}_{dim}dim.npy')
    
    @classmethod
    def get_convergence_filename(cls, dim):
        # Estructura: RUN_DIR / ObjFunc / {dim}D / Convergence / Convergence.npy
        folder = os.path.join(cls.RUN_DIR, cls.OBJ_FUNC, f"{dim}D", "Convergence")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f'Convergence_{dim}dim.npy')
