import numpy as np
import os

class Config:
    # Directorios
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR = os.path.join(BASE_DIR, 'img')
    
    # Parámetros del Algoritmo
    N = 30          # Número de agentes/población
    T = 100         # Número máximo de iteraciones
    
    # Listas para ejecución automática
    DIMENSIONS_LIST = [3, 4, 5, 6, 7] # Dimensiones a probar
    OBJ_FUNCTIONS_LIST = ['Kapur', 'Tsallis', 'Otsu'] # Funciones objetivo a probar
    
    # Valores por defecto (se sobrescriben en main.py durante la ejecución automática)
    DIM = 7         
    LB = 0          # Límite inferior
    UB = 255        # Límite superior
    EJECUCIONES = 30 # Número de ejecuciones por imagen
    
    # Algoritmos a ejecutar
    ALGORITHMS = ['TSO', 'HBA', 'BES', 'GWO', 'HHO', 'CSA']
    
    # Función Objetivo actual
    OBJ_FUNC = 'Kapur'

    # Configuración de salida
    @staticmethod
    def get_output_excel(obj_func, dim):
        return f'Resultados_{obj_func}_{dim}dim.xlsx'
    
    @staticmethod
    def get_thresholds_filename(algo_name, dim):
        return f'Thresholds_{algo_name}_{dim}dim.npy'
    
    @staticmethod
    def get_convergence_filename(dim):
        return f'Convergence_{dim}dim.npy'
