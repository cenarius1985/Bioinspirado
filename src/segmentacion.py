import numpy as np
import cv2
from mealpy.swarm_based import TSO, HBA, BES, GWO, HHO, CSA
from mealpy import FloatVar
from src.utils import segment_image, generate_single_channel_image, calculate_psnr, calculate_ssim

# --- Generic Runner ---

def run_algorithm(algo_name, image, obj_func, config):
    """
    Runs a specified algorithm on an image with a given objective function.
    
    Args:
        algo_name (str): Name of the algorithm (TSO, HBA, BES, GWO, HHO, CSA).
        image (np.ndarray): Image to segment.
        obj_func (callable): Objective function func(thresholds, image) -> scalar (to be maximized).
        config (Config class): Configuration with N, T, DIM, LB, UB.
        
    Returns:
        tuple: (Best_Fitness, Best_Thresholds, Convergence_List)
    """
    
    N = config.N
    T = config.T
    dim = config.DIM
    lb = np.full(dim, config.LB)
    ub = np.full(dim, config.UB)
    
    # Mealpy Algorithms
    # Mealpy minimizes, so we negate the objective function (which we assume maximizes)
    def mealpy_obj_wrapper(solution):
        return -obj_func(solution, image)
        
    bounds = FloatVar(lb=lb, ub=ub, name="thresholds")
    problem_dict = {
        "obj_func": mealpy_obj_wrapper,
        "bounds": bounds,
        "minmax": "min",
        "log_to": None,
    }
    
    model = None
    if algo_name == 'TSO':
        model = TSO.OriginalTSO(epoch=T, pop_size=N)
    elif algo_name == 'HBA':
        model = HBA.OriginalHBA(epoch=T, pop_size=N)
    elif algo_name == 'BES':
        model = BES.OriginalBES(epoch=T, pop_size=N)
    elif algo_name == 'GWO':
        model = GWO.OriginalGWO(epoch=T, pop_size=N)
    elif algo_name == 'HHO':
        model = HHO.OriginalHHO(epoch=T, pop_size=N)
    elif algo_name == 'CSA':
        model = CSA.OriginalCSA(epoch=T, pop_size=N)
    else:
        raise ValueError(f"Algorithm {algo_name} not supported or not available in mealpy.")
        
    g_best = model.solve(problem_dict)
    
    # Return results (Fitness negated back to positive, thresholds sorted)
    best_fitness = -g_best.target.fitness
    best_thresholds = sorted(g_best.solution)
    convergence = [-x for x in model.history.list_global_best_fit]
    
    return best_fitness, best_thresholds, convergence
