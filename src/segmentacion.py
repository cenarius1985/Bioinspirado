import numpy as np
from mealpy.swarm_based import TSO, HBA, BES, GWO, HHO, CSA, WOA, SCSO
from mealpy import FloatVar

# --- Internal Core Execution ---

def _execute_mealpy(algo_name, image, obj_func, N, T, dim, lb_val, ub_val):
    """
    Core function to execute Mealpy algorithms with explicit parameters.
    """
    lb = np.full(dim, lb_val)
    ub = np.full(dim, ub_val)
    
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
    elif algo_name == 'WOA':
        model = WOA.OriginalWOA(epoch=T, pop_size=N)
    elif algo_name == 'SCSO':
        model = SCSO.OriginalSCSO(epoch=T, pop_size=N)
    else:
        raise ValueError(f"Algorithm {algo_name} not supported or not available in mealpy.")
        
    g_best = model.solve(problem_dict)
    
    # Return results (Fitness negated back to positive, thresholds sorted and converted to standard floats)
    best_fitness = -g_best.target.fitness
    best_thresholds = [float(x) for x in sorted(g_best.solution)]
    convergence = [-x for x in model.history.list_global_best_fit]
    
    return best_fitness, best_thresholds, convergence


# --- Public Runners ---

def run_algorithm(algo_name, image, obj_func, config):
    """
    PROD Runner: Uses configuration from Config object.
    """
    return _execute_mealpy(
        algo_name, 
        image, 
        obj_func, 
        config.N, 
        config.T, 
        config.DIM, 
        config.LB, 
        config.UB
    )

def run_algorithm_test(algo_name, image, obj_func, config):
    """
    TEST Runner: Uses hardcoded fast parameters for testing.
    Ignores Config.N and Config.T, but respects DIM/LB/UB from context.
    """
    # Configuración Específica para Test (Rápida)
    TEST_N = 10      # Población muy pequeña
    TEST_T = 5       # Muy pocas iteraciones
    
    # print(f"  [TEST RUN] Algo: {algo_name} | N={TEST_N} | T={TEST_T}")
    
    return _execute_mealpy(
        algo_name, 
        image, 
        obj_func, 
        TEST_N, 
        TEST_T, 
        config.DIM, 
        config.LB, 
        config.UB
    )
