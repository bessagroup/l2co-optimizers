from .parameters import Parameters
import numpy as np
# EDIT (l2co-optimizers): IOHexperimenter is not installed; a stub stands in
# for the class, which the code only compares types against.
class IOH_function:  # was: from IOHexperimenter import IOH_function
    pass
from abc import abstractmethod

class auc_func(IOH_function):
    def __init__(self, *args, **kwargs):
        budget = kwargs.pop('budget')
        super().__init__(*args, **kwargs)
        self.auc = budget
        #         print(f" AUC initialized as {self.auc}")
        self.budget = budget
        powers = np.round(np.linspace(2, -8, 51), decimals=1)
        self.target_values = np.power([10] * 51, powers)

    def __call__(self, x):
        if self.evaluations >= self.budget:
            return np.infty
        y = self.f.evaluate(x)
        if self.y_comparison(y, self.yopt):
            self.yopt = y
            self.xopt = x
        self.auc -= sum(self.best_so_far_precision > self.target_values) / 51
        if self.logger is not None:
            self.logger.process_evaluation(self.f.loggerCOCOInfo())
        return y

    def reset(self):
        self.auc = self.budget
        super().reset()
        
class Algorithm:
    """ Generic algorithm class.

    Parameters:
    --------------
    func : object, callable
        The function to be optimized.
    
    Attributes:
    --------------
    dim : int
        The number of dimenions, problem variables.

    budget : int
        The available budget for function evaluations.

    popsize : int
        The number of individuals in the population.

    """

    def __init__(self, func, **kwargs):
        seed = kwargs.get("seed", None)
        self.verbose = kwargs.get("verbose", False)
        if type(seed) == tuple:
            np.random.set_state(seed)
        elif type(seed) == int:
            np.random.seed(seed)
        self.uses_old_ioh = (type(func) == IOH_function or type(func) == auc_func)

        if self.uses_old_ioh:
            self.dim = func.number_of_variables
        else:
            self.dim = func.meta_data.n_variables
        self.func = func
        self.budget = 0
        self.popsize = 1
        self.x_hist = []
        self.f_hist = []
#         self.verbose = verbose
    
    def stop(self):
        return False
    
    def get_params(self, parameters):
        # Update the essential parameters
#         xhist = parameters.get('x_hist', []).flatten()
#         xhist.append(self.x_hist.flatten())
#         parameters['x_hist'] = np.reshape(xhist, (self.dim, -1))
#         fhist = parameters.get('f_hist', [])
#         fhist.append(self.f_hist)
#         parameters['f_hist'] = fhist
        if self.uses_old_ioh:
            parameters['x_opt'] = self.func.best_so_far_variables
            parameters['f_opt'] = self.func.best_so_far_precision
        else:
            parameters['x_opt'] = self.func.state.current_best.x
            parameters['f_opt'] = self.func.state.current_best.y
            
        parameters['x_hist'] = self.x_hist
        parameters['f_hist'] = self.f_hist
        return parameters
    
    def set_params(self, parameters):
        self.x_hist = parameters.get('x_hist', [])
        self.f_hist = parameters.get('f_hist', [])
        

    def set_stopping_criteria(self, stopping_criteria):
        self.stop = stopping_criteria

    @abstractmethod
    def run(self):
        pass
    
        return self
    
    @classmethod
    def get_hyperparams(cls):
        return None
    
    def set_hyperparams(self, hyperparam_dict):
        for k,v in hyperparam_dict.items():
            setattr(self, k, v)

class Switching_algorithm:
    """ Class to implement the switch between algorithms.

    Parameters:
    ---------------
    func : object, callable
        The function to be optimized.

    budget : int
        The budget that is available for function evaluations.

    data : dict
        A dictionary with tau as key, algorithm as value


    Attibutes:
    ---------------
    None

    """

    def __init__(self, func, config, verbose = False, seed = None, **kwargs):
#         assert(type(func) == IOH_function or type(func) == auc_func)
        self.func = func
        self.config = config
        self.uses_old_ioh = (type(func) == IOH_function or type(func) == auc_func)
        if self.uses_old_ioh:
            self.budget = kwargs.get('budget_factor', 10000) * func.number_of_variables
        else:
            self.budget = kwargs.get('budget_factor', 10000) * func.meta_data.n_variables
#         self.store_during_run = kwargs.get('store_parameters', False)
        self.fixed_target = kwargs.get('fixed_target', True)
        self.verbose = verbose
        np.random.seed(seed)
        
    def __call__(self):
        """ Runs the switch routine. 
        
        Returns:
        -----------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.
        """

        # Initialize parameters
        if self.fixed_target:
            params = {'budget' : self.budget}
        else:
            params = {}
#         with open("log.txt", 'a') as f:
#             f.write(f"{self.config} \n")
        # Iterate through dictionary
        for tau, alg_config in self.config.items():
#             with open("log.txt", 'a') as f:
#                 f.write(f"{self.config} ||| Starting {alg_config} \n")
            alg = alg_config['class']
            algorithm = alg(self.func, verbose = self.verbose, seed = np.random.get_state())
            # Set algorithm parameters based on parameters object
            algorithm.set_hyperparams(alg_config['hyperparams'])
            
            # Define stopping criteria
            func = self.func
            if self.fixed_target:
#                 print("FT")
                budget = self.budget
            else: 
#                 print("FB")
                budget = tau
                params['budget'] = tau
                
                
            algorithm.set_params(params)
            
#             algorithm.set_hyperparams(alg_config['hyperparams'])


                
            if self.uses_old_ioh and self.fixed_target:
#                 print(f"Stopping criteria: old FT {budget}")
                def stopping_criteria():
                    return (func.best_so_far_precision) <= tau or (
                        func.evaluations >= budget)
            elif not self.fixed_target and not self.uses_old_ioh:
#                 print(f"Stopping criteria: Budget {budget}")
                def stopping_criteria():
                    return func.state.evaluations >= budget
            else: #Temporary
#                 print(f"Stopping criteria: Other {budget}")
                def stopping_criteria():
                    return func.state.evaluations >= budget
            algorithm.set_stopping_criteria(stopping_criteria)

            # Run algorithm and extract parameters
            algorithm.run()
            params = algorithm.get_params(params)
            
#             if self.store_during_run:
                
            
            if self.uses_old_ioh:
                if self.fixed_target and self.func.evaluations >= budget:
                    break
            else:
                if self.fixed_target and self.func.state.evaluations >= budget:
                    break
#         with open("log.txt", 'a') as f:
#             f.write(f"FINISHED {self.config} \n")
        # Return best point and fitness value found by last run algorithm
#         return (self.func.best_so_far_variables, self.func.best_so_far_precision, params) 
        return params
