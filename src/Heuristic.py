from Base import BaseSolver
import pandas as pd


class BaseHeuristicSolver(BaseSolver):
   
    def __init__(self, path: str, path_valid: str = None):
        super(BaseHeuristicSolver, self).__init__(path, path_valid)
    
    def check_heuristics(self):
        pass

    def check_baselines(self):
        self.all_methods()




