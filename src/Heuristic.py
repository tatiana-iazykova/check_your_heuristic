from src.Base import BaseSolver
from typing import Dict, Any


class BaseHeuristicSolver(BaseSolver):
   
    def __init__(self, path: str, config: Dict[str, Any], path_valid: str = None):
        super(BaseHeuristicSolver, self).__init__(path, path_valid)
    
    def check_heuristics(self):
        pass
