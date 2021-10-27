from src.Base import BaseSolver
from typing import Dict, Any
from src.Dataset import Dataset

class BaseHeuristicSolver(BaseSolver):
   
    def __init__(self, config: Dict[str, Any], dataset: Dataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
    
    def check_heuristics(self):
        pass
