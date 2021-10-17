from Base import BaseSolver
import pandas as pd


class BaseHeuristicSolver(BaseSolver):
   
    def __init__(self, path: str, path_valid: str = None):
        super(BaseHeuristicSolver, self).__init__(path, path_valid)

    def preprocess(self, columns: pd.Series):
        for column in columns:
            self.train[f"{column}_tokenized"] = self.train[column].apply(self.clean_text)
            self.valid[f"{column}_tokenized"] = self.valid[column].apply(self.clean_text)

    def clean_text(self, text: str) -> str:
      return text
    
    def check_heuristics(self):
        pass

    def check_baselines(self):
        self.all_methods()




