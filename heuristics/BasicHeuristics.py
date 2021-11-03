from src.Dataset import Dataset
from src.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, Union
import pandas as pd
from src.utils import get_target_list
from collections import Counter


class BasicHeuristics(BaseHeuristicSolver):

    def __init__(self, config: Dict[str, Any], dataset: Dataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
        self.column_1 = config["column_name1"]
        self.column_2 = config["column_name2"]
        self.data = self.valid if self.valid is not None else self.train
        self.target_list = get_target_list(self.data[self.target_name])

    @staticmethod
    def check_substring(text1: str, text2: str):
        text1 = text1.lower()
        return text1 in text2.lower()

    def check_heuristics(self) -> Dict[str, Dict[str, Union[str, Dict[str, Dict[str, str]]]]]:
        length = len(self.data)
        result = {"check_substring": {}}
        substring_heuristic = self.data.apply(
            lambda row: self.check_substring(
                text1=row[self.column_1],
                text2=row[self.column_2]
            ),
            axis=1
        )
        result["check_substring"]["coverage"] = substring_heuristic.sum() / length
        result["check_substring"]["correlation"] = self._get_correlation(
            heuristic_result=substring_heuristic
        )
        return print(result)

    def _get_correlation(self, heuristic_result: pd.Series) -> Dict[str, str]:
        correlation = {}
        for target in self.target_list:
            indexes = self.data[self.data[self.target_name] == target].index
            correlation[target] = dict(Counter(heuristic_result.iloc[indexes].to_list()))
        return correlation
