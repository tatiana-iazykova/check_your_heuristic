from check_your_heuristic.dataset.Dataset import Dataset
from check_your_heuristic.heuristics.BasicHeuristics import BasicHeuristics
from check_your_heuristic.utils import get_target_list
from typing import Dict, Any, Union
import pandas as pd


class WordInContextHeuristics(BasicHeuristics):
    """
    Basic Heuristics for the Word-in-Context-like datasets (RUSSE in Russian SuperGLUE)
    """

    def __init__(self, config: Dict[str, Any], dataset: Dataset):
        super(WordInContextHeuristics, self).__init__(dataset=dataset, config=config)
        self.column_1 = config["column_name1"]
        self.column_2 = config["column_name2"]
        self.start1 = config["start1"]
        self.end1 = config["end1"]
        self.start2 = config["start2"]
        self.end2 = config["end2"]
        self.target_list = get_target_list(self.train[self.target_name])

    def heuristic_same_form(self, data: pd.DataFrame, length: int) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Heuristic that checks whether the words are in the same form
        :param data: data set provided for checking
        :param length: Length of the dataset provided
        :return:
        {
        'coverage': '13.85%',
        'correlation':
        {
        label_1: {False: '83.94%', True: '16.06%'},
        label_2: {False: '87.41%', True: '12.59%'}
        }
        }
        """

        result = {}
        same_form_heuristic = data.apply(
            lambda row:
            row[self.column_1][row[self.start1]:row[self.end1]].lower() ==
            row[self.column_2][row[self.start2]:row[self.end2]].lower(),
            axis=1
        )
        result["coverage"] = f'{same_form_heuristic.sum() / length * 100:.2f}%'
        result["correlation"] = self._get_correlation(
            heuristic_result=same_form_heuristic,
            data=data
        )

        return result

    def check_heuristics(self, render_pandas=False) -> Dict[str, Dict[str, Union[str, Dict[str, Dict[str, str]]]]]:
        """
        Checks how the heuristics are present in the data sets and prints the results
        :return: json-like objects with all the heuristics
        """
        result = dict()

        result["check_substring_train"] = self.check_substring(data=self.train, length=len(self.train))
        result["vocab_intersection_train"] = self.heuristic_vocab_intersection(data=self.train, length=len(self.train))
        result["same_form_heuristic"] = self.heuristic_same_form(data=self.train, length=len(self.train))
        if self.valid is not None:
            result["check_substring_valid"] = self.check_substring(data=self.valid, length=len(self.valid))
            result["vocab_intersection_valid"] = self.heuristic_vocab_intersection(data=self.valid,
                                                                                   length=len(self.valid))
            result["same_form_heuristic"] = self.heuristic_same_form(data=self.train, length=len(self.train))
        if render_pandas:
            result = self._render_pandas_results(res_dict=result)

        return result
