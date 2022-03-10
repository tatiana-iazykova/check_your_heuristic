from check_your_heuristic.dataset.Dataset import Dataset
from check_your_heuristic.heuristics.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, Union, Tuple
import pandas as pd
from check_your_heuristic.utils import get_target_list
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class BasicHeuristics(BaseHeuristicSolver):

    def __init__(self, config: Dict[str, Any], dataset: Dataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
        self.column_1 = config["column_name1"]
        self.column_2 = config["column_name2"]
        self.target_list = get_target_list(self.train[self.target_name])
        self.output_dir = config["output_dir"] if "output_dir" in config and config["output_dir"] else '.'

    @staticmethod
    def check_substring_function(text1: str, text2: str) -> Tuple[bool, bool]:
        """
        Heuristic that converts strings to lowercase to
        check whether one is substring of another and vice versa

        :param text1: "The cat is sleeping"
        :param text2: "The cat is sleeping in the box"
        :return: True, False
        """
        text1 = text1.lower()
        text2 = text2.lower()
        left = text1 in text2
        right = text2 in text1

        return left, right

    def check_number_of_words(self, data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Function that computes mean and median length of strings in regards to labels
        :param data: data set provided for checking

        Example output
        :return: {
        'mean':
        {'lengths_column1_entailment': 6, 'lengths_column2_entailment': 34,
        'lengths_column1_not_entailment': 6, 'lengths_column2_not_entailment': 32},
        'median':
        {'lengths_column1_entailment': 5, 'lengths_column2_entailment': 32,
        'lengths_column1_not_entailment': 5, 'lengths_column2_not_entailment': 29}}
        """

        result = {"mean": {}, "median": {}}
        data["lengths_column1"] = data[self.column_1].apply(lambda x: len(x.split()))
        data["lengths_column2"] = data[self.column_2].apply(lambda x: len(x.split()))
        for target in self.target_list:
            result["mean"][f"lengths_column1_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column1"].mean())
            result["mean"][f"lengths_column2_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column2"].mean())
            result["median"][f"lengths_column1_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column1"].median())
            result["median"][f"lengths_column2_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column2"].median())

        self._plot_boxplot(data=data, column_name='lengths_column2', output_name=self.column_2)
        self._plot_boxplot(data=data, column_name='lengths_column1', output_name=self.column_1)
        print(result)
        return result

    def check_substring(self, data: pd.DataFrame, length: int) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
        """
        Check if the heuristic check_substring is in the data frame and returns the percent of rows that have
        this heuristic present as well as the how it correlates with the labels

        :param data: Train or Validation dataset
        :param length: Length of the dataset provided
        :return:  {
        'coverage':
        {
        'hypothesis_premise': '0.92%', 'premise_hypothesis': '0.00%'
        },
        'correlation':
        {
        'hypothesis_premise': {'entailment': {False: '98.68%', True: '1.32%'},
                               'not_entailment': {False: '99.52%', True: '0.48%'}},
        'premise_hypothesis': {'entailment': {False: '100.00%'}, 'not_entailment': {False: '100.00%'}}
        }
        }
        """
        result = {"coverage": {}, "correlation": {}}
        substring_heuristic = pd.DataFrame()
        substring_heuristic[["left", "right"]] = data.apply(
            lambda row: self.check_substring_function(
                text1=row[self.column_1],
                text2=row[self.column_2]
            ),
            axis=1,
            result_type="expand"
        )
        result["coverage"][f"{self.column_1}_{self.column_2}"] = \
            f'{substring_heuristic["left"].sum() / length * 100:.2f}%'
        result["coverage"][f"{self.column_2}_{self.column_1}"] = \
            f'{substring_heuristic["right"].sum() / length:.2f}%'

        result["correlation"][f"{self.column_1}_{self.column_2}"] = self._get_correlation(
            heuristic_result=substring_heuristic["left"],
            data=data
        )
        result["correlation"][f"{self.column_2}_{self.column_1}"] = self._get_correlation(
            heuristic_result=substring_heuristic["right"],
            data=data
        )

        return result

    @staticmethod
    def check_vocab_intersection(text1: str, text2: str) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
        """
        Heuristic that converts strings to lowercase, splits them into word tokens to
        checks whether their vocabulary intersects by the following thresholds:
        0.1, 0.25, 0.33, 0.5, 0.66, 0.75 and 0.9

        :param text1: "Cat is sleeping in the comfy chair"
        :param text2: "The cat is sitting at the dining table"
        :return: (True, False, False, False, False, False, False)
        """
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        ten_percent = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.10
        quarter = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.25
        third = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.33
        half = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.5
        two_thirds = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.66
        two_quarters = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.75
        ninety_percent = len(tokens1 & tokens2) / len(tokens1 | tokens2) > 0.9

        return ten_percent, quarter, third, half, two_thirds, two_quarters, ninety_percent

    def heuristic_vocab_intersection(self, data: pd.DataFrame, length: int) -> \
            Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
        """
        Check if the heuristic check_vocab_intersection is in the data frame and returns the percent of rows that have
        this heuristic present as well as the how it correlates with the labels
        :param data: Train or Validation dataset
        :param length: Length of the dataset provided
        :return:
         {'coverage':
         {
         'threshold_0.1': '23.81%', 'threshold_0.25': '4.01%', 'threshold_0.33': '2.48%', 'threshold_0.5': '0.38%',
         'threshold_0.66': '0.08%', 'threshold_0.75': '0.00%', 'threshold_0.9': '0.00%'
         },
         'correlation':
         {
         'threshold_0.1':
         {'entailment': {True: '25.17%', False: '74.83%'}, 'not_entailment': {True: '22.35%', False: '77.65%'}},
         'threshold_0.25':
         {'entailment': {False: '96.17%', True: '3.83%'}, 'not_entailment': {False: '95.78%', True: '4.22%'}},
         'threshold_0.33':
         {'entailment': {False: '97.87%', True: '2.13%'}, 'not_entailment': {False: '97.14%', True: '2.86%'}},
         'threshold_0.5':
         {'entailment': {False: '99.71%', True: '0.29%'}, 'not_entailment': {False: '99.52%', True: '0.48%'}},
         'threshold_0.66':
         {'entailment': {False: '99.93%', True: '0.07%'}, 'not_entailment': {False: '99.92%', True: '0.08%'}},
         'threshold_0.75':
         {'entailment': {False: '100.00%'}, 'not_entailment': {False: '100.00%'}},
         'threshold_0.9':
        {'entailment': {False: '100.00%'}, 'not_entailment': {False: '100.00%'}}
        }
        }
        """
        result = {"coverage": {}, "correlation": {}}
        thresholds_columns = ["0.1", "0.25", "0.33", "0.5", "0.66", "0.75", "0.9"]
        vocab_intersection = pd.DataFrame(
            columns=thresholds_columns
        )
        vocab_intersection[thresholds_columns] = data.apply(
            lambda row: self.check_vocab_intersection(
                text1=row[self.column_1],
                text2=row[self.column_2]
            ),
            axis=1,
            result_type="expand"
        )
        for element in thresholds_columns:
            result["coverage"][f"threshold_{element}"] = \
                f'{vocab_intersection[element].sum() / length * 100:.2f}%'
            result["correlation"][f"threshold_{element}"] = self._get_correlation(
                heuristic_result=vocab_intersection[element],
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
        if self.valid is not None:
            result["check_substring_valid"] = self.check_substring(data=self.valid, length=len(self.valid))
            result["vocab_intersection_valid"] = self.heuristic_vocab_intersection(data=self.valid,
                                                                                   length=len(self.valid))
        if render_pandas:
            result = self._render_pandas_results(res_dict=result)

        return result

    def _get_correlation(self, heuristic_result: pd.Series, data: pd.DataFrame) -> Dict[str, str]:
        """
        Computes the correlation between the heuristic results and the labels
        labels are taken from the dataset provided for checking

        :param data: dataset provided for checking
        :param heuristic_result: pd.Series([1, 2, 2, 1])
        :return: {'label_1': {1: '50.00%'}, 'label_2': {2: '50.00%'}}
        """
        correlation = {}
        for target in self.target_list:
            indexes = data[data[self.target_name] == target].index
            samples = heuristic_result.iloc[indexes].to_list()
            counts = dict(Counter(samples))
            counts = {k: f"{v / len(samples) * 100:.2f}%" for k, v in counts.items()}
            correlation[target] = counts
        return correlation

    def _render_pandas_results(
            self,
            res_dict: Dict[str, Dict[str, Union[str, Dict[str, Dict[str, str]]]]]
    ) -> pd.DataFrame:
        columns = ['heuristic', "additional_info", 'coverage']
        for target in self.target_list:
            columns.append(f'correlation_{target}')

        df = pd.DataFrame(columns=columns)

        for k in res_dict.keys():
            for key in res_dict[k]['coverage'].keys():
                res = {"heuristic": k, "additional_info": key, "coverage": res_dict[k]['coverage'][key]}
                for label in self.target_list:
                    r = res_dict[k]["correlation"][key][label]
                    if True in r:
                        res[f"correlation_{label}"] = res_dict[k]["correlation"][key][label][True]
                    else:
                        res[f"correlation_{label}"] = "0.00%"
                df = df.append(res, ignore_index=True)
        return df

    def _plot_boxplot(self, data: pd.DataFrame, column_name: str, output_name=str) -> None:
        """

        :param data: data frame provided for checking
        :param column_name: column to compute box plot
        :param output_name: file_name to save
        :return: None
        """
        sns.boxplot(x=self.target_name, y=column_name, data=data)
        plt.xlabel("Labels")
        plt.ylabel("Number of words")
        plt.title(f'Relation between label and number of words in {output_name}', fontsize=14)
        plt.savefig(f"lengths_{output_name}.png")
        plt.close()

    def get_visuals(self):

        _ = plt.title('Label distribution in train data', fontsize=14)
        _ = plt.pie(self.train[self.target_name].value_counts(), autopct="%.1f%%", explode=[0.05] * 2,
                    labels=self.train[self.target_name].value_counts().keys(), pctdistance=0.5, textprops=dict(fontsize=12))
        plt.savefig("Label_distribution_in_train_data.png")
        plt.close()
        self.check_number_of_words(data=self.train)

        if self.valid is not None:
            _ = plt.title('Label distribution in validation data', fontsize=14)
            _ = plt.pie(self.valid[self.target_name].value_counts(), autopct="%.1f%%", explode=[0.05] * 2,
                        labels=self.valid[self.target_name].value_counts().keys(), pctdistance=0.5,
                        textprops=dict(fontsize=12))
            plt.savefig("Label_distribution_in_validation_data.png")
            plt.close()
            self.check_number_of_words(data=self.valid)

