from src.dataset.Dataset import Dataset
from src.heuristics.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, Union, Tuple
import pandas as pd
from src.utils import get_target_list
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class BasicHeuristics(BaseHeuristicSolver):

    def __init__(self, config: Dict[str, Any], dataset: Dataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
        self.column_1 = config["column_name1"]
        self.column_2 = config["column_name2"]
        self.target_list = get_target_list(self.train[self.target_name])

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

    def check_number_of_words(self, data: pd.DataFrame) -> None:  # Dict[str, int]
        """
        Function that computes mean and median length of strings in regards to labels
        :param data: data set provided for checking

        Example output
        :return: {'mean_lengths_column1_label1': 6, 'mean_lengths_column2_label1': 34,
        'median_lengths_column1_label1': 5, 'median_lengths_column2_label1': 32,
        'mean_lengths_column1_label2': 6, 'mean_lengths_column2_label2': 32,
        'median_lengths_column1_label2': 5, 'median_lengths_column2_label2': 29}
        """
        result = {}
        data["lengths_column1"] = data[self.column_1].apply(lambda x: len(x.split()))
        data["lengths_column2"] = data[self.column_2].apply(lambda x: len(x.split()))
        for target in self.target_list:
            result[f"mean_lengths_column1_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column1"].mean())
            result[f"mean_lengths_column2_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column2"].mean())
            result[f"median_lengths_column1_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column1"].median())
            result[f"median_lengths_column2_{target}"] = \
                round(data[data[self.target_name] == target]["lengths_column2"].median())

        self._plot_boxplot(data=data, column_name='lengths_column2', output_name=self.column_2)
        self._plot_boxplot(data=data, column_name='lengths_column1', output_name=self.column_1)
        print(result)

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
        plt.savefig(f"output/lengths_{output_name}.png")
        plt.close()

    def check_substring(self, data: pd.DataFrame, length: int) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Check if the heuristic check_substring is in the data frame and returns the percent of rows that have
        this heuristic present as well as the how it correlates with the labels

        :param data: Train or Validation dataset
        :param length: Length of the dataset provided
        :return: {'coverage_hypothesis_premise': '1.30%', 'coverage_premise_hypothesis': '0.00%',
        'correlation_hypothesis_premise': {'entailment': {False: '98.69%', True: '1.31%'},
        'not_entailment': {False: '98.70%', True: '1.30%'}},
        'correlation__premise_hypothesis': {'entailment': {False: '100.00%'}, 'not_entailment': {False: '100.00%'}}}
        """
        result = {}
        substring_heuristic = pd.DataFrame()
        substring_heuristic[["left", "right"]] = data.apply(
            lambda row: self.check_substring_function(
                text1=row[self.column_1],
                text2=row[self.column_2]
            ),
            axis=1,
            result_type="expand"
        )
        result[f"coverage_{self.column_1}_{self.column_2}"] = \
            f'{substring_heuristic["left"].sum() / length * 100:.2f}%'
        result[f"coverage_{self.column_2}_{self.column_1}"] = \
            f'{substring_heuristic["right"].sum() / length:.2f}%'

        result[f"correlation_{self.column_1}_{self.column_2}"] = self._get_correlation(
            heuristic_result=substring_heuristic["left"],
            data=data
        )
        result[f"correlation_{self.column_2}_{self.column_1}"] = self._get_correlation(
            heuristic_result=substring_heuristic["right"],
            data=data
        )

        return result

    def check_heuristics(self) -> None:  # Dict[str, Dict[str, Union[str, Dict[str, Dict[str, str]]]]]:
        """
        Checks how the heuristics are present in the data sets and prints the results
        :return: None
        """
        result = dict()
        result["check_substring_train"] = self.check_substring(data=self.train, length=len(self.train))
        if self.valid is not None:
            result["check_substring_valid"] = self.check_substring(data=self.valid, length=len(self.valid))

        for key, value in result.items():
            print(key, '\n', value, '\n')

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

    def get_visuals(self):

        _ = plt.title('Label distribution in train data', fontsize=14)
        _ = plt.pie(self.train[self.target_name].value_counts(), autopct="%.1f%%", explode=[0.05] * 2,
                    labels=self.train[self.target_name].value_counts().keys(), pctdistance=0.5, textprops=dict(fontsize=12))
        plt.savefig("output/Label_distribution_in_train_data.png")
        plt.close()
        self.check_number_of_words(data=self.train)

        if self.valid is not None:
            _ = plt.title('Label distribution in validation data', fontsize=14)
            _ = plt.pie(self.valid[self.target_name].value_counts(), autopct="%.1f%%", explode=[0.05] * 2,
                        labels=self.valid[self.target_name].value_counts().keys(), pctdistance=0.5,
                        textprops=dict(fontsize=12))
            plt.savefig("output/Label_distribution_in_validation_data.png")
            plt.close()
            self.check_number_of_words(data=self.valid)

