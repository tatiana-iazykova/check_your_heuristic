from check_your_heuristic.dataset.ReCoRDDataset import ReCoRDDataset
from check_your_heuristic.heuristics.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, List
import pandas as pd
import string
import numpy as np
import logging


class ReCoRDHeuristics(BaseHeuristicSolver):
    def __init__(self, config: Dict[str, Any], dataset: ReCoRDDataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
        self.passage_column = config["passage_column"]
        self.question_column = config["question_column"]
        self.entities_column = config["entities_column"]

    @staticmethod
    def normalize_answer(text: str):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def white_space_fix(line):
            return ' '.join(line.split())

        def remove_punct(line):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in line if ch not in exclude)

        return white_space_fix(remove_punct(text.lower()))

    def _get_entities(self, row, column_name:str):
        words = [
            row[self.passage_column][x["start"]: x["end"]]
            for x in row[column_name]
        ]
        return words

    def get_basic_pred(self,
                       row: pd.DataFrame,
                       words: List[str],
                       _words: List[str],
                       line_candidates: List[str]
                       ) -> str:
        if len(_words) == 0:
            if len(words) == 1:
                pred = words[0]
            else:
                for word in words:
                    line_candidates.append(row[self.question_column].replace("@placeholder", word))
                pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                            size=1)[0]
                pred = np.array(words)[pred_idx]
        elif len(_words) == 1:
            pred = _words[0]
        else:
            for word in _words:
                line_candidates.append(row[self.question_column].replace("@placeholder", word))
            pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                        size=1)[0]
            pred = np.array(_words)[pred_idx]
        return pred

    def filtration_count_heuristic(self, row: pd.DataFrame) -> str:
        """
        Heuristic that removes some candidates and filters out candidates depended on times they occurred in the text
        If there are a lot of candidates the one is chosen randomly
        """
        line_candidates = []
        _words = []
        text = row[self.passage_column].split()
        words = self._get_entities(row=row, column_name=self.entities_column)
        for word in words:
            if word[:-2] not in row[self.question_column] or text.count(words[:-2]) >= 2:
                _words.append(word)
        pred = self.get_basic_pred(row=row, words=words, _words=_words, line_candidates=line_candidates)
        return self.normalize_answer(pred)

    def remove_candidates_heuristic(self, row: pd.DataFrame) -> str:
        """
        Heuristic that removes candidates that occur in the question
        """
        words = self._get_entities(row=row, column_name=self.entities_column)
        line_candidates = []
        _words = []
        for word in words:
            if word[:-1] not in row[self.question_column]:
                _words.append(word)

        pred = self.get_basic_pred(row=row, words=words, _words=_words, line_candidates=line_candidates)
        return self.normalize_answer(pred)

    def metric_max_over_ground_truths(self, row: pd.DataFrame, predictions_colname: str) -> float:
        """
        As there is several true answers, we go over all and compute metric for all
        :param row: row of the data frame predicted
        :param predictions_colname: the name for heuristic
        :return:
        """
        scores_for_ground_truths = [0]
        prediction = row[predictions_colname]
        ground_truths = self._get_entities(row=row, column_name=self.target_name)
        for ground_truth in ground_truths:
            score = self.exact_match_score(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def check_heuristics(self) -> Dict[str, float]:
        """
        Checks how the heuristics are present in the data sets and prints the results
        :return: json-like object with all the results
        """

        result = {}

        self.train["pred_remove_candidates_heuristic"] = self.train.apply(
            self.remove_candidates_heuristic,
            axis=1
        )
        result["exact_match_score_remove_candidates_heuristic"] = np.mean(self.train.apply(
            lambda row: self.metric_max_over_ground_truths(
                row=row,
                predictions_colname="pred_remove_candidates_heuristic"
            ),
            axis=1).to_list()
        )

        self.train["pred_filtration_count_heuristic"] = self.train.apply(
            self.filtration_count_heuristic,
            axis=1
        )
        self.train["true_filtration_count_heuristic"] = self.train[self.target_name]
        result["exact_match_score_filtration_count_heuristic_train"] = np.mean(self.train.apply(
            lambda row: self.metric_max_over_ground_truths(
                row=row,
                predictions_colname="true_filtration_count_heuristic"
            ),
            axis=1
        ).to_list()
        )
        if self.valid is not None:
            result_df = pd.DataFrame()
            result_df["pred_remove_candidates_heuristic"] = self.valid.apply(
                self.remove_candidates_heuristic,
                axis=1
            )
            result_df["true_labels"] = self.valid[self.target_name]
            result["exact_match_score_remove_candidates_heuristic_valid"] = np.mean(result_df.apply(
                lambda row: self.metric_max_over_ground_truths(
                    row=row,
                    predictions_colname="pred_remove_candidates_heuristic"
                ),
                axis=1
                ).to_list()
            )

            result_df["pred_filtration_count_heuristic"] = self.valid.apply(
                self.filtration_count_heuristic,
                axis=1
            )
            result_df["true_filtration_count_heuristic"] = self.valid[self.target_name]
            result["exact_match_score_filtration_count_heuristic_valid"] = np.mean(result_df.apply(
                lambda row: self.metric_max_over_ground_truths(
                    row=row,
                    predictions_colname="pred_filtration_count_heuristic"
                ),
                axis=1).to_list()
            )
        for key, value in result.items():
            print(key, '\n', value, '\n')

        return result

    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        return prediction == self.normalize_answer(ground_truth)

    def all_methods(self):
        logging.error("Method is deprecated for this type of dataset")
        raise AttributeError

    def random_balanced_choice(self):
        logging.error("Method is deprecated for this type of dataset")
        raise AttributeError

    def random_choice(self):
        logging.error("Method is deprecated for this type of dataset")
        raise AttributeError

    def majority_class(self):
        logging.error("Method is deprecated for this type of dataset")
        raise AttributeError

    def show_report(self):
        logging.error("Method is deprecated for this type of dataset")
        raise AttributeError




