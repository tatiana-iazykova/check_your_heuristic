from src.dataset.ReCoRDDataset import ReCoRDDataset
from src.heuristics.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, Union, Tuple
import pandas as pd
import string
import numpy as np


class ReCoRDHeuristics(BaseHeuristicSolver):
    def __init__(self, config: Dict[str, Any], dataset: ReCoRDDataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)
        self.passage_column = config["passage_column"]
        self.entities_column = config["entities_column"]

    @staticmethod
    def normalize_answer(text: str):
        """Lower text and remove punctuation, articles and extra whitespace."""

        @staticmethod
        def white_space_fix(line):
            return ' '.join(line.split())

        @staticmethod
        def remove_punct(line):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in line if ch not in exclude)

        return white_space_fix(remove_punct(text.lower()))

    @staticmethod
    def _get_entities(row):
        words = [
            row["text"][x["start"]: x["end"]]
            for x in row["entities"]
        ]
        return words

    def filtration_count_heuristic(self, row):

        line_candidates = []
        _words = []
        text = row['text'].split()
        words = self._get_entities(row)
        for word in words:
            if word[:-2] not in row['question'] or text.count(words[:-2]) >= 2:
                _words.append(word)

        if len(_words) == 0:
            for word in words:
                line_candidates.append(row["question"].replace("@placeholder", word))
            pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                        size=1)[0]
            pred = np.array(words)[pred_idx]
        elif len(_words) == 1:
            pred = _words[0]
        else:
            for word in _words:
                line_candidates.append(row["question"].replace("@placeholder", word))
            pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                        size=1)[0]
            pred = np.array(_words)[pred_idx]
        return pred

    def remove_candidates_heuristic(self, row):
        words = self._get_entities(row)
        line_candidates = []
        _words = []
        for word in words:
            if word[:-1] not in row['question']:
                _words.append(word)
        if len(_words) == 0:
            for word in words:
                line_candidates.append(row['question'].replace("@placeholder", word))
            pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                        size=1)[0]
            pred = np.array(words)[pred_idx]
        elif len(_words) == 1:
            pred = _words[0]
        else:
            for word in _words:
                line_candidates.append(row['question'].replace("@placeholder", word))
            pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                        size=1)[0]
            pred = np.array(_words)[pred_idx]
        return pred



