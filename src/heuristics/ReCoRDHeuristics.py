from src.dataset.ReCoRDDataset import ReCoRDDataset
from src.heuristics.Heuristic import BaseHeuristicSolver
from typing import Dict, Any, Union, Tuple
import pandas as pd
import string
import numpy as np


class ReCoRDHeuristics(BaseHeuristicSolver):
    def __init__(self, config: Dict[str, Any], dataset: ReCoRDDataset):
        super(BaseHeuristicSolver, self).__init__(dataset=dataset, config=config)

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

    def get_row_pred(row, vect):
        res = []
        words = [
            row["passage"]["text"][x["start"]: x["end"]]
            for x in row["passage"]["entities"]]
        text = row['passage']['text'].split()
        for line in row["qas"]:
            line_candidates = []
            _words = []
            for word in words:
                if word[:-2] not in line['query'] or text.count(words[:-2]) >= 2:
                    _words.append(word)
            if len(_words) == 0:
                for word in words:
                    line_candidates.append(line["query"].replace("@placeholder", word))
                pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                            size=1)[0]
                pred = np.array(words)[pred_idx]
            elif len(_words) == 1:
                pred = _words[0]
            else:
                for word in _words:
                    line_candidates.append(line["query"].replace("@placeholder", word))
                pred_idx = np.random.choice(np.arange(1, len(line_candidates)),
                                            size=1)[0]
                pred = np.array(_words)[pred_idx]
            res.append(pred)
        return " ".join(res)


