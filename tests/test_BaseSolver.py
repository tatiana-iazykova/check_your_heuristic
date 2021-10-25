import unittest

import pandas as pd
from parameterized import parameterized

from src.Base import BaseSolver
from src.utils import load_config

class TestBaseSolver(unittest.TestCase):
    """Class for testing BaseSolver."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with lemmatizer and preprocessor."""

        config = load_config(path="tests/configs/config.yaml")
        solver = BaseSolver(
            config=config,
            path="tests/data/train.jsonl",
            path_valid="tests/data/val.jsonl",
            seed=23
        )
        cls.solver = solver 

    @parameterized.expand(
        [
            ("tests/data/train.jsonl", 1, ["entailment"]),
            ("tests/data/val.jsonl", 3, ["entailment", "entailment", "entailment"]),
        ]
    )
    def test_majority_class(self, data, size, labels_true) -> None:
        """Testing majority class prediction"""
        import os
        print(os.getcwd())
        data = pd.read_json(path_or_buf=data, lines=True, encoding='utf-8')
        labels_pred = self.solver.majority_class(test_size=size)
        self.assertListEqual(labels_true, labels_pred)

    @parameterized.expand(
        [
            ("tests/data/train.jsonl", 1, ["not_entailment"]),
            ("tests/data/val.jsonl", 3, ["not_entailment", "entailment", "entailment"]),
        ]
    )
    def test_random_choice(self, data, size, labels_true) -> None:
        """Testing random choice prediction"""

        data = pd.read_json(path_or_buf=data, lines=True)
        labels_pred = self.solver.random_choice(test_size=size)
        labels_pred = list(labels_pred)
        self.assertListEqual(labels_true, labels_pred)
    
    @parameterized.expand(
        [
            ("tests/data/train.jsonl", 1, ["entailment"]),
            ("tests/data/val.jsonl", 3, ["entailment", "not_entailment", "not_entailment"]),
        ]
    )
    def test_random_balanced_choice(self, data, size, labels_true) -> None:
        """Testing random balanced choice prediction"""

        data = pd.read_json(path_or_buf=data, lines=True)
        labels_pred = self.solver.random_balanced_choice(test_size=size)
        labels_pred = list(labels_pred)
        self.assertListEqual(labels_true, labels_pred)
