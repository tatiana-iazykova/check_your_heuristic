import unittest

from parameterized import parameterized

from check_your_heuristic.Base import BaseSolver
from check_your_heuristic.dataset.Dataset import Dataset
from check_your_heuristic.utils import load_config


class TestBaseSolver(unittest.TestCase):
    """Class for testing BaseSolver."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with lemmatizer and preprocessor."""

        config = load_config(path="configs/config.yaml")
        dataset = Dataset(
            path="data/train.jsonl",
            path_valid="data/val.jsonl",
            path_test="data/test.jsonl"
        )
        solver = BaseSolver(
            config=config,
            dataset=dataset,
            seed=23
        )
        cls.dataset = dataset
        cls.solver = solver 

    @parameterized.expand(
        [
            (1, ["entailment"]),
            (3, ["entailment", "entailment", "entailment"]),
        ]
    )
    def test_majority_class(self, size, labels_true) -> None:
        """Testing majority class prediction"""
        labels_pred = self.solver.majority_class(test_size=size)
        self.assertListEqual(labels_true, labels_pred)

    @parameterized.expand(
        [
            (1, ["not_entailment"]),
            (3, ["not_entailment", "entailment", "entailment"]),
        ]
    )
    def test_random_choice(self, size, labels_true) -> None:
        """Testing random choice prediction"""

        labels_pred = self.solver.random_choice(test_size=size)
        labels_pred = list(labels_pred)
        self.assertListEqual(labels_true, labels_pred)
    
    @parameterized.expand(
        [
            (1, ["entailment"]),
            (3, ["entailment", "not_entailment", "not_entailment"]),
        ]
    )
    def test_random_balanced_choice(self, size, labels_true) -> None:
        """Testing random balanced choice prediction"""

        labels_pred = self.solver.random_balanced_choice(test_size=size)
        labels_pred = list(labels_pred)
        self.assertListEqual(labels_true, labels_pred)
