import pandas as pd
from pathlib import Path
from src.dataset.BaseDataset import BaseDataset
import json


class MultiRCDataset(BaseDataset):
    """
    MultiRC is Multi-Sentence Reading Comprehension dataset in SuperGLUE benchmark.
    The corresponding name for this type of datasets in Russian SuperGLUE is MuSeRC or
    Russian Multi-Sentence Reading Comprehension.

    NB! As of now this class can only work with json data format
    """

    def __init__(self, path: str, path_valid: str = None, path_test: str = None):
        super(MultiRCDataset, self).__init__(path=path, path_valid=path_valid, path_test=path_test)

    def read_data(self, path: Path):
        """ get json file content as a pandas DataFrame"""

        df = pd.DataFrame(columns=['question', 'text', 'label', 'passage'])

        lines = self.yield_lines(path=path)

        for passage_id, line in enumerate(lines):
            passage, questions = self.split_texts_and_questions(line=line)
            questions = pd.json_normalize(
                questions, 'answers', 'question'
            )[['question', 'text', 'label']]

            questions['passage'] = passage

            df = pd.concat([df, questions])

        df = df.rename(columns={'text': 'answer'})

        return df

    def yield_lines(self, path: Path):
        """ yields json lines one by one """
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def split_texts_and_questions(self, line):
        """ transforms a complex json object into a single row dataframe"""
        text = line['passage']['text']
        questions = line['passage']['questions']

        return text, questions
