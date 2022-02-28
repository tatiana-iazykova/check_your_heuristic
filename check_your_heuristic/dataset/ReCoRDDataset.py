import pandas as pd
from pathlib import Path
from check_your_heuristic.dataset.MultiRCDataset import MultiRCDataset
from typing import Any, Tuple


class ReCoRDDataset(MultiRCDataset):
    """
    ReCorD is Reading Comprehension with Commonsense Reasoning dataset in SuperGLUE benchmark.
    The corresponding name for this type of datasets in Russian SuperGLUE is RuCoS or
    Russian reading comprehension with Commonsense reasoning.

    NB! As of now this class can only work with json data format
    """

    def __init__(self, path: str, path_valid: str = None, path_test: str = None):
        super(ReCoRDDataset, self).__init__(path=path, path_valid=path_valid, path_test=path_test)

    def split_texts_and_questions(self, line) -> Tuple[str, Any, Any]:
        """ transforms a complex json object into a single row dataframe"""
        text = line['passage']['text']
        entities = line['passage']['entities']
        questions = line['qas']

        return text, entities, questions

    def read_data(self, path: Path) -> pd.DataFrame:
        """ get json file content as a pandas DataFrame"""
        df = pd.DataFrame()

        lines = self.yield_lines(path=path)

        for passage_id, line in enumerate(lines):
            text, entities, questions = self.split_texts_and_questions(line=line)

            for i in range(len(questions)):
                df = df.append(
                    {
                        'text': text,
                        'entities': entities,
                        'question': questions[i]['query'],
                        'answers': questions[i]['answers']
                    },
                    ignore_index=True)
        return df

