import pandas as pd
from pathlib import Path, PurePosixPath
import os


class Dataset:

    def __init__(self, path_to_train: str, path_to_valid: str = None, path_to_test: str = None):

        self.valid_data_types = {
            '.csv': self._read_csv,
            '.xls': self._read_excel,
            '.xlsx': self._read_excel,
            '.json': self._read_json,
            '.jsonl': self._read_json
        }

        self.train = self.read_data(Path(path_to_train).as_posix())
        self.valid = self.read_data(Path(path_to_valid).as_posix()) if path_to_valid is not None else None
        self.test = self.read_data(Path(path_to_test).as_posix()) if path_to_test is not None else None

    def read_data(self, path: str) -> pd.DataFrame:
        """
        Given the path to the file returns extension of that file

        Example:
        path: "../input/some_data.csv"
        :return: ".csv"
        """
        _, extension = os.path.splitext(path)
        if extension in self.valid_data_types:
            return self.valid_data_types[extension](path)

    def _read_csv(self, path: str):
        return pd.read_csv(filepath_or_buffer=path)

    def _read_excel(self, path: str):
        return pd.read_excel(io=path)

    def _read_json(self, path: str):
        return pd.read_json(path_or_buf=path, lines=True)



