import pandas as pd
from pathlib import Path
from check_your_heuristic.dataset.BaseDataset import BaseDataset
import os


class Dataset(BaseDataset):
    """
    Dataset class for datasets that have simple structure
    """
    def __init__(self, path: str, path_valid: str = None, path_test: str = None):

        self.valid_data_types = {
            '.csv': self._read_csv,
            '.xls': self._read_excel,
            '.xlsx': self._read_excel,
            '.json': self._read_json,
            '.jsonl': self._read_json
        }

        super(Dataset, self).__init__(path=path, path_valid=path_valid, path_test=path_test)

    def read_data(self, path: Path) -> pd.DataFrame:
        """
        Given the path to the file returns extension of that file

        Example:
        path: "../input/some_data.csv"
        :return: ".csv"
        """
        _, extension = os.path.splitext(path)
        if extension.lower() in self.valid_data_types:
            return self.valid_data_types[extension](path)
        else:
            raise ValueError(f"Your data type ({extension}) is not supported, please convert your dataset "
                             f"to one of the following formats {list(self.valid_data_types.keys())}.")

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """
        Reads a csv file given its path
        :param path: Path("../../some_file.csv")
        :return: dataframe
        """
        return pd.read_csv(filepath_or_buffer=path, encoding="utf-8")

    @staticmethod
    def _read_excel(path: Path) -> pd.DataFrame:
        """
        Reads a xls or xlsx file given its path
        :param path: Path("../../some_file.xlsx")
        :return: dataframe
        """
        return pd.read_excel(io=path, engine="openpyxl", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> pd.DataFrame:
        """
        Reads a json or jsonl file given its path
        :param path: Path("../../some_file.jsonl")
        :return: dataframe
        """
        return pd.read_json(path_or_buf=path, lines=True, encoding="utf-8")



