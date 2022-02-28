import pandas as pd
from pathlib import Path


class BaseDataset:

    def __init__(self, path: str, path_valid: str = None, path_test: str = None):
        self.train = self.read_data(Path(repr(path)[1:-1]).as_posix())
        self.valid = self.read_data(Path(repr(path_valid)[1:-1]).as_posix()) if path_valid is not None else None
        self.test = self.read_data(Path(repr(path_test)[1:-1]).as_posix()) if path_test is not None else None

    def read_data(self, path: Path) -> pd.DataFrame:
        raise NotImplementedError



