import pandas as pd
from typing import List

def get_target_list(target_column: pd.Series) -> List:
    return target_column.unique().to_list()

def get_correlated_words(input: pd.DataFrame) -> pd.DataFrame:
    pass