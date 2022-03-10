import pandas as pd
from typing import List
import yaml
from argparse import ArgumentParser


def get_target_list(target_column: pd.Series) -> List:
    return target_column.unique().tolist()


def load_config(path: str):
    with open(path, "r") as yamlfile:
        data = yaml.safe_load(yamlfile)
        return data


def get_argparse() -> ArgumentParser:
    """Get argument parser.
    Returns:
        ArgumentParser: Argument parser.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser