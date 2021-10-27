from src.Heuristic import BaseHeuristicSolver
from src.utils import load_config
from src.Dataset import Dataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = load_config("config.yaml")
    dataset = Dataset(path=config['train_dataset_dir'], path_valid=config['valid_dataset_dir'])
    MyHeuristicCheck = BaseHeuristicSolver(dataset=dataset, config=config)
    MyHeuristicCheck.check_heuristics()
    MyHeuristicCheck.all_methods()


if __name__ == "__main__":
    main()
