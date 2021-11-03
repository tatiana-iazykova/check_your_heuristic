from heuristics.BasicHeuristics import BasicHeuristics
from src.utils import load_config
from src.Dataset import Dataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = load_config("config.yaml")
    dataset = Dataset(path=config['train_dataset_dir'], path_valid=config['valid_dataset_dir'])
    solver = BasicHeuristics(dataset=dataset, config=config)
    solver.check_heuristics()
    solver.all_methods()


if __name__ == "__main__":
    main()
