from src.Heuristic import BaseHeuristicSolver
from src.utils import load_config


def main():
    config = load_config("config.yaml")
    MyHeusticCheck = BaseHeuristicSolver(path=config['train_dataset_dir'],
                                         path_valid=config['valid_dataset_dir'],
                                         config=config)
    MyHeusticCheck.check_heuristics()
    MyHeusticCheck.all_methods()


if __name__ == "__main__":
    main()