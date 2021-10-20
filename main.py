from src.Heuristic import BaseHeuristicSolver
from src.utils import load_config


def main():
    config = load_config("config.yaml")
    print(config)
    MyHeuristicCheck = BaseHeuristicSolver(path=config['train_dataset_dir'],
                                         path_valid=config['valid_dataset_dir'],
                                         config=config)
    MyHeuristicCheck.check_heuristics()
    MyHeuristicCheck.all_methods()


if __name__ == "__main__":
    main()
