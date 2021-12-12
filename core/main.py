from core.src.heuristics.BasicHeuristics import BasicHeuristics
from core.src.utils import load_config
from core.src.dataset.Dataset import Dataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = load_config("config.yaml")
    dataset = Dataset(path=config['train_dataset_dir'], path_valid=config['valid_dataset_dir'])
    solver = BasicHeuristics(dataset=dataset, config=config)
    solver.get_visuals()
    solver.check_heuristics()
    solver.all_methods()

if __name__ == "__main__":
    main()
