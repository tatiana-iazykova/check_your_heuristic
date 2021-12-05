from src.heuristics.BasicHeuristics import BasicHeuristics
from src.dataset.MultiRCDataset import MultiRCDataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = dict(
        train_dataset_dir="resources/MuSeRC/train.jsonl",
        valid_dataset_dir="resources/MuSeRC/val.jsonl",
        column_name1="question",
        column_name2="answer",
        target_name="label",
    )
    dataset = MultiRCDataset(path=config['train_dataset_dir'], path_valid=config['valid_dataset_dir'])
    solver = BasicHeuristics(dataset=dataset, config=config)
    solver.check_heuristics()
    solver.get_visuals()
    solver.all_methods()


if __name__ == "__main__":
    main()
