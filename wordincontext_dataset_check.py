from check_your_heuristic.heuristics.WordInContextHeuristics import WordInContextHeuristics
from check_your_heuristic.dataset.Dataset import Dataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = dict(
        train_dataset_dir="resources/RUSSE/train.jsonl",
        valid_dataset_dir="resources/RUSSE/val.jsonl",
        column_name1="sentence1",
        column_name2="sentence2",
        start1="start1",
        start2="start2",
        end1="end1",
        end2="end2",
        target_name="label",
    )
    dataset = Dataset(path=config['train_dataset_dir'])
    solver = WordInContextHeuristics(dataset=dataset, config=config)
    solver.get_visuals()
    solver.check_heuristics()
    solver.all_methods()


if __name__ == "__main__":
    main()
