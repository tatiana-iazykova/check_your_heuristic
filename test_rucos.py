from src.heuristics.BasicHeuristics import BasicHeuristics
from src.dataset.ReCoRDDataset import ReCoRDDataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = dict(
        train_dataset_dir="resources/RuCoS/val.jsonl",
        column_name1="question",
        column_name2="answer",
        target_name="label",
    )
    dataset = ReCoRDDataset(path=config['train_dataset_dir'])
    solver = BasicHeuristics(dataset=dataset, config=config)
    solver.all_methods()


if __name__ == "__main__":
    main()
