from src.heuristics.ReCoRDHeuristics import ReCoRDHeuristics
from src.dataset.ReCoRDDataset import ReCoRDDataset
import warnings
warnings.filterwarnings("ignore")


def main():
    config = dict(
        train_dataset_dir="resources/RuCoS/val.jsonl",
        passage_column='text',
        question_column="question",
        entities_column="entities",
        target_name="answers",
    )
    dataset = ReCoRDDataset(path=config['train_dataset_dir'])
    solver = ReCoRDHeuristics(dataset=dataset, config=config)
    solver.check_heuristics()


if __name__ == "__main__":
    main()
