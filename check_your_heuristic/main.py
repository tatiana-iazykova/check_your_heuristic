from check_your_heuristic.utils import load_config, get_argparse
from check_your_heuristic.dataset import Dataset, MultiRCDataset, ReCoRDDataset
from check_your_heuristic.heuristics import BasicHeuristics, ReCoRDHeuristics, WordInContextHeuristics


def base_case():
    parser = get_argparse()
    args = parser.parse_args()

    config = load_config(args.path_to_config)

    path_valid = config['valid_dataset_dir'] if 'valid_dataset_dir' in config else None
    dataset = Dataset(path=config['train_dataset_dir'], path_valid=path_valid)
    solver = BasicHeuristics(dataset=dataset, config=config)

    solver.get_visuals()
    solver.check_heuristics()
    solver.all_methods()


def multirc_case():
    parser = get_argparse()
    args = parser.parse_args()

    config = load_config(args.path_to_config)

    path_valid = config['valid_dataset_dir'] if 'valid_dataset_dir' in config else None

    dataset = MultiRCDataset(path=config['train_dataset_dir'], path_valid=path_valid)
    solver = BasicHeuristics(dataset=dataset, config=config)

    solver.get_visuals()
    solver.check_heuristics()
    solver.all_methods()


def record_case():
    parser = get_argparse()
    args = parser.parse_args()

    config = load_config(args.path_to_config)

    path_valid = config['valid_dataset_dir'] if 'valid_dataset_dir' in config else None

    dataset = ReCoRDDataset(path=config['train_dataset_dir'], path_valid=path_valid)
    solver = ReCoRDHeuristics(dataset=dataset, config=config)

    solver.check_heuristics()


def wordincontext_case():
    parser = get_argparse()
    args = parser.parse_args()

    config = load_config(args.path_to_config)

    path_valid = config['valid_dataset_dir'] if 'valid_dataset_dir' in config else None

    dataset = Dataset(path=config['train_dataset_dir'], path_valid=path_valid)
    solver = WordInContextHeuristics(dataset=dataset, config=config)

    solver.get_visuals()
    solver.check_heuristics()
    solver.all_methods()

