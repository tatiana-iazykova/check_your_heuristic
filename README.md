# check_your_heuristic

#Quick start

## Installation

```
pip install check_your_heuristic
```

## Configurations
To check your dataset fill the [config](config.yaml), using **unix-like** paths.

Example config:
```yaml
train_dataset_dir: "dataset/dir/train.jsonl"
valid_dataset_dir: "dataset/dir/val.jsonl"
column_name1: "premise"
column_name2: "hypothesis"
target_name: "label"
```

Other config variations can be found [here](check_your_heuristic/configs.py)

# CLI Use

Our library offers four build-in commands for checking your datasets depending on the dataset structure you have

1. Base case or two text columns + one target column 
(for example, CommitmentBank from SuperGLUE or TERRa from Russian SuperFLUE)

```
run-base-case --path_to_config config.yaml 
```

2. When you have some long text and some questions and answers for it
(for example, MultiRC from SuperGLUE or MuSeRC from Russian SuperFLUE)

```
run-multirc-case --path_to_config config.yaml 
```

3. When you have passage, questions and some NERs (or entities) that serve as answers
(for example, ReCoRD from SuperGLUE or RuCoS from Russian SuperFLUE)

```
run-record-case --path_to_config config.yaml 
```

4. Case when you have two cases and need to compare some words in them
(for example, Words in Context (WiC) from SuperGLUE or RUSSE from Russian SuperFLUE)

```
run-wordincontext-case --path_to_config config.yaml 
```