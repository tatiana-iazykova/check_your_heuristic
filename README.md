# check_your_heuristic

Python >= 3.7 is required
```pip install -r requirements.txt```

To check your dataset fill the [config](config.yaml), using **unix-like** paths.

Example config:
```yaml
train_dataset_dir: "dataset/dir/train.jsonl"
valid_dataset_dir: "dataset/dir/val.jsonl"
column_name1: "premise"
column_name2: "hypothesis"
target_name: "label"
```

Then in console run:
 ```python
python main.py
```

| heuristic                                             | requirements                                   | approximate time |
| ----------------------------------------------------- | ---------------------------------------------- | ---------------- |
| one is a substring of another                         | none                                           | *5.6s all*       |
| vocabulary overlap by 1/3                             | lemmatisation                                  | *5.6s all* 
| vocabulary overlap by 3/4                             | lemmatisation                                  | *5.6s all* 
| vocabulary overlap by 2/3                             | lemmatisation                                  | *5.6s all* 
| vocabulary overlap by 100%                            | lemmatisation                                  | *5.6s all* 
| less than some words                                  | calculate correlation                          | *5.6s all* 
| more than some words                                  | calculate correlation                          | *5.6s all* 
| presence of specific words                            | calculate correlation?? how? eli5 or manually? |  411ms           |
| Parus heuristic (more shared lemmas with the premise) | lemmatisation                                  | **8.9s all**     |
| Parus heuristic (more words than in another)          | lemmatisation                                  | **8.9s all**  
| all lemmas occur                                      | lemmatisation                                  | *15s  all*       |
| some number of overlapping lemmas                     | calculate correlation, lemmatisation           | *15s  all*
| difference btw number of tokens                       | none                                           | 400ms            |
