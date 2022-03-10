base_config = dict(
    train_dataset_dir="",
    valid_dataset_dir="",
    column_name1="hypothesis",
    column_name2="premise",
    target_name="label",
)

wordincontext_config = dict(
    train_dataset_dir="",
    valid_dataset_dir="",
    column_name1="sentence1",
    column_name2="sentence2",
    start1="start1",
    start2="start2",
    end1="end1",
    end2="end2",
    target_name="label",
)

record_config = dict(
        train_dataset_dir="",
        passage_column='text',
        question_column="question",
        entities_column="entities",
        target_name="answers",
    )