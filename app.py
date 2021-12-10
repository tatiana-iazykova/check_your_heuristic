import os
from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from core.src.dataset.Dataset import Dataset
from core.src.dataset.MultiRCDataset import MultiRCDataset
from core.src.heuristics.BasicHeuristics import BasicHeuristics
from core.src.heuristics.WordInContextHeuristics import WordInContextHeuristics
from pathlib import Path

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_ALLOWED_FILE_CUSTOM = True,
    DROPZONE_ALLOWED_FILE_TYPE = '.csv, .xls, .xlsx, .json, .jsonl',
    DROPZONE_MAX_FILES=1,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='submit',
)

dropzone = Dropzone(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])
def handle_upload():
    mod_path = Path(__file__).parent
    with open(os.path.join(app.config['UPLOADED_PATH'], "logger.txt"), "w", encoding="utf-8") as logger:
        for key, f in request.files.items():
            _, file_extension = os.path.splitext(f.filename)
            if key.startswith('file'):
                save_dir = os.path.join(app.config['UPLOADED_PATH'], f.filename)
                f.save(save_dir)
                logger.write(f"{Path(repr(save_dir)[1:-1]).relative_to(mod_path)}\n")

    return '', 204


@app.route('/form', methods=['POST', 'GET'])
def handle_form():
    logger_path = os.path.join(app.config['UPLOADED_PATH'], "logger.txt")
    file_path = open(logger_path).readlines()[-1].strip()
    dataset_type = request.form.get('contact')

    if dataset_type in ["Base", "MultiRC"]:
        config = dict(
            dataset_type=dataset_type,
            train_dataset_dir=Path(file_path).as_posix(),
            column_name1=request.form.get('column_1'),
            column_name2=request.form.get('column_2'),
            target_name=request.form.get('target_name')
                )
    else:
        config = dict(
            dataset_type=dataset_type,
            train_dataset_dir=Path(file_path).as_posix(),
            column_name1=request.form.get('column_1'),
            column_name2=request.form.get('column_name2'),
            start1=request.form.get('start1'),
            start2=request.form.get('start2'),
            end1=request.form.get('end1'),
            end2=request.form.get('end2'),
            target_name=request.form.get('target_name')
        )

    return render_template(
        'heuristics.html.j2',
        heuristic_results=heuristic_library(
            dataset_type=dataset_type,
            config=config
        )
    )


def heuristic_library(dataset_type, config):
    if dataset_type == "Base":
        dataset = Dataset(path=config['train_dataset_dir'])
        solver = BasicHeuristics(dataset=dataset, config=config)
        return solver.check_heuristics()
    elif dataset_type == "MultiRC":
        dataset = MultiRCDataset(path=config['train_dataset_dir'])
        solver = BasicHeuristics(dataset=dataset, config=config)
        return solver.check_heuristics()
    else:
        dataset = Dataset(path=config['train_dataset_dir'])
        solver = WordInContextHeuristics(dataset=dataset, config=config)
        return solver.check_heuristics()

if __name__ == '__main__':
    app.run(debug=True)