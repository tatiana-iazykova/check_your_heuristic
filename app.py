import os
from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from core.src.dataset.Dataset import Dataset
from core.src.dataset.MultiRCDataset import MultiRCDataset
from core.src.heuristics.BasicHeuristics import BasicHeuristics
from core.src.heuristics.WordInContextHeuristics import WordInContextHeuristics
from pathlib import Path
from uuid import uuid4
from datetime import datetime


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
variable = "unique_token"

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_MAX_FILE_SIZE=10,
    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='.csv, .xls, .xlsx, .json, .jsonl',
    DROPZONE_MAX_FILES=1,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='handle_upload',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='submit'
)

dropzone = Dropzone(app)

"""
    ROUTES
"""


def get_token():
    rand_token = uuid4()
    date = datetime.now()
    return f"{rand_token}{date}"


@app.route('/')
def index():
    return render_template('index.html.j2')


@app.route('/about')
def guides():
    return render_template('about.html.j2')


@app.route('/load_dataset')
def submit_dataset():
    global variable
    variable = get_token()
    return render_template('submission.html.j2', variable=variable)


@app.route('/contacts')
def contacts():
    return render_template('contacts.html.j2')


"""
    METHODS
"""


@app.route('/upload', methods=['POST', 'GET'])
def handle_upload():
    mod_path = Path(__file__).parent
    with open(os.path.join(app.config['UPLOADED_PATH'], "logger.txt"), "w", encoding="utf-8") as logger:
        for key, f in request.files.items():
            _, file_extension = os.path.splitext(f.filename)
            if key.startswith('file'):
                global variable
                filename = variable + file_extension
                save_dir = os.path.join(app.config['UPLOADED_PATH'], filename)
                f.save(save_dir)
                logger.write(f"{Path(repr(save_dir)[1:-1]).relative_to(mod_path)}\n")
    return '', 204


@app.route('/form', methods=['POST', 'GET'])
def handle_form():
    print(request.form.get('token'))
    logger_path = os.path.join(app.config['UPLOADED_PATH'], "logger.txt")
    file_path = open(logger_path).readlines()[-1].strip()
    dataset_type = request.form.get('dataset_type')

    config = dict(
        train_dataset_dir=Path(file_path).as_posix(),
        column_name1=request.form.get('column_1'),
        column_name2=request.form.get('column_2'),
        target_name=request.form.get('target_name')
        )

    if dataset_type == 'WordInContext':
        config['start1'] = request.form.get('start1')
        config['start2'] = request.form.get('start2')
        config['end1'] = request.form.get('end1')
        config['end2'] = request.form.get('end2')

    try:
        solver = heuristic_library(dataset_type=dataset_type, config=config)
    except KeyError:
        return render_template('wrong_column.html.j2')

    path_to_visuals = Path(__file__).parent / "static/uploads"
    return render_template(
        'output.html.j2',
        heuristic_results=get_df(
            solver=solver
        ),
        visuals=get_images(
            path=path_to_visuals.as_posix()
        ),
        cl_reports=get_classification_reports(
            solver=solver
        )
    )


def get_images(path: str):
    res = []
    for file in os.listdir(path):
        path_to_doc = Path(file)
        if path_to_doc.suffix == ".png":
            res.append(path_to_doc.as_posix())
    return res


def get_df(solver):
    df = solver.check_heuristics(render_pandas=True)
    df['heuristic'] = df['heuristic'].str.replace("_train", '')
    df['heuristic'] = df['heuristic'].str.replace("_", ' ')
    df['additional info'] = df['additional info'].str.replace("_", ' ')
    return df.to_html()


def get_classification_reports(solver):
    cl_reports = solver.all_methods()
    return cl_reports


def heuristic_library(dataset_type, config):
    global variable

    if dataset_type == 'WordInContext':
        dataset = Dataset(path=config['train_dataset_dir'])
        solver = WordInContextHeuristics(dataset=dataset, config=config, variable=variable)
    elif dataset_type == "MultiRC":
        dataset = MultiRCDataset(path=config['train_dataset_dir'])
        solver = BasicHeuristics(dataset=dataset, config=config, variable=variable)
    elif dataset_type == 'Base':
        dataset = Dataset(path=config['train_dataset_dir'])
        solver = BasicHeuristics(dataset=dataset, config=config, variable=variable)

    solver.get_visuals()
    return solver


if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0'
    )
