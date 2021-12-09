# -*- coding: utf-8 -*-
"""
    :author: Grey Li <withlihui@gmail.com>
    :copyright: (c) 2017 by Grey Li.
    :license: MIT, see LICENSE for more details.
"""
import os
import sys

sys.path.append("..")

from flask import Flask, render_template, request
from flask_dropzone import Dropzone
from src.heuristics.BasicHeuristics import BasicHeuristics
from src.utils import load_config
from src.dataset.Dataset import Dataset

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
    with open(os.path.join(app.config['UPLOADED_PATH'], "logger.txt"), "w", encoding="utf-8") as logger:
        for key, f in request.files.items():
            _, file_extension = os.path.splitext(f.filename)
            if key.startswith('file'):
                save_dir = os.path.join(app.config['UPLOADED_PATH'], f.filename)
                f.save(save_dir)
                logger.write(f"{save_dir}\n")

    return '', 204


@app.route('/form', methods=['POST', 'GET'])
def handle_form():
    dataset_type = request.form.get('contact')
    return render_template("heuristics.html.j2", data=get_config(dataset_type=dataset_type))

def get_config(dataset_type):
    logger_path = os.path.join(app.config['UPLOADED_PATH'], "logger.txt")
    file_path = open(logger_path).readlines()[-1].strip()
    f = open(logger_path, "wb") 
    f.close()

    if dataset_type in ["Base", "MultiRC"]:
        return dict(
            dataset_type=dataset_type,
            train_dataset_dir=file_path,
            column_name1="",
            column_name2="",
            target_name=""
                )
    elif dataset_type == "WordInContext":
        return dict(
            dataset_type=dataset_type,
            train_dataset_dir=file_path,
            column_name1="",
            column_name2="",
            start1="",
            start2="",
            end1="",
            end2="",
            target_name=""
        )   

def simple_heuristic(config):
    dataset = Dataset(path=config['train_dataset_dir'])
    solver = BasicHeuristics(dataset=dataset, config=config)
    return str(solver.check_heuristics())

if __name__ == '__main__':
    app.run(debug=True)
