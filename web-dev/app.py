# -*- coding: utf-8 -*-
"""
    :author: Grey Li <withlihui@gmail.com>
    :copyright: (c) 2017 by Grey Li.
    :license: MIT, see LICENSE for more details.
"""
import os
import sys

sys.path.append("..")

from flask import Flask, config, render_template, request
from flask_dropzone import Dropzone
from src.heuristics.BasicHeuristics import BasicHeuristics
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
    logger_path = os.path.join(app.config['UPLOADED_PATH'], "logger.txt")
    file_path = open(logger_path).readlines()[-1].strip()
    dataset_type = request.form.get('contact')

    if dataset_type in ["Base", "MultiRC"]:
        config = dict(
            dataset_type=dataset_type,
            train_dataset_dir=file_path,
            column_name1=request.form.get('column_1'),
            column_name2=request.form.get('column_2'),
            target_name=request.form.get('target_name')
                )
    else:
        config = dict(
            dataset_type=dataset_type,
            train_dataset_dir=file_path,
            column_name1=request.form.get('column_1'),
            column_name2=request.form.get('column_name2'),
            start1=request.form.get('start1'),
            start2=request.form.get('start2'),
            end1=request.form.get('end1'),
            end2=request.form.get('end2'),
            target_name=request.form.get('target_name')
        )   

    return  'title: %s<br> config: %s' % (dataset_type, config)

if __name__ == '__main__':
    app.run(debug=True)
