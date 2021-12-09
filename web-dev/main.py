from flask import Flask, render_template, request, escape
from scipy.spatial import kdtree
from src.Model import Model
import json

app = Flask(__name__)

@app.route("/")

def hello_world():
    q = request.args.get("q", "")
    k = request.args.get("k", "")
    return render_template("index.html.j2", q=escape(q), data=get_data(query=q, k=k), k=escape(k))

def get_data(query, k):
    k = None if k == '' else int(k) 
    if query != '':
        model = Model() #put your model here
        data = json.load(open("data/some_data.json")) #put your data here
        model.fit(data)
        return model.get_top_k(query=query, k=k)
    else: 
        return []

if __name__ == "__main__":
    app.run(debug=True)

