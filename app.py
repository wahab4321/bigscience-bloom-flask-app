from flask import Flask, request, render_template
app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Hello, there!'

# @app.route('/')
# def hello_world():
#     return render_template('index.html', value='well well ok')

# @app.route('/', methods=['GET', 'POST'])
# def hello_world():
#  if request.method == 'GET':
#   return render_template('index.html', value='hi')
#  if request.method == 'POST':
#   return render_template('result.html',text ="text")



from transformers import pipeline
import numpy as np
from random import choice


def get_pred(text, model, p=0.7):
    res = model(text, do_sample=True, min_length=20)
    return res[0]['generated_text']

@app.route('/', methods=['GET', 'POST'])
def gpt_predictor(n=3):
    if request.method == 'GET':
        return render_template('index.html', value='hi')

    if request.method == 'POST':
        model = pipeline('text-generation', model='bigscience/bloom')
        text = request.form.get('text')
        n = request.form.get('n')
        n = 1
        pred = get_pred(text, model)
        text = pred
        return render_template('result.html',text = text)


if __name__ == '__main__':
    app.run(debug=True)


