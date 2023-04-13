import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    #condition of refused or accepted
    if output == 0:
        return render_template('index.html', prediction_text='Your credit has been refused')
    else:
        return render_template('index.html', prediction_text='Your credit has been accepted')

if __name__ == "__main__":
    app.run(debug=True)