import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the HTML form
    temp = int(request.form['temp'])
    humidity = int(request.form['humidity'])
    og = int(request.form['og'])

    # Make prediction
    final_features = np.array([[og, temp, humidity]])
    prediction = model.predict(final_features)
    output = np.round(prediction[0], 2)
    return render_template('index.html', prediction_text="fire prediction : {} ".format(output))

if __name__ == '__main__':
    app.run()
