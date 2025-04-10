from flask import Flask, request, jsonify, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model

app = Flask(__name__)
model = load_model('iris_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Get prediction
    prediction = predict_model(model, data=input_data)
    result = prediction['prediction_label'][0]

    return render_template('index.html', prediction=result)

# Ensure the Flask app runs only when the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

