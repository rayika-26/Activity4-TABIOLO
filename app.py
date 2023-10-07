import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder='template')

# Load the trained model
model = pickle.load(open('model/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])

    # Create a NumPy array from the input features
    features = np.array([[feature1, feature2, feature3]])

    # Make the prediction using the loaded model
    prediction = model.predict(features)

    # Extract the prediction value as a regular Python float
    predicted_value = float(prediction[0])

    return render_template('index.html', prediction_text=f'Predicted Profit is {predicted_value}')


if __name__ == "__main__":
    app.run(debug=True)
