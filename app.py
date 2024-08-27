from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get input values from the form
        area = float(request.form['Area'])
        production = float(request.form['Production'])
        annual_rainfall = float(request.form['Annual_Rainfall'])
        fertilizer = float(request.form['Fertilizer'])
        pesticide = float(request.form['Pesticide'])

        # Prepare the data for prediction
        input_features = np.array([[area, production, annual_rainfall, fertilizer, pesticide]])
        input_features_scaled = scaler.transform(input_features)

        # Predict
        prediction = model.predict(input_features_scaled)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
