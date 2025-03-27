from flask import Flask, render_template, request # type: ignore
import pickle
import numpy as np # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Class mapping
class_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Define the home route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Define the predict route to process form data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare the input for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict the class
        predicted_class = model.predict(features)[0]
        species = class_mapping[predicted_class]

        # Render result.html with the predicted species
        return render_template('result.html', species=species)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('result.html', species="Error: Unable to predict. Check your input.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    print("successfully implemented")
