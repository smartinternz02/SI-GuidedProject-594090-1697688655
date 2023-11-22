from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input features from the form
        features = [float(request.form['age']),
                    float(request.form['sex']),
                    float(request.form['tsh']),
                    float(request.form['t3']),
                    float(request.form['tt4']),
                    float(request.form['t4u']),
                    float(request.form['fti'])]

        # Convert the features to a DataFrame
        input_data = pd.DataFrame([features])

        # Make predictions using the model
        prediction = model.predict(input_data)[0]

        return render_template('submit.html', prediction=prediction)

    return render_template('predict.html')
@app.route('/submit')
def submit():
    if request.method == 'POST':
        # Get the prediction from the form data
        prediction = request.form.get('prediction')
        
        return render_template('submit.html', prediction=prediction)
    else:
        # If the form is not submitted, render the page without the prediction
        return render_template('submit.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
