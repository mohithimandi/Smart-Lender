import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open(r'rdf.pkl', 'rb'))
scale = pickle.load(open(r'scale1.pkl', 'rb'))

@app.route('/')  # rendering the home template
def home():
    return render_template('home.html')

@app.route('/predict', methods=["POST", "GET"])  # rendering the input template
def predict():
    return render_template("input.html")

@app.route('/submit', methods=["POST", "GET"])  # route to show predictions in web UI
def submit():
    # reading the inputs given by the user
    input_feature = [int(x) for x in request.form.values()]
    input_feature = [np.array(input_feature)]
    print(input_feature)

    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
             'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data = pd.DataFrame(input_feature, columns=names)
    print(data)

    # scale the data
    data_scaled = scale.transform(data)

    # predictions using the loaded model file
    prediction = model.predict(data_scaled)
    print(prediction)
    prediction = int(prediction[0])
    print(type(prediction))

    if prediction == 0:
        return render_template("output.html", result="Loan will NOT be Approved ❌")
    else:
        return render_template("output.html", result="Loan will be Approved ✅")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)
