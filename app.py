from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    FLAG_CONT_MOBILE = int(request.form.get('FLAG_CONT_MOBILE',False))
    FLAG_MOBIL = int(request.form.get('FLAG_MOBIL',False))
    FLAG_EMP_PHONE = int(request.form.get('FLAG_EMP_PHONE',False))
    NAME_TYPE_SUITE = int(request.form.get('NAME_TYPE_SUITE',False))
    NAME_EDUCATION_TYPE = int(request.form.get('NAME_EDUCATION_TYPE',False))
    NAME_HOUSING_TYPE = int(request.form.get('NAME_HOUSING_TYPE',False))
    REGION_RATING_CLIENT_W_CITY = int(request.form.get('REGION_RATING_CLIENT_W_CITY',False))
    REGION_RATING_CLIENT	 = int(request.form.get('REGION_RATING_CLIENT',False))
    FLAG_DOCUMENT_3 = int(request.form.get('FLAG_DOCUMENT_3',False))
    FLAG_OWN_REALTY = int(request.form.get('FLAG_OWN_REALTY',False))
    EMERGENCYSTATE_MODE = int(request.form.get('EMERGENCYSTATE_MODE',False))
    NAME_INCOME_TYPE = int(request.form.get('NAME_INCOME_TYPE',False))

    x = np.array([FLAG_CONT_MOBILE,FLAG_MOBIL,FLAG_EMP_PHONE,NAME_TYPE_SUITE,NAME_EDUCATION_TYPE,NAME_HOUSING_TYPE,REGION_RATING_CLIENT_W_CITY,REGION_RATING_CLIENT,
         FLAG_DOCUMENT_3,FLAG_OWN_REALTY,EMERGENCYSTATE_MODE,NAME_INCOME_TYPE]).reshape(1, -1)

    scaler_path = os.path.join('C:/Users/sakshi/Downloads/Credit Card Detection Project/','models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)

    model_path = os.path.join('C:/Users/sakshi/Downloads/Credit Card Detection Project/models/rf.sav')
    rf = joblib.load(model_path)

    y_pred = rf.predict(x)

    # for Credit Card Fraud Detection
    if y_pred == 1:
        return render_template('nofraud.html')
    else:
        return render_template('fraud.html')

if __name__ == "__main__":
    app.run(debug=True,port=7368)
