from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import os
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "4Etlh_YbuFRyyeW28vCEn3hmQ5-LIOtAtJYqy4si80s3"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
app = Flask(__name__)
model = pickle.load(open('PCA_model.pkl','rb'))
@app.route('/')
def home():
    return render_template("pca.html")
@app.route('/predict',methods=["POST","GET"])

def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=['Global_reactive_power','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    df=pd.DataFrame(features_value,columns=features_name)
    payload_scoring = {"input_data": [{"fields": [['Global_reactive_power','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']], "values": [input_features]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/a916b21e-0e3d-4233-b3f9-6587a39fdf07/predictions?version=2021-07-02', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
   
    output = model.predict(df)
    
    
    return render_template('result1.html',prediction_text=output)

if __name__=="__main__":
    app.run(debug=False)
    
