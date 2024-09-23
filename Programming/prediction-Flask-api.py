
from flask import Flask,request,render_template,jsonify
from src.exception import CustomException
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    try:
        if request.method=='GET':
            return render_template('home.html')
        else:
            data = request.get_json()
            print("X1")
            data= {
            "MolLogP":[data.get('MolLogP')],
            "MolWt":[data.get('MolWt')],
            "NumRotatableBonds":[data.get('NumRotatableBonds')],
            "AromaticProportion":[data.get('AromaticProportion')],            }
                   
            #pred_df=data.get_data_as_data_frame()
            pred_df=pd.DataFrame(data)
            print(pred_df)
            print("Before Prediction")
            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results= predict_pipeline.predict(pred_df)
            print("after Prediction")
            return jsonify({"answer": results[0]}), 200
            #return jsonify({"answer": response["result"]}), 200
            #return render_template('home.html',results=results[0])
    except Exception as e:
        raise CustomException(e,sys)  

if __name__=="__main__":
    app.run(host="0.0.0.0")        


