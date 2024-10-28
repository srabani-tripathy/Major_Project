from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            tempmax=float(request.form.get('tempmax')),
            dew=float(request.form.get('dew')),
            humidity=float(request.form.get('humidity')),
            windgust=float(request.form.get('windgust')),
            windspeed=float(request.form.get('windspeed')),
            Heat_Index=float(request.form.get('Heat_Index')),
            Severity_Score=float(request.form.get('Severity_Score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        