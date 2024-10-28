import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        tempmax:float,
        dew:float,
        humidity:float,
        windgust:float,
        windspeed:float,
        Heat_Index:float,
        Severity_Score:float):

        self.tempmax = tempmax

        self.dew = dew

        self.humidity = humidity

        self.windgust = windgust

        self.windspeed = windspeed

        self.Heat_Index = Heat_Index

        self.Severity_Score = Severity_Score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "tempmax":[self.tempmax],
                "dew":[self.dew],
                "humidity":[self.humidity],
                "windgust":[self.windgust],
                "windspeed":[self.windspeed],
                "Heat_Index":[self.Heat_Index],
                "Severity_Score":[self.Severity_Score],
            }

            return pd.DataFrame(custom_data_input_dict)
        

        except Exception as e:
            raise CustomException(e,sys)
        