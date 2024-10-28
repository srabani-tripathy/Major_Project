import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'tempmax','dew','humidity','windgust',
                'windspeed','Heat_Index','Severity_Score'
            ]    

            
                

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",KNNImputer(n_neighbors=5)),
                    ("scaler",StandardScaler())
                ]
            )

            

            
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)
                ]
            )

            return preprocessor
                    
                    
                
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")
            logging.info("Dropping unnecessary columns")

            columns_to_drop = [
                    'datetime', 'sunrise', 'sunset', 
                   'preciptype', 'snowdepth', 'stations', 
                   'Condition_Code', 'conditions', 'description', 
                   'icon', 'source', 'City', 'Season', 'Day_of_Week',
                   'sunriseEpoch', 'sunsetEpoch', 'datetimeEpoch',
                   'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
                   'precip', 'precipprob', 'precipcover', 'snow',
                   'winddir', 'pressure', 'cloudcover', 'visibility',
                   'solarradiation', 'solarenergy', 'severerisk',
                   'moonphase', 'Month', 'Is_Weekend', 'tempmin',
                   'Temp_Range', 'uvindex'
            ]

            for df in [train_df, test_df]:
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

             # Replace infinite values with NaN to allow KNNImputer to handle them
            for df in [train_df, test_df]:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Log the number of missing values after dropping columns
            logging.info(f"Train missing values: {train_df.isnull().sum().sum()}")
            logging.info(f"Test missing values: {test_df.isnull().sum().sum()}")
            
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = 'Health_Risk_Score'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)