import sys
#sys.path.insert(0, '../../src')
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import FunctionTransformer

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()



def get_data_transformer_object(self):
    """
    Creates a preprocessor object using FunctionTransformer for numerical and categorical columns.
    
    Returns:
    Pipeline: Preprocessing pipeline that applies transformations to datasets.
    """
    try:
        numerical_columns = ["MolLogP", "MolWt","NumRotatableBonds", "AromaticProportion"]
        categorical_columns = []
        # Define transformation functions for numerical and categorical columns
        def process_numerical_data(df):
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            return num_pipeline.fit_transform(df[self.numerical_columns])

        def process_categorical_data(df):
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            return cat_pipeline.fit_transform(df[self.categorical_columns])

        # FunctionTransformer applies the transformation to the respective columns
        transformers = []
        if self.numerical_columns:
            num_transformer = FunctionTransformer(lambda df: process_numerical_data(df))
            transformers.append(("num_transformer", num_transformer, self.numerical_columns))
            
        if self.categorical_columns:
            cat_transformer = FunctionTransformer(lambda df: process_categorical_data(df))
            transformers.append(("cat_transformer", cat_transformer, self.categorical_columns))

        # Create a custom pipeline that applies the transformations
        preprocessing_pipeline = Pipeline(
            steps=[
                ("transform", FunctionTransformer(lambda df: pd.concat(
                    [pd.DataFrame(process_numerical_data(df)), pd.DataFrame(process_categorical_data(df))], axis=1)))
            ]
        )

        logging.info("Preprocessing pipeline using FunctionTransformer created successfully.")
        return preprocessing_pipeline

    except Exception as e:
        raise CustomException(e, sys)

        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="logS"
            

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
