import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class TimeAndAgeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract time-based features and calculate age.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            # 1. Date Conversions
            if 'trans_date_trans_time' in X.columns:
                X['trans_date_trans_time'] = pd.to_datetime(X['trans_date_trans_time'])
                X['hour'] = X['trans_date_trans_time'].dt.hour
                X['day'] = X['trans_date_trans_time'].dt.day
                X['month'] = X['trans_date_trans_time'].dt.month
                X['year'] = X['trans_date_trans_time'].dt.year
                X['dayofweek'] = X['trans_date_trans_time'].dt.dayofweek
            
            # 2. Age Calculation
            if 'dob' in X.columns and 'trans_date_trans_time' in X.columns:
                X['dob'] = pd.to_datetime(X['dob'])
                X['card_holder_age'] = (X['trans_date_trans_time'] - X['dob']).dt.days // 365
            
            # 3. Drop unused columns (High cardinality ID-like columns or PII)
            cols_to_drop = [
                "Unnamed: 0", "trans_date_trans_time", "dob", "trans_num", "unix_time", 
                "first", "last", "street", "city", "state", "zip", "merch_zipcode", "cc_num"
            ]
            X.drop(columns=[c for c in cols_to_drop if c in X.columns], inplace=True, errors='ignore')
            
            logger.info(f"Feature engineering complete. Columns: {list(X.columns)}")
            return X
        except Exception as e:
            logger.error(f"Error in Feature Engineering: {e}")
            raise e

def get_fraud_preprocessing_pipeline():
    """
    Returns the complete preprocessing pipeline.
    """
    # Columns requiring specific handling
    # Note: These column names must match what produces TimeAndAgeFeatureEngineer
    categorical_features = ['merchant', 'category', 'gender', 'job']
    numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'card_holder_age', 'hour', 'day', 'month', 'year', 'dayofweek']

    # 1. Feature Engineering Step
    feature_engineering = TimeAndAgeFeatureEngineer()

    # 2. Transformers for specific columns
    # We use OrdinalEncoder for tree-based models (XGBoost) as it preserves information better than OneHot for high cardinality (like merchant/job) without exploding dim
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)) 
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop anything else not specified
    )

    # 4. Full Pipeline
    pipeline = Pipeline(steps=[
        ('features', feature_engineering),
        ('preprocessor', preprocessor)
    ])

    return pipeline
