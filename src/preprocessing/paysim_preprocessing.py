import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class PaySimFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering for PaySim.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            # 1. Error Balance Features
            # Flagged fraud is often where the transaction clears out an account or moves more than expected
            X['errorBalanceOrig'] = X['newbalanceOrig'] + X['amount'] - X['oldbalanceOrg']
            X['errorBalanceDest'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
            
            # 2. Step conversion
            # Step maps to 1 hour of time.
            X['hour_of_day'] = X['step'] % 24
            
            # 3. Drop irrelevant columns
            cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud'] # Dropping leakage/ID columns
            X.drop(columns=[c for c in cols_to_drop if c in X.columns], inplace=True, errors='ignore')
            
            return X
        except Exception as e:
            logger.error(f"Error in PaySim Feature Engineering: {e}")
            raise e

def get_paysim_preprocessing_pipeline():
    """
    Returns the PaySim preprocessing pipeline.
    """
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest', 'hour_of_day']
    categorical_features = ['type'] # TRANSFER, CASH_OUT etc.

    # 1. Feature Engineering
    feature_engineering = PaySimFeatureEngineer()
    
    # 2. Transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 4. Full Pipeline
    pipeline = Pipeline(steps=[
        ('features', feature_engineering),
        ('preprocessor', preprocessor)
    ])

    return pipeline
