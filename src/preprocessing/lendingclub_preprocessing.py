import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class LendingClubFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineering for Lending Club.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            # 1. Term parsing: " 36 months" -> 36
            if 'term' in X.columns:
                X['term'] = X['term'].astype(str).str.extract(r'(\d+)').astype(float)
            
            # 2. Emp Length parsing: "10+ years" -> 10, "< 1 year" -> 0
            if 'emp_length' in X.columns:
                X['emp_length'] = X['emp_length'].astype(str).str.replace(r'\D', '', regex=True)
                X['emp_length'] = pd.to_numeric(X['emp_length'], errors='coerce')
            
            return X
        except Exception as e:
            logger.error(f"Error in Lending Club Feature Engineering: {e}")
            raise e

def get_lendingclub_preprocessing_pipeline():
    """
    Returns the Lending Club preprocessing pipeline.
    """
    numerical_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'term', 'emp_length']
    categorical_features = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'application_type']

    # 1. Feature Engineering
    feature_engineering = LendingClubFeatureEngineer()
    
    # 2. Transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 3. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop all other columns (including leakage like recoveries)
    )

    # 4. Full Pipeline
    pipeline = Pipeline(steps=[
        ('features', feature_engineering),
        ('preprocessor', preprocessor)
    ])

    return pipeline
