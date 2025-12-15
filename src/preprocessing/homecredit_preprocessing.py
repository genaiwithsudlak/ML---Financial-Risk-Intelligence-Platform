import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class HomeCreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineering for Home Credit.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            # Domain Knowledge Features
            # 1. Credit Income Percent: How much credit relative to income
            X['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1e-5)
            
            # 2. Annuity Income Percent: Loan burden
            X['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1e-5)
            
            # 3. Credit Term: How long to pay back
            X['CREDIT_TERM'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + 1e-5)
            
            # 4. Days Employed Percent: How long employed relative to age
            X['DAYS_EMPLOYED_PERCENT'] = X['DAYS_EMPLOYED'] / (X['DAYS_BIRTH'] + 1e-5)
            
            # 5. Fix Days Employed 365243 -> NaN (Common quirk in this dataset)
            X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
            
            return X
        except Exception as e:
            logger.error(f"Error in Home Credit Feature Engineering: {e}")
            raise e

def get_homecredit_preprocessing_pipeline():
    """
    Returns the Home Credit preprocessing pipeline.
    """
    # LightGBM handles categories well, so simple ordinal encoding is often enough.
    # However, for Scikit-Pipeline compatibility, we define a robust flow.
    
    # We will let the user pass the dataframe to fit, but hardcoding some known num/cat cols for MVP
    # Ideally, we should detect them dynamically in a 'Selector' step, but we define generic transformers here.
    
    class TypeSelector(BaseEstimator, TransformerMixin):
        def __init__(self, dtype):
            self.dtype = dtype
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X.select_dtypes(include=self.dtype)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # We use a trick to apply transformations based on types dynamically if we don't list all 120 cols
    # But ColumnTransformer needs names. For this MVP, we will use a global strategy:
    # "Engineer Features" -> "Auto-detect types and Encode/Scale"
    
    # Since ColumnTransformer requires explicit columns, and we have 120+, 
    # we will implement a Custom Preprocessor that wraps the logic.
    
    class AutoPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.num_pipeline = numerical_transformer
            self.cat_pipeline = categorical_transformer
            self.num_cols = []
            self.cat_cols = []
            
        def fit(self, X, y=None):
            self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if self.num_cols:
                self.num_pipeline.fit(X[self.num_cols])
            if self.cat_cols:
                self.cat_pipeline.fit(X[self.cat_cols])
            return self
            
        def transform(self, X):
            X_out = X.copy()
            if self.num_cols:
                X_out[self.num_cols] = self.num_pipeline.transform(X[self.num_cols])
            if self.cat_cols:
                X_out[self.cat_cols] = self.cat_pipeline.transform(X[self.cat_cols])
            return X_out

    pipeline = Pipeline(steps=[
        ('features', HomeCreditFeatureEngineer()),
        ('preprocessor', AutoPreprocessor())
    ])

    return pipeline
