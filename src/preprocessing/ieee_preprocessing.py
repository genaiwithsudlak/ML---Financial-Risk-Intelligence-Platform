import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class IEEEFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects relevant features and drops IDs/High-Null columns.
    """
    def __init__(self, null_threshold: float = 0.9):
        self.null_threshold = null_threshold
        self.features_to_keep = []
        
    def fit(self, X, y=None):
        # Identify columns with too many nulls
        null_counts = X.isnull().mean()
        self.features_to_keep = list(null_counts[null_counts < self.null_threshold].index)
        
        # Explicitly drop ID columns if present
        drop_cols = ['TransactionID', 'TransactionDT', 'isFraud']
        self.features_to_keep = [f for f in self.features_to_keep if f not in drop_cols]
        
        return self

    def transform(self, X):
        return X[self.features_to_keep]

def get_ieee_preprocessing_pipeline():
    """
    Returns the preprocessing pipeline for IEEE-CIS.
    """
    # 1. Selector Step (Fit logic required to know which columns to keep)
    # Note: scikit-learn pipelines usually require static column definitions for ColumnTransformer
    # For simplicity in this template, we will rely on LightGBM's ability to handle NaNs 
    # and just do Ordinal Encoding for strings.
    
    class StringIndexer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = X.copy()
            # Convert object columns to category codes or fillna
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].fillna("unknown").astype(str)
            return X

    # Simple pipeline: Object -> String -> Ordinal
    # Numerics -> Keep as is (LightGBM handles NaNs)
    
    pipeline = Pipeline(steps=[
        ('indexer', StringIndexer()),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # In a full production system, we would have a more complex ColumnTransformer here.
    # But for IEEE-CIS, simply encoding strings to Ints effectively allows LightGBM to work.
    
    return pipeline
