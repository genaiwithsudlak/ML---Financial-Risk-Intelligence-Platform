import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class GMSCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature Engineer for Give Me Some Credit.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            # 1. Cap Outliers
            # Columns like NumberOfTime30-59Days... have 96/98 as codes for something else or error
            outlier_cols = [
                'NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate'
            ]
            for col in outlier_cols:
                if col in X.columns:
                    X.loc[X[col] > 90, col] = 0 # Replace code with 0 or mean
            
            # 2. Income Per Person (Interactive feature)
            if 'MonthlyIncome' in X.columns and 'NumberOfDependents' in X.columns:
                # Fill check to avoid div by zero/nan handled later
                # Here we just create the feature if possible, but imputation happens next in pipeline
                pass 
                
            return X
        except Exception as e:
            logger.error(f"Error in GMSC Feature Engineering: {e}")
            raise e

def get_loan_preprocessing_pipeline():
    """
    Returns the GMCS preprocessing pipeline.
    """
    numerical_features = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 
        'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
        'NumberOfDependents'
    ]

    # 1. Feature Engineering
    feature_engineering = GMSCFeatureEngineer()
    
    # 2. Transformers
    # Separate imputer for different columns if needed, but SimpleImputer(median) works well for all here
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'
    )

    # 4. Full Pipeline
    pipeline = Pipeline(steps=[
        ('features', feature_engineering),
        ('preprocessor', preprocessor)
    ])

    return pipeline
