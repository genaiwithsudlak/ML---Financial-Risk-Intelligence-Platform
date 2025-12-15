import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.make_dataset import load_homecredit_data
from src.preprocessing.homecredit_preprocessing import get_homecredit_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, test_size: float = 0.2):
    """
    Trains LightGBM model for Home Credit Default Risk.
    """
    try:
        # 1. Load Data
        df = load_homecredit_data(data_path)
        
        # 2. Split
        # Target is 'TARGET'
        X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        y = df['TARGET']
        
        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training Data Shape: {X_train.shape}")
        
        # 3. Preprocessing
        preprocessor = get_homecredit_preprocessing_pipeline()
        
        logger.info("Preprocessing data...")
        # Fit on Train, Transform both
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # 4. Train LightGBM
        dtrain = lgb.Dataset(X_train_proc, label=y_train)
        dtest = lgb.Dataset(X_test_proc, label=y_test, reference=dtrain)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True # Handle imbalance
        }
        
        mlflow.set_experiment("Loan_Default_HomeCredit")
        
        with mlflow.start_run():
            logger.info("Starting LightGBM training...")
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dtest],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
            )
            
            # 5. Evaluation
            y_prob = model.predict(X_test_proc, num_iteration=model.best_iteration)
            auc = roc_auc_score(y_test, y_prob)
            
            logger.info(f"Test AUC: {auc}")
            mlflow.log_metric("roc_auc", auc)
            
            mlflow.lightgbm.log_model(model, "homecredit_model")
            logger.info("Model logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    DEFAULT_PATH = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Loan Default Prediction\Home Credit Default Risk\application_train.csv"
    train_model(DEFAULT_PATH)
