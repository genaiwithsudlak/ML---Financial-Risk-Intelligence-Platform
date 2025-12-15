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
from src.data.make_dataset import load_ieee_data
from src.preprocessing.ieee_preprocessing import get_ieee_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_folder: str, test_size: float = 0.2):
    """
    Trains LightGBM model for IEEE-CIS Fraud Detection.
    """
    try:
        # 1. Load Data
        df = load_ieee_data(data_folder)
        
        # 2. Time-based Split
        # TransactionDT is seconds from start. Sort by it.
        df = df.sort_values('TransactionDT')
        
        X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'])
        y = df['isFraud']
        
        # Split
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training Data Shape: {X_train.shape}")
        logger.info(f"Test Data Shape: {X_test.shape}")
        
        # 3. Preprocessing
        # We handle categorical encoding before passing to LGBM
        # Note: Ideally LGBM handles categories natively if dtype is 'category', but our pipeline encodes them to ints.
        preprocessor = get_ieee_preprocessing_pipeline()
        
        logger.info("Preprocessing data...")
        # Fit on Train, Transform both
        # Warning: For very large data, this in-memory transformation might OOM.
        # Phase 2 optimization would be chunking.
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # 4. Train LightGBM
        # Create dataset
        dtrain = lgb.Dataset(X_train_proc, label=y_train)
        dtest = lgb.Dataset(X_test_proc, label=y_test, reference=dtrain)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        mlflow.set_experiment("Fraud_Detection_IEEE")
        
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
            
            # Log model
            mlflow.lightgbm.log_model(model, "ieee_model")
            logger.info("Model logged to MLflow.")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    DEFAULT_FOLDER = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\IEEE-CIS"
    train_model(DEFAULT_FOLDER)
