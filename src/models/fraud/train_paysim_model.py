import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.make_dataset import load_paysim_data
from src.preprocessing.paysim_preprocessing import get_paysim_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Trains XGBoost model for PaySim.
    """
    try:
        # 1. Load Data
        df = load_paysim_data(data_path)
        
        # 2. Filter for relevant types (Transfer/CashOut) as per domain knowledge
        # Analysis confirms fraud ONLY happens in these types
        df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
        logger.info(f"Filtered data to {len(df_filtered)} rows (TRANSFER/CASH_OUT only).")
        
        X = df_filtered.drop(columns=['isFraud', 'isFlaggedFraud']) # isFlagged is target leakage/rule-based
        y = df_filtered['isFraud']
        
        # 3. Split
        # Chronological split using 'step'
        split_step = int(df_filtered['step'].max() * (1 - test_size))
        
        train_mask = df_filtered['step'] <= split_step
        test_mask = df_filtered['step'] > split_step
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        logger.info(f"Training Data Shape: {X_train.shape}")
        logger.info(f"Test Data Shape: {X_test.shape}")
        
        # 4. Pipeline
        preprocessor = get_paysim_preprocessing_pipeline()
        
        ratio = float(np.sum(y == 0)) / np.sum(y == 1)
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 5. MLflow
        mlflow.set_experiment("Fraud_Detection_PaySim")
        
        with mlflow.start_run():
            logger.info("Starting training...")
            model_pipeline.fit(X_train, y_train)
            
            # 6. Evaluation
            y_prob = model_pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            metrics = {"roc_auc": auc, "pr_auc": pr_auc}
            logger.info(f"Metrics: {metrics}")
            mlflow.log_metrics(metrics)
            
            mlflow.sklearn.log_model(model_pipeline, "paysim_model")
            logger.info("Model logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    DEFAULT_PATH = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\Synthetic Financial Fraud Detection\PS_20174392719_1491204439457_log.csv"
    train_model(DEFAULT_PATH)
