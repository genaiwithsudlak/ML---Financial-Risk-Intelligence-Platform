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
from src.data.make_dataset import load_gmsc_data
from src.preprocessing.loan_preprocessing import get_loan_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Trains XGBoost model for Give Me Some Credit.
    """
    try:
        # 1. Load Data
        df = load_gmsc_data(data_path)
        
        # 2. Split
        # Target: SeriousDlqin2yrs
        X = df.drop(columns=['SeriousDlqin2yrs', 'Unnamed: 0'], errors='ignore')
        y = df['SeriousDlqin2yrs']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training Data Shape: {X_train.shape}")
        logger.info(f"Test Data Shape: {X_test.shape}")
        
        # 3. Pipeline
        preprocessor = get_loan_preprocessing_pipeline()
        
        ratio = float(np.sum(y == 0)) / np.sum(y == 1)
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
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

        # 4. MLflow
        mlflow.set_experiment("Loan_Default_GMSC")
        
        with mlflow.start_run():
            logger.info("Starting training...")
            model_pipeline.fit(X_train, y_train)
            
            # 5. Evaluation
            y_prob = model_pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            metrics = {"roc_auc": auc, "pr_auc": pr_auc}
            logger.info(f"Metrics: {metrics}")
            mlflow.log_metrics(metrics)
            
            mlflow.sklearn.log_model(model_pipeline, "gmsc_loan_model")
            logger.info("Model logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    DEFAULT_PATH = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Loan Default Prediction\Give Me Some Credit\cs-training.csv"
    train_model(DEFAULT_PATH)
