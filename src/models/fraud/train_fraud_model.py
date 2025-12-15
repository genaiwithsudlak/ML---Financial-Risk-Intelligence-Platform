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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.make_dataset import load_fraud_data
from src.preprocessing.fraud_preprocessing import get_fraud_preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Trains the XGBoost model for fraud detection.
    """
    try:
        # 1. Load Data
        df = load_fraud_data(data_path)
        
        # 2. Split Data
        # Drop Target
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
        
        # Time-based split preferred for production, but random split for initial MVP verification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training Data Shape: {X_train.shape}")
        logger.info(f"Test Data Shape: {X_test.shape}")

        # 3. Model Definition
        # Using scale_pos_weight to handle class imbalance (approx ratio of legit/fraud)
        # Note: In real production, this should be tuned via hyperparams
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
        
        preprocessor = get_fraud_preprocessing_pipeline()
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 4. MLflow Experiment Setup
        mlflow.set_experiment("Fraud_Detection_CreditCard")
        
        with mlflow.start_run():
            # Log params
            mlflow.log_params({
                "model_type": "XGBoost",
                "test_size": test_size,
                "scale_pos_weight": ratio
            })
            
            # 5. Training
            logger.info("Starting training...")
            model_pipeline.fit(X_train, y_train)
            logger.info("Training complete.")
            
            # 6. Evaluation
            y_pred = model_pipeline.predict(X_test)
            y_proba = model_pipeline.predict_proba(X_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            metrics = {
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            logger.info(f"Metrics: {metrics}")
            mlflow.log_metrics(metrics)
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # 7. Model Artifact Logging
            mlflow.sklearn.log_model(model_pipeline, "fraud_model")
            logger.info("Model logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    # Default path based on earlier checks
    DEFAULT_DATA_PATH = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\Credit Card\credit_card_transactions.csv"
    train_model(DEFAULT_DATA_PATH)
