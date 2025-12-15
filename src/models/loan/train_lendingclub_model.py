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
from sklearn.metrics import roc_auc_score, classification_report

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.data.make_dataset import load_lendingclub_data
from src.preprocessing.lendingclub_preprocessing import get_lendingclub_preprocessing_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, test_size: float = 0.2, sample_size: int = 200000):
    """
    Trains XGBoost model for Lending Club.
    Uses a sample of data if dataset is too large for quick MVP.
    """
    try:
        # 1. Load Data
        df = load_lendingclub_data(data_path)
        
        # 2. Filter Target
        # Interest is only in Completed loans: Fully Paid vs Charged Off / Default
        valid_status = ['Fully Paid', 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off']
        df = df[df['loan_status'].isin(valid_status)].copy()
        
        # Map to Binary: 0 = Paid, 1 = Default/Charged Off
        def map_target(status):
            if 'Fully Paid' in status:
                return 0
            return 1
            
        df['target'] = df['loan_status'].apply(map_target)
        logger.info(f"Filtered to {len(df)} completed loans.")
        logger.info(f"Target Distribution: \n{df['target'].value_counts(normalize=True)}")
        
        # Downsample for speed if needed (The file is 1.6GB, usually 2M rows)
        if len(df) > 50000:
            logger.info(f"Downsampling to 50000 for faster MVP training...")
            df = df.sample(n=50000, random_state=42)
        
        X = df.drop(columns=['target', 'loan_status'], errors='ignore')
        y = df['target']
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 4. Pipeline
        preprocessor = get_lendingclub_preprocessing_pipeline()
        
        ratio = float(np.sum(y == 0)) / np.sum(y == 1)
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=ratio,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 5. MLflow
        mlflow.set_experiment("Loan_Default_LendingClub")
        
        with mlflow.start_run():
            logger.info("Starting training...")
            model_pipeline.fit(X_train, y_train)
            
            # 6. Evaluation
            y_prob = model_pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            
            logger.info(f"Test AUC: {auc}")
            mlflow.log_metric("roc_auc", auc)
            
            mlflow.sklearn.log_model(model_pipeline, "lendingclub_model")
            logger.info("Model logged to MLflow.")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    DEFAULT_PATH = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Loan Default Prediction\Lending Club Loan Data\accepted_2007_to_2018Q4.csv"
    train_model(DEFAULT_PATH)
