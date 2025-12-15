import optuna
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.make_dataset import load_fraud_data, load_ieee_data, load_paysim_data, load_gmsc_data, load_homecredit_data, load_lendingclub_data
from src.preprocessing.credit_card_preprocessing import get_preprocessing_pipeline as get_cc_pipe
from src.preprocessing.paysim_preprocessing import get_paysim_preprocessing_pipeline as get_ps_pipe
from src.preprocessing.loan_preprocessing import get_loan_preprocessing_pipeline as get_gmsc_pipe
# Need strict imports or dynamic logic. For MVP, let's focus on LendingClub since it's the current one.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, X_train, y_train, X_valid, y_valid):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    accuracy_history = []
    
    # Pruning callback
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    
    bst = xgb.train(param, dtrain, num_boost_round=1000, evals=[(dvalid, "validation")], 
                    early_stopping_rounds=50, verbose_eval=False, callbacks=[pruning_callback])
                    
    preds = bst.predict(dvalid)
    auc = roc_auc_score(y_valid, preds)
    return auc

def tune_lending_club():
    data_path = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Loan Default Prediction\Lending Club Loan Data\accepted_2007_to_2018Q4.csv"
    
    # Load Sample
    df = load_lendingclub_data(data_path)
    # Downsample for Tuning Speed
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    # Preprocessing (Quick & Dirty for Tuning - reuse logic)
    # Filter Target
    valid_status = ['Fully Paid', 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off']
    df = df[df['loan_status'].isin(valid_status)].copy()
    def map_target(status):
        if 'Fully Paid' in status: return 0
        return 1
    df['target'] = df['loan_status'].apply(map_target)
    
    X = df.drop(columns=['target', 'loan_status'], errors='ignore')
    y = df['target']
    
    # Feature Engineering (Manual or Pipeline? Pipeline is complex to unpack for DMatrix if pure XGB)
    # For Optuna, we usually want to search parameters. 
    # Let's use the sklearn pipeline wrapper inside objective if possible, OR just use the preprocessor separately.
    
    from src.preprocessing.lendingclub_preprocessing import get_lendingclub_preprocessing_pipeline
    preprocessor = get_lendingclub_preprocessing_pipeline()
    
    X_proc = preprocessor.fit_transform(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)
    
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.params}")
    
    return study.best_trial.params

if __name__ == "__main__":
    print("Starting hyperparameter tuning for Lending Club...")
    best_params = tune_lending_club()
    print(f"Best Params: {best_params}")
