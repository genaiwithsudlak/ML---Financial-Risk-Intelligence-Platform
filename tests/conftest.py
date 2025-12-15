import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import mlflow


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory):
    """Creates a temporary directory with sample CSVs for all datasets."""
    data_dir = tmp_path_factory.mktemp("data")
    
    # 1. Credit Card
    df_fraud = pd.DataFrame({
        'trans_date_trans_time': ['2023-01-01 12:00:00'] * 50,
        'cc_num': np.random.randint(1000, 9999, 50),
        'merchant': ['M_ID_1'] * 50,
        'category': ['grocery_pos'] * 50,
        'amt': np.random.uniform(10, 500, 50),
        'first': ['John'] * 50,
        'last': ['Doe'] * 50,
        'gender': ['M'] * 50,
        'street': ['Street'] * 50,
        'city': ['City'] * 50,
        'state': ['ST'] * 50,
        'zip': [10001] * 50,
        'lat': [40.0] * 50,
        'long': [-70.0] * 50,
        'city_pop': [10000] * 50,
        'job': ['Job'] * 50,
        'dob': ['1990-01-01'] * 50,
        'trans_num': [f't_{i}' for i in range(50)],
        'unix_time': [1600000000] * 50,
        'merch_lat': [40.0] * 50,
        'merch_long': [-70.0] * 50,
        'is_fraud': [0] * 50
    })
    # Add fraud cases (needs >=2 for StratifiedShuffleSplit)
    df_fraud.loc[0:2, 'is_fraud'] = 1
    df_fraud.to_csv(data_dir / "credit_card_transactions.csv", index=False)
    
    # 2. IEEE-CIS
    df_ieee_trans = pd.DataFrame({
        'TransactionID': range(100000, 100050),
        'isFraud': [0] * 50,
        'TransactionDT': range(50),
        'TransactionAmt': np.random.uniform(10, 100, 50),
        'ProductCD': ['W'] * 50,
        'card1': [1000] * 50,
        'card2': [100.0] * 50,
        'card3': [150.0] * 50,
        'card4': ['visa'] * 50,
        'card5': [100.0] * 50,
        'card6': ['debit'] * 50,
        'addr1': [300.0] * 50,
        'addr2': [87.0] * 50,
        'dist1': [10.0] * 50,
        'dist2': [10.0] * 50,
        'P_emaildomain': ['gmail.com'] * 50,
        'R_emaildomain': ['gmail.com'] * 50,
    })
    # Add dummy C and D columns as required by pipeline likely
    for i in range(1, 15):
        df_ieee_trans[f'C{i}'] = 1.0
        df_ieee_trans[f'D{i}'] = 0.0
    for i in range(1, 10):
        df_ieee_trans[f'V{i}'] = 0.0 # Minimal V cols
        
    df_ieee_ident = pd.DataFrame({
        'TransactionID': range(100000, 100050),
        'DeviceType': ['mobile'] * 50,
        'DeviceInfo': ['iOS'] * 50
    })
    df_ieee_trans.to_csv(data_dir / "train_transaction.csv", index=False)
    df_ieee_ident.to_csv(data_dir / "train_identity.csv", index=False)
    
    # 3. PaySim
    df_paysim = pd.DataFrame({
        'step': range(1, 51),
        'type': ['TRANSFER'] * 50, # CHANGED from PAYMENT to TRANSFER for filtering
        'amount': [100.0] * 50,
        'nameOrig': ['C1'] * 50,
        'oldbalanceOrg': [100.0] * 50,
        'newbalanceOrig': [0.0] * 50,
        'nameDest': ['M1'] * 50,
        'oldbalanceDest': [0.0] * 50,
        'newbalanceDest': [0.0] * 50,
        'isFraud': [0] * 50,
        'isFlaggedFraud': [0] * 50
    })
    df_paysim.to_csv(data_dir / "PS_sample.csv", index=False)
    
    # 4. Give Me Some Credit
    df_gmsc = pd.DataFrame({
        'SeriousDlqin2yrs': [0] * 50,
        'RevolvingUtilizationOfUnsecuredLines': [0.5] * 50,
        'age': [30] * 50,
        'NumberOfTime30-59DaysPastDueNotWorse': [0] * 50,
        'DebtRatio': [0.3] * 50,
        'MonthlyIncome': [5000.0] * 50,
        'NumberOfOpenCreditLinesAndLoans': [5] * 50,
        'NumberOfTimes90DaysLate': [0] * 50,
        'NumberRealEstateLoansOrLines': [1] * 50,
        'NumberOfTime60-89DaysPastDueNotWorse': [0] * 50,
        'NumberOfDependents': [0] * 50
    })
    # Add positive class for stratify split
    df_gmsc.loc[0:2, 'SeriousDlqin2yrs'] = 1
    df_gmsc.to_csv(data_dir / "cs-training.csv", index=False)
    
    # 5. Home Credit
    df_home = pd.DataFrame({
        'SK_ID_CURR': range(100, 150),
        'TARGET': [0] * 50,
        'NAME_CONTRACT_TYPE': ['Cash loans'] * 50,
        'CODE_GENDER': ['M'] * 50,
        'FLAG_OWN_CAR': ['N'] * 50,
        'FLAG_OWN_REALTY': ['Y'] * 50,
        'CNT_CHILDREN': [0] * 50,
        'AMT_INCOME_TOTAL': [50000.0] * 50,
        'AMT_CREDIT': [100000.0] * 50,
        'AMT_ANNUITY': [5000.0] * 50,
        'AMT_GOODS_PRICE': [100000.0] * 50,
        'NAME_TYPE_SUITE': ['Unaccompanied'] * 50,
        'NAME_INCOME_TYPE': ['Working'] * 50,
        'NAME_EDUCATION_TYPE': ['Secondary / secondary special'] * 50,
        'NAME_FAMILY_STATUS': ['Single / not married'] * 50,
        'NAME_HOUSING_TYPE': ['House / apartment'] * 50,
        'REGION_POPULATION_RELATIVE': [0.02] * 50,
        'DAYS_BIRTH': [-10000] * 50,
        'DAYS_EMPLOYED': [-1000] * 50,
        'DAYS_REGISTRATION': [-100.0] * 50,
        'DAYS_ID_PUBLISH': [-100] * 50,
        'OWN_CAR_AGE': [5.0] * 50,
        'FLAG_MOBIL': [1] * 50,
        'FLAG_EMP_PHONE': [1] * 50,
        'FLAG_WORK_PHONE': [0] * 50,
        'FLAG_CONT_MOBILE': [1] * 50,
        'FLAG_PHONE': [0] * 50,
        'FLAG_EMAIL': [0] * 50,
        'OCCUPATION_TYPE': ['Laborers'] * 50,
        'CNT_FAM_MEMBERS': [1.0] * 50,
        'REGION_RATING_CLIENT': [2] * 50,
        'REGION_RATING_CLIENT_W_CITY': [2] * 50,
        'WEEKDAY_APPR_PROCESS_START': ['MONDAY'] * 50,
        'HOUR_APPR_PROCESS_START': [10] * 50,
        'REG_REGION_NOT_LIVE_REGION': [0] * 50,
        'REG_REGION_NOT_WORK_REGION': [0] * 50,
        'LIVE_REGION_NOT_WORK_REGION': [0] * 50,
        'REG_CITY_NOT_LIVE_CITY': [0] * 50,
        'REG_CITY_NOT_WORK_CITY': [0] * 50,
        'LIVE_CITY_NOT_WORK_CITY': [0] * 50,
        'ORGANIZATION_TYPE': ['Business Entity Type 3'] * 50,
        'EXT_SOURCE_1': [0.5] * 50,
        'EXT_SOURCE_2': [0.5] * 50,
        'EXT_SOURCE_3': [0.5] * 50,
    })
    # Add positive class for stratify
    df_home.loc[0:2, 'TARGET'] = 1
    df_home.to_csv(data_dir / "application_train.csv", index=False)
    
    # 6. Lending Club
    df_lc = pd.DataFrame({
        'loan_amnt': [10000.0] * 50,
        'int_rate': [10.0] * 50,
        'annual_inc': [60000.0] * 50,
        'dti': [15.0] * 50,
        'grade': ['A'] * 50,
        'sub_grade': ['A1'] * 50,
        'term': [' 36 months'] * 50,
        'emp_length': ['10+ years'] * 50,
        'home_ownership': ['RENT'] * 50,
        'verification_status': ['Verified'] * 50,
        'purpose': ['debt_consolidation'] * 50,
        'addr_state': ['CA'] * 50,
        'initial_list_status': ['w'] * 50,
        'application_type': ['Individual'] * 50,
        'open_acc': [10.0] * 50,
        'pub_rec': [0.0] * 50,
        'revol_bal': [5000.0] * 50,
        'revol_util': [50.0] * 50,
        'total_acc': [20.0] * 50,
        'mort_acc': [0.0] * 50,
        'pub_rec_bankruptcies': [0.0] * 50,
        'loan_status': ['Fully Paid'] * 50,
        'installment': [300.0] * 50
    })
    # Add positive class for stratify (Charged Off)
    df_lc.loc[0:2, 'loan_status'] = 'Charged Off'
    df_lc.to_csv(data_dir / "loan.csv", index=False)
    
    return data_dir

@pytest.fixture(autouse=True)
def setup_mlflow(tmp_path):
    """
    Sets up a temporary MLflow tracking URI for each test to avoid 
    pollution and permissions issues on CI/CD (especially with committed mlruns).
    """
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri
