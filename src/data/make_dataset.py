import pandas as pd
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fraud_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from a CSV file.

    Args:
        filepath (str): Absolute path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If essential columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic Validation
    required_columns = ['trans_date_trans_time', 'amt', 'is_fraud']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def load_ieee_data(folder_path: str) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS Fraud Detection dataset.
    
    Args:
        folder_path (str): Path to folder containing train_transaction.csv and train_identity.csv
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    path = Path(folder_path)
    trans_path = path / "train_transaction.csv"
    ident_path = path / "train_identity.csv"
    
    if not trans_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {trans_path}")
        
    logger.info("Loading IEEE-CIS Transaction Data...")
    df_trans = pd.read_csv(trans_path)
    
    logger.info("Loading IEEE-CIS Identity Data...")
    if ident_path.exists():
        df_ident = pd.read_csv(ident_path)
        logger.info("Merging Transaction and Identity data...")
        df = pd.merge(df_trans, df_ident, on='TransactionID', how='left')
    else:
        logger.warning(f"Identity file not found at {ident_path}. Proceeding with Transaction data only.")
        df = df_trans
        
    logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns.")
    return df

def load_paysim_data(filepath: str) -> pd.DataFrame:
    """
    Load PaySim dataset.
    
    Args:
        filepath (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PaySim file not found: {filepath}")
        
    logger.info("Loading PaySim Data...")
    df = pd.read_csv(filepath)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def load_gmsc_data(filepath: str) -> pd.DataFrame:
    """
    Load Give Me Some Credit dataset.
    
    Args:
        filepath (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GMSC file not found: {filepath}")
        
    logger.info("Loading Give Me Some Credit Data...")
    df = pd.read_csv(filepath)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def load_homecredit_data(filepath: str) -> pd.DataFrame:
    """
    Load Home Credit Default Risk dataset (Application Train).
    
    Args:
        filepath (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Home Credit file not found: {filepath}")
        
    logger.info("Loading Home Credit Data...")
    df = pd.read_csv(filepath)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

def load_lendingclub_data(filepath: str) -> pd.DataFrame:
    """
    Load Lending Club dataset (Accepted Loans).
    
    Args:
        filepath (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Lending Club file not found: {filepath}")
        
    logger.info("Loading Lending Club Data (this may take a while)...")
    # Low_memory=False to handle mixed types
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Successfully loaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    # Test run
    sample_path = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\Credit Card\credit_card_transactions.csv"
    try:
        df = load_fraud_data(sample_path)
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
