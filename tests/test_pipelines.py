import pytest
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.fraud.train_fraud_model import train_model as train_cc
# from src.models.fraud.train_ieee_model import train_model as train_ieee
from src.models.fraud.train_paysim_model import train_model as train_paysim
from src.models.loan.train_loan_model import train_model as train_gmsc
from src.models.loan.train_homecredit_model import train_model as train_home
from src.models.loan.train_lendingclub_model import train_model as train_lc

# Note: We skip IEEE train smoke test if importing it is hard, or we mock it.
# train_ieee needs a folder path.

def test_pipeline_cc(sample_data_dir):
    data_path = sample_data_dir / "credit_card_transactions.csv"
    # Run with small sample (file is small already)
    # The functions should use the path provided
    try:
        train_cc(str(data_path))
    except Exception as e:
        pytest.fail(f"CC Pipeline failed: {e}")

def test_pipeline_paysim(sample_data_dir):
    data_path = sample_data_dir / "PS_sample.csv"
    try:
        train_paysim(str(data_path))
    except Exception as e:
        pytest.fail(f"PaySim Pipeline failed: {e}")

def test_pipeline_gmsc(sample_data_dir):
    data_path = sample_data_dir / "cs-training.csv"
    try:
        train_gmsc(str(data_path))
    except Exception as e:
        pytest.fail(f"GMSC Pipeline failed: {e}")

def test_pipeline_home(sample_data_dir):
    data_path = sample_data_dir / "application_train.csv"
    try:
        train_home(str(data_path))
    except Exception as e:
        pytest.fail(f"HomeCredit Pipeline failed: {e}")

def test_pipeline_lc(sample_data_dir):
    data_path = sample_data_dir / "loan.csv"
    try:
        # Lending Club logic needs 'Fully Paid' etc.
        # Conftest provides 'Fully Paid'
        train_lc(str(data_path), sample_size=10) # Minimal sample
    except Exception as e:
        pytest.fail(f"LendingClub Pipeline failed: {e}")
