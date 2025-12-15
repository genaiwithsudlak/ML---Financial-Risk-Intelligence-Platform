import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.make_dataset import load_fraud_data, load_ieee_data, load_paysim_data, load_gmsc_data, load_homecredit_data, load_lendingclub_data

def test_load_fraud_data(sample_data_dir):
    data_path = sample_data_dir / "credit_card_transactions.csv"
    df = load_fraud_data(str(data_path))
    assert not df.empty
    assert 'amt' in df.columns
    assert 'is_fraud' in df.columns
    assert len(df) == 50

def test_load_ieee_data(sample_data_dir):
    data_dir = sample_data_dir
    df = load_ieee_data(str(data_dir))
    assert not df.empty
    assert 'TransactionAmt' in df.columns
    assert 'DeviceInfo' in df.columns # Merged check
    assert len(df) == 50

def test_load_paysim_data(sample_data_dir):
    data_path = sample_data_dir / "PS_sample.csv"
    df = load_paysim_data(str(data_path))
    assert not df.empty
    assert 'type' in df.columns
    assert len(df) == 50

def test_load_gmsc_data(sample_data_dir):
    data_path = sample_data_dir / "cs-training.csv"
    df = load_gmsc_data(str(data_path))
    assert not df.empty
    assert 'SeriousDlqin2yrs' in df.columns
    assert len(df) == 50

def test_load_homecredit_data(sample_data_dir):
    data_path = sample_data_dir / "application_train.csv"
    df = load_homecredit_data(str(data_path))
    assert not df.empty
    assert 'TARGET' in df.columns
    assert len(df) == 50

def test_load_lendingclub_data(sample_data_dir):
    data_path = sample_data_dir / "loan.csv"
    df = load_lendingclub_data(str(data_path))
    assert not df.empty
    assert 'loan_status' in df.columns
    assert len(df) == 50
