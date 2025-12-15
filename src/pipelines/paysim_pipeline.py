import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.fraud.train_paysim_model import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(data_path: str = None):
    """
    Orchestrates the PaySim fraud detection pipeline.
    """
    if data_path is None:
        # Default path
        data_path = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\Synthetic Financial Fraud Detection\PS_20174392719_1491204439457_log.csv"
    
    logger.info("Starting PaySim Fraud Detection Pipeline...")
    logger.info(f"Using data path: {data_path}")
    
    try:
        train_model(data_path)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PaySim Fraud Detection Pipeline")
    parser.add_argument("--data_path", type=str, help="Path to PaySim CSV", default=None)
    
    args = parser.parse_args()
    run_pipeline(args.data_path)
