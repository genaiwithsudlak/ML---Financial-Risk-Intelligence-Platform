import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.fraud.train_ieee_model import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(data_folder: str = None):
    """
    Orchestrates the IEEE-CIS fraud detection pipeline.
    """
    if data_folder is None:
        # Default path
        data_folder = r"c:\GitHub_Master\ML---Financial-Risk-Intelligence-Platform\data\raw\Fraud Detection\IEEE-CIS"
    
    logger.info("Starting IEEE-CIS Fraud Detection Pipeline...")
    logger.info(f"Using data folder: {data_folder}")
    
    try:
        train_model(data_folder)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IEEE-CIS Fraud Detection Pipeline")
    parser.add_argument("--data_folder", type=str, help="Folder containing IEEE CSVs", default=None)
    
    args = parser.parse_args()
    run_pipeline(args.data_folder)
