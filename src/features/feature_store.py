import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    A simple Feature Store implementation managing Offline (Parquet) and Online (Memory/File) stores.
    """
    def __init__(self, base_path: str = None):
        if base_path:
             self.base_path = Path(base_path)
        else:
             self.base_path = Path(__file__).resolve().parents[2] / "data" / "feature_store"
        
        self.offline_path = self.base_path / "offline"
        self.online_path = self.base_path / "online"
        
        # Ensure directories exist
        self.offline_path.mkdir(parents=True, exist_ok=True)
        self.online_path.mkdir(parents=True, exist_ok=True)

    def write_features(self, df: pd.DataFrame, feature_set_name: str, online: bool = True):
        """
        Writes dataframe to feature store.
        
        Args:
            df: Feature DataFrame.
            feature_set_name: Name of the feature set (e.g., 'fraud_features').
            online: Whether to ALSO write to online store (latest values).
        """
        try:
            # 1. Offline Store - Append with timestamp/version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{feature_set_name}_{timestamp}.parquet"
            save_path = self.offline_path / filename
            
            df.to_parquet(save_path, index=False)
            logger.info(f"Saved {len(df)} rows to Offline Store: {save_path}")
            
            # 2. Online Store - Overwrite with latest state (Simulated Key-Value)
            # In production, this would write to Redis. Here we save a single 'latest' parquet/pkl for fast loading.
            if online:
                online_file = self.online_path / f"{feature_set_name}_latest.parquet"
                df.to_parquet(online_file, index=False)
                logger.info(f"Updated Online Store: {online_file}")
                
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            raise e

    def get_online_features(self, feature_set_name: str) -> pd.DataFrame:
        """
        Retrieves the latest features from the Online Store.
        Simulates low-latency lookup.
        """
        try:
            online_file = self.online_path / f"{feature_set_name}_latest.parquet"
            if not online_file.exists():
                raise FileNotFoundError(f"Feature set {feature_set_name} not found in Online Store.")
            
            df = pd.read_parquet(online_file)
            return df
        except Exception as e:
            logger.error(f"Failed to read online features: {e}")
            raise e
