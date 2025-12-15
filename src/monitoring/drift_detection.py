import pandas as pd
import json
import logging
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects data drift using Evidently AI.
    """
    def __init__(self, output_path: str = None):
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = Path(__file__).resolve().parents[2] / "reports" / "drift"
        
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_drift_report(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, report_name: str = "fraud_drift"):
        """
        Generates a data drift report comparing reference (train) and current (production) data.
        
        Args:
            reference_data: DataFrame used for training.
            current_data: DataFrame from production/recent batch.
            report_name: Name for the report file.
        """
        try:
            logger.info("Generating drift report...")
            report = Report(metrics=[
                DataDriftPreset(), 
            ])
            
            report.run(reference_data=reference_data, current_data=current_data)
            
            # Save as HTML
            html_path = self.output_path / f"{report_name}.html"
            report.save_html(str(html_path))
            
            # Save as JSON for programmatic checks
            json_path = self.output_path / f"{report_name}.json"
            report.save_json(str(json_path))
            
            logger.info(f"Drift report saved to {self.output_path}")
            
            # Check for drift (simple logic based on JSON output could be added here)
            return str(html_path)
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise e
