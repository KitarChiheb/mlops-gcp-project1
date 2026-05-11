import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
import google.cloud.exceptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataIngester:
    def __init__(self, config_path: str):
        load_dotenv()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.bucket_name = self.config["gcp"]["bucket_name"]
        self.raw_data_path = self.config["data"]["raw_data_path"]
        
        self.client = None

    def _create_local_dirs(self):
        """Creates the data/raw directory structure."""
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        logger.info("Local directories ensured.")

    def connect(self):
        """Initializes the GCS Client."""
        try:
            self.client = storage.Client()
            logger.info(f"Successfully connected to GCS. Project: {self.client.project}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise

    def download_raw_data(self) -> str:
        """Downloads the blob from GCS to local disk."""
        if self.client is None:
            logger.error("GCS client not initialized. Call connect() first.")
            raise RuntimeError("GCS client is None. Authentication required.")
        filename = os.path.basename(self.raw_data_path)
        local_path = os.path.join("data/raw", filename)
        
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(self.raw_data_path)
        
        logger.info(f"Attempting to download {self.raw_data_path} from bucket {self.bucket_name}...")
        
        try:
            blob.download_to_filename(local_path)
            logger.info(f"Download complete! File saved to: {local_path}")
        except google.cloud.exceptions.NotFound:
            logger.error(f"Error: The file {self.raw_data_path} was not found in bucket {self.bucket_name}.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
            
        return local_path

    def run(self) -> str:
        """Orchestrator for the ingestion process."""
        logger.info("--- Starting Data Ingestion Component ---")
        self._create_local_dirs()
        self.connect()
        local_file_path = self.download_raw_data()
        logger.info("--- Ingestion Component Finished Successfully ---")
        return local_file_path

if __name__ == "__main__":
    ingester = DataIngester(config_path="config/config.yaml")
    ingester.run()