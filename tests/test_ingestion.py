import os
from src.data_ingestion.ingest import DataIngester

def test_data_ingestion_creates_local_file():
    ingester = DataIngester(config_path="config/config.yaml")
    local_path = ingester.run()

    # Assertions
    assert os.path.exists(local_path)
    assert os.path.getsize(local_path) > 0
    assert "train.csv" in local_path