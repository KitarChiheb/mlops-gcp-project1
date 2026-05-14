import os
import pandas as pd
import pytest
from src.data_processing.process import DataProcessor

@pytest.fixture(scope="module")
def processed_data():
    """
    Fixture to run the processor once and provide paths to the outputs.
    This prevents running the whole pipeline multiple times for each test.
    """
    config_path = "config/config.yaml"
    processor = DataProcessor(config_path=config_path)
    processor.run()
    
    return {
        "X_train": "data/processed/X_train.csv",
        "X_test": "data/processed/X_test.csv",
        "y_train": "data/processed/y_train.csv",
        "y_test": "data/processed/y_test.csv",
        "preprocessor": "models/preprocessor.joblib"
    }

def test_processing_output_files_exist(processed_data):
    """All 4 processed CSVs and preprocessor.joblib must exist after run."""
    for file_path in processed_data.values():
        assert os.path.exists(file_path), f"File {file_path} was not created!"

def test_no_missing_values_after_processing(processed_data):
    """Processed X_train must have zero missing values (Imputation check)."""
    df = pd.read_csv(processed_data["X_train"])
    missing_count = df.isnull().sum().sum()
    assert missing_count == 0, f"Found {missing_count} missing values in X_train!"

def test_train_test_split_ratio(processed_data):
    """X_train should be ~80% of total, X_test ~20%."""
    train_len = len(pd.read_csv(processed_data["X_train"]))
    test_len = len(pd.read_csv(processed_data["X_test"]))
    total = train_len + test_len
    
    test_ratio = test_len / total
    # Check if ratio is roughly 0.2 (allowing for minor rounding)
    assert 0.19 <= test_ratio <= 0.21, f"Test ratio is {test_ratio}, expected ~0.2"

def test_feature_engineering_columns_exist(processed_data):
    """FamilySize and IsAlone must be present in output."""
    df = pd.read_csv(processed_data["X_train"])
    assert 'FamilySize' in df.columns, "FamilySize column missing!"
    assert 'IsAlone' in df.columns, "IsAlone column missing!"