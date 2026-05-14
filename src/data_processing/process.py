import os
import yaml
import logging
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config_path: str):
        load_dotenv()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.bucket_name = self.config["gcp"]["bucket_name"]
        self.raw_data_path = Path("data/raw/train.csv")
        self.processed_data_path = self.config["data"]["processed_data_path"]
        
        self.local_processed_dir = Path("data/processed")
        self.local_model_dir = Path("models")
        
        self.preprocessor = None
        self.client = None

    def _load_raw_data(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Raw data shape: {df.shape}")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns {cols_to_drop}. Remaining: {df.columns.tolist()}")
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        logger.info("Feature engineering complete: FamilySize and IsAlone added.")
        return df 

    def _separate_features_target(self, df: pd.DataFrame):
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        return X, y

    def _build_preprocessor(self):
        numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Pclass']
        categorical_features = ['Sex', 'Embarked']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return preprocessor

    def _save_locally(self, X_train, X_test, y_train, y_test):
        self.local_processed_dir.mkdir(parents=True, exist_ok=True)
        self.local_model_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(self.local_processed_dir / "X_train.csv", index=False)
        X_test.to_csv(self.local_processed_dir / "X_test.csv", index=False)
        y_train.to_csv(self.local_processed_dir / "y_train.csv", index=False)
        y_test.to_csv(self.local_processed_dir / "y_test.csv", index=False)

        if self.preprocessor:
            joblib.dump(self.preprocessor, self.local_model_dir / "preprocessor.joblib")
            logger.info("Saved all datasets and preprocessor.joblib locally.")

    def _upload_to_gcs(self):
        try:
            self.client = storage.Client()
            bucket = self.client.bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            return

        for file_name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            local_path = self.local_processed_dir / file_name
            blob = bucket.blob(f"{self.processed_data_path}/{file_name}")
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {file_name} to GCS.")

        model_blob = bucket.blob("models/preprocessor.joblib")
        model_blob.upload_from_filename(self.local_model_dir / "preprocessor.joblib")
        logger.info("Uploaded preprocessor.joblib to GCS.")

    def run(self):
        logger.info("--- Starting Data Processing Component ---")
        
        df = self._load_raw_data()
        df = self._drop_columns(df)
        df = self._feature_engineering(df)
        X, y = self._separate_features_target(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.preprocessor = self._build_preprocessor()
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Pclass']
        cat_encoder = self.preprocessor.named_transformers_['cat']['onehot']
        cat_cols = cat_encoder.get_feature_names_out(['Sex', 'Embarked'])
        all_cols = num_cols + list(cat_cols)
        
        X_train_final = pd.DataFrame(X_train_transformed, columns=all_cols)
        X_test_final = pd.DataFrame(X_test_transformed, columns=all_cols)
        
        self._save_locally(X_train_final, X_test_final, y_train, y_test)
        self._upload_to_gcs()
        
        logger.info("--- Processing Component Finished Successfully ---")

if __name__ == "__main__":
    processor = DataProcessor(config_path="config/config.yaml")
    processor.run()