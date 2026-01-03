import argparse
import yaml
import mlflow
import logging
import pandas as pd
from automl.ingestion import load_data
from automl.validation import validate_data
from automl.tuning import run_tuning
from mlflow.tracking import MlflowClient


4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(data_path: str, config_path: str):
    logger.info("Starting AutoML Pipeline...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        target = config.get('target_column', 'churn')
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(config.get("experiment_name", "Default_Experiment"))
    try:
        df = load_data(data_path)
        validate_data(df, target) 
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return
    X = df.drop(columns=[target, 'customer_id']) # Drop ID if exists
    y = df[target]

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Features: {len(num_cols)} Numerical, {len(cat_cols)} Categorical")

    results_df = run_tuning(config, X, y, num_cols, cat_cols)

    if results_df.empty:
        logger.warning("No models were trained.")
        return

    best_run = results_df.iloc[0]
    logger.info(f"Winner: {best_run['model'].upper()} (Score: {best_run['score']:.4f})")
    
    model_uri = f"runs:/{best_run['run_id']}/model"
    mlflow.register_model(model_uri, "Production_Model")
    logger.info("Best model registered to 'Production_Model'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Pipeline CLI")
    parser.add_argument("--data", type=str, default="data/raw/Churn_Modelling.csv", help="Path to dataset CSV")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config YAML")
    
    args = parser.parse_args()
    main(args.data, args.config)