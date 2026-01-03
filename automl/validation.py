import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, target: str = "churn") -> None:
    if df.empty:
        raise ValueError("Data validation failed: The dataset is empty.")
    if target not in df.columns:
        raise ValueError(f"Data validation failed: Target column '{target}' not found in dataset. Available columns: {list(df.columns)}")
    if df[target].isnull().any():
        logger.warning(f"Warning: Target column '{target}' contains missing values. These rows will likely be dropped.")
    logger.info("Data validation passed.")