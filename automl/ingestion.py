import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    column_mapping = {
        "CreditScore": "credit_score",
        "Geography": "country",
        "Gender": "gender",
        "Age": "age",
        "Tenure": "tenure",
        "Balance": "account_balance",  
        "NumOfProducts": "num_products",
        "HasCrCard": "has_credit_card",
        "IsActiveMember": "is_active_member",
        "EstimatedSalary": "salary",       
        "Exited": "churn",
        "RowNumber": "drop",             
        "CustomerId": "customer_id",
        "Surname": "drop"                   
    }   
    df = df.rename(columns=column_mapping) # rename columns to match API schema

    drop_cols = [c for c in df.columns if c == "drop"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.columns = [c.lower() for c in df.columns]

    logger.info(f"Loaded {len(df)} rows. Columns standardized.")
    return df