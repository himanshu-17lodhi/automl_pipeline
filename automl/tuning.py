import optuna
import mlflow
import sklearn.model_selection
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, List
from .preprocessing import build_pipeline

logger = logging.getLogger(__name__)

def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    if model_name == 'random_forest':
        return RandomForestClassifier(**params, random_state=42)
    elif model_name == 'xgboost':
        return XGBClassifier(**params, random_state=42, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def objective(trial: optuna.Trial, model_name: str, config: Dict, 
              X_train: pd.DataFrame, y_train: pd.Series, 
              num_cols: List[str], cat_cols: List[str]) -> float:
    params = {}
    grid = config['models'][model_name]['param_grid']
    
    for key, values in grid.items():
        if isinstance(values[0], float):
            params[key] = trial.suggest_float(key, min(values), max(values))
        elif isinstance(values[0], int):
            params[key] = trial.suggest_int(key, min(values), max(values))
        else:
            params[key] = trial.suggest_categorical(key, values)

    model = get_model(model_name, params)
    pipeline = build_pipeline(model, num_cols, cat_cols)

    scores = sklearn.model_selection.cross_val_score(
        pipeline, X_train, y_train, cv=3, scoring='f1_macro'
    )
    return scores.mean()

def run_tuning(config: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
               num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    results = []

    for model_name, settings in config['models'].items():
        if not settings.get('active', False):
            continue
            
        logger.info(f"Optimizing {model_name}...")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, model_name, config, X_train, y_train, num_cols, cat_cols),
            n_trials=10,
            timeout=config.get('timeout_seconds', 600)
        )

        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best {model_name} Score: {best_score:.4f}")

        with mlflow.start_run(run_name=f"{model_name}_optimized"):
            mlflow.log_params(best_params)
            mlflow.log_metric("f1_macro", best_score)
            
            final_model = get_model(model_name, best_params)
            final_pipeline = build_pipeline(final_model, num_cols, cat_cols)
            final_pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
            
            results.append({
                "model": model_name,
                "score": best_score,
                "run_id": mlflow.active_run().info.run_id
            })

    return pd.DataFrame(results).sort_values(by="score", ascending=False)