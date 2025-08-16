"""
Optimized ML pipeline with parallel processing and advanced caching
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple, Optional, Any
import joblib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import multiprocessing

from performance_utils import timer, PerformanceTimer
from data_cache import cached, model_cache
from vectorized_features import optimize_dataframe_memory

logger = logging.getLogger(__name__)

class OptimizedMLPipeline:
    """High-performance ML pipeline with parallel processing"""
    
    def __init__(self, n_jobs: int = -1, use_gpu: bool = False):
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.use_gpu = use_gpu
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        self.best_params = None
        self.feature_importance = None
        
        logger.info(f"Initialized ML pipeline with {self.n_jobs} workers")
    
    @timer("Parallel feature selection")
    def select_features_parallel(self, X: pd.DataFrame, y: pd.Series, 
                                max_features: int = 50) -> List[str]:
        """
        Parallel feature selection using multiple algorithms
        """
        from sklearn.feature_selection import mutual_info_regression, f_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Parallel feature importance calculation
        def calculate_rf_importance(data):
            X_chunk, y_chunk = data
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X_chunk, y_chunk)
            return rf.feature_importances_
        
        def calculate_mutual_info(data):
            X_chunk, y_chunk = data
            return mutual_info_regression(X_chunk, y_chunk, random_state=42)
        
        def calculate_f_score(data):
            X_chunk, y_chunk = data
            f_scores, _ = f_regression(X_chunk, y_chunk)
            return f_scores
        
        # Prepare data
        data = (X, y)
        
        # Run feature selection methods in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'rf_importance': executor.submit(calculate_rf_importance, data),
                'mutual_info': executor.submit(calculate_mutual_info, data),
                'f_score': executor.submit(calculate_f_score, data)
            }
            
            results = {}
            for method, future in futures.items():
                try:
                    results[method] = future.result()
                except Exception as e:
                    logger.warning(f"Feature selection method {method} failed: {e}")
        
        # Combine feature scores
        feature_scores = pd.DataFrame(index=X.columns)
        
        if 'rf_importance' in results:
            feature_scores['rf_importance'] = results['rf_importance']
        
        if 'mutual_info' in results:
            feature_scores['mutual_info'] = results['mutual_info']
        
        if 'f_score' in results:
            feature_scores['f_score'] = results['f_score']
        
        # Normalize scores and create composite score
        for col in feature_scores.columns:
            feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                                 (feature_scores[col].max() - feature_scores[col].min())
        
        feature_scores['composite_score'] = feature_scores.mean(axis=1)
        
        # Select top features
        selected_features = feature_scores.nlargest(max_features, 'composite_score').index.tolist()
        
        logger.info(f"Selected {len(selected_features)} features using parallel selection")
        return selected_features
    
    @timer("Parallel hyperparameter optimization")
    def optimize_hyperparameters_parallel(self, X: pd.DataFrame, y: pd.Series, 
                                         n_trials: int = 100) -> Dict:
        """
        Parallel hyperparameter optimization with Optuna
        """
        def objective(trial):
            # XGBoost parameters with GPU support
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': 1,  # Optuna handles parallelization
            }
            
            if self.use_gpu:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            
            # Cross-validation with parallel processing
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=3, scoring='neg_mean_absolute_error',
                n_jobs=1  # Let Optuna handle the parallelization
            )
            return -scores.mean()
        
        # Create study with parallel processing
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Use parallel optimization
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=min(self.n_jobs, 4),  # Don't overwhelm the system
            show_progress_bar=True
        )
        
        logger.info(f"Best MAE found: {study.best_value:.3f}")
        return study.best_params
    
    @cached("ml_training", ttl=7200)  # Cache for 2 hours
    def train_model_optimized(self, X: pd.DataFrame, y: pd.Series, 
                            optimize_hyperparameters: bool = True,
                            use_feature_selection: bool = True) -> Dict[str, Any]:
        """
        Optimized model training with caching and parallel processing
        """
        with PerformanceTimer("Complete ML training pipeline"):
            
            # Memory optimization
            X = optimize_dataframe_memory(X)
            
            # Feature selection
            if use_feature_selection and len(X.columns) > 50:
                selected_features = self.select_features_parallel(X, y)
                X = X[selected_features]
                self.features = selected_features
            else:
                self.features = list(X.columns)
            
            logger.info(f"Training with {len(self.features)} features on {len(X)} samples")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Feature scaling
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Hyperparameter optimization
            if optimize_hyperparameters:
                self.best_params = self.optimize_hyperparameters_parallel(X_train_scaled, y_train)
            else:
                self.best_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': self.n_jobs
                }
                
                if self.use_gpu:
                    self.best_params['tree_method'] = 'gpu_hist'
                    self.best_params['gpu_id'] = 0
            
            # Train final model
            self.model = xgb.XGBRegressor(**self.best_params)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            y_pred = self.model.predict(X_test_scaled)
            
            performance_metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(self.features)
            }
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Cache the complete model pipeline
            model_id = model_cache.save_model_pipeline(
                self.model, self.scaler, self.features,
                self.best_params, performance_metrics
            )
            
            logger.info(f"Model training complete. MAE: {performance_metrics['mae']:.3f}")
            
            return {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'performance': performance_metrics,
                'model_id': model_id
            }
    
    def load_cached_model(self) -> Optional[Dict]:
        """Load the most recent cached model"""
        return model_cache.load_latest_model()
    
    @timer("Batch prediction")
    def predict_batch(self, X: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
        """
        Memory-efficient batch prediction for large datasets
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model_optimized first.")
        
        if len(X) <= batch_size:
            X_scaled = self.scaler.transform(X[self.features])
            return self.model.predict(X_scaled)
        
        # Process in batches
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i:i + batch_size]
            X_batch_scaled = self.scaler.transform(batch[self.features])
            batch_pred = self.model.predict(X_batch_scaled)
            predictions.extend(batch_pred)
        
        return np.array(predictions)

class ParallelDataProcessor:
    """Parallel data processing utilities"""
    
    @staticmethod
    @timer("Parallel data processing")
    def process_years_parallel(years: List[int], process_func: callable, 
                             max_workers: int = None) -> List[Any]:
        """
        Process multiple years of data in parallel
        """
        max_workers = max_workers or min(len(years), multiprocessing.cpu_count())
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, year): year for year in years}
            results = []
            
            for future in futures:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    year = futures[future]
                    logger.error(f"Error processing year {year}: {e}")
        
        return results
    
    @staticmethod
    @timer("Parallel feature engineering")
    def engineer_features_parallel(data_chunks: List[pd.DataFrame], 
                                 feature_func: callable,
                                 max_workers: int = None) -> pd.DataFrame:
        """
        Apply feature engineering to data chunks in parallel
        """
        max_workers = max_workers or min(len(data_chunks), multiprocessing.cpu_count())
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(feature_func, chunk) for chunk in data_chunks]
            processed_chunks = []
            
            for future in futures:
                try:
                    result = future.result()
                    processed_chunks.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel feature engineering: {e}")
        
        if processed_chunks:
            return pd.concat(processed_chunks, ignore_index=True)
        else:
            return pd.DataFrame()

def create_optimized_pipeline(use_gpu: bool = False, n_jobs: int = -1) -> OptimizedMLPipeline:
    """
    Factory function to create optimized ML pipeline
    """
    pipeline = OptimizedMLPipeline(n_jobs=n_jobs, use_gpu=use_gpu)
    
    # Check for GPU availability
    if use_gpu:
        try:
            import cupy
            logger.info("GPU acceleration enabled")
        except ImportError:
            logger.warning("GPU requested but CuPy not available, falling back to CPU")
            pipeline.use_gpu = False
    
    return pipeline

# Utility functions for memory optimization
@timer("Memory cleanup")
def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    import gc
    gc.collect()
    
@timer("Data chunking")
def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """
    Split DataFrame into memory-efficient chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    
    logger.info(f"Split data into {len(chunks)} chunks of ~{chunk_size} rows each")
    return chunks