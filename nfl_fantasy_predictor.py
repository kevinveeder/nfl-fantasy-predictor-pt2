"""
NFL Fantasy Football Draft Predictor
====================================
Scrapes NFL data, trains an XGBoost model, and generates draft recommendations.
Upgraded from basic linear regression to something actually useful.

NEW: Now includes season-ending injury history analysis!
- Tracks major injuries (ACL, Achilles, season-ending surgeries, etc.)
- Creates injury risk scores based on frequency and severity
- Applies injury multipliers to projections (0.75x - 1.0x range)
- Integrates with existing QB-WR chemistry and QB support features

Author: Kevin Veeder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import requests
from bs4 import BeautifulSoup
import lxml
import time
import warnings
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import pickle
import os
from pathlib import Path
# Remove nfl_data_py dependency - use PFR scraping instead
NFL_DATA_AVAILABLE = True  # We'll implement our own injury scraping
warnings.filterwarnings('ignore')

class NFLFantasyPredictor:
    def __init__(self):
        self.model = None # XGBoost model - way better than linear regression
        self.scaler = StandardScaler() # gotta normalize the features
        self.features = [] # starts empty, gets populated with all the good stuff
        self.historical_data = None 
        self.projections_data = {} # holds current year projections by position
        self.feature_importance = None # helps us see what actually matters
        self.best_params = None # hyperparams found by optuna
        self.qb_wr_chemistry_data = {} # stores QB-WR chemistry scores and historical data
        self.play_by_play_data = {} # stores detailed game-by-game connection data
        self.qb_multiplier_data = {} # stores QB performance multipliers for RB support and O-line
        self.team_support_data = {} # stores team-level RB and O-line data by year
        self.injury_data = {} # stores historical injury data for players
        self.player_injury_history = {} # stores processed injury features by player
        
        # Cache to prevent redundant calculations
        self._enhanced_projections_cache = None
        self._cache_timestamp = None
        
        # Model caching setup
        self.cache_dir = Path("model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.model_cache_file = self.cache_dir / "trained_model.pkl"
        self.data_hash_file = self.cache_dir / "data_hash.txt"

    def validate_training_data(self, df: pd.DataFrame) -> bool:
        """
        Comprehensive data validation to prevent training on corrupted data.
        This ensures data quality and prevents common machine learning pitfalls.
        """
        print("Running data validation checks...")
        
        # Check for required core columns that every dataset should have
        required_columns = ['Player', 'FantPt', 'G', 'Year']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing critical columns: {missing_columns}")
        
        # Validate data types and ranges
        if not pd.api.types.is_numeric_dtype(df['FantPt']):
            raise ValueError("Fantasy points column must be numeric")
            
        if not pd.api.types.is_numeric_dtype(df['G']):
            raise ValueError("Games column must be numeric")
        
        # Check for excessive missing data in target variable
        missing_fantasy_points = df['FantPt'].isna().sum()
        if missing_fantasy_points > 0.1 * len(df):
            raise ValueError(f"Too many missing fantasy points: {missing_fantasy_points}/{len(df)} ({missing_fantasy_points/len(df)*100:.1f}%)")
        
        # Check for unrealistic fantasy point values
        # Most players score between 0-30 points per game, anything above 50 is suspicious
        max_reasonable_fppg = 50
        outliers = df[df['FantPt'] > max_reasonable_fppg * df['G']]
        if len(outliers) > 0:
            print(f"Warning: Found {len(outliers)} players with unusually high fantasy points")
            print("Players with potentially corrupted data:")
            for _, player in outliers.head().iterrows():
                print(f"  {player.get('Player', 'Unknown')}: {player['FantPt']:.1f} points in {player['G']} games")
        
        # Check for negative values where they shouldn't exist
        non_negative_columns = ['G', 'FantPt', 'Att', 'Tgt', 'Rec', 'Yds', 'TD']
        for col in non_negative_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"Warning: Found {negative_count} negative values in {col} column")
        
        # Check for temporal data integrity - ensure no future data leakage
        current_year = 2024  # Update this as needed
        future_years = df[df['Year'] > current_year]
        if len(future_years) > 0:
            raise ValueError(f"Found data from future years: {sorted(future_years['Year'].unique())}")
        
        # Validate minimum games played filter
        min_games = 4
        players_few_games = df[df['G'] < min_games]
        if len(players_few_games) > 0.3 * len(df):
            print(f"Warning: {len(players_few_games)} players have fewer than {min_games} games")
        
        # Check for duplicate entries (same player, same year)
        if 'Player' in df.columns and 'Year' in df.columns:
            duplicates = df.duplicated(subset=['Player', 'Year'], keep=False)
            if duplicates.sum() > 0:
                print(f"Warning: Found {duplicates.sum()} duplicate player-year entries")
                duplicate_players = df[duplicates][['Player', 'Year']].drop_duplicates()
                print("Duplicate entries found for:")
                for _, row in duplicate_players.head().iterrows():
                    print(f"  {row['Player']} - {row['Year']}")
        
        # Validate feature engineering didn't create invalid values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        inf_columns = []
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_columns.append((col, inf_count))
        
        if inf_columns:
            print("Warning: Found infinite values in columns:")
            for col, count in inf_columns:
                print(f"  {col}: {count} infinite values")
        
        print(f"Data validation complete. Dataset has {len(df)} players across {df['Year'].nunique()} years")
        return True

    def validate_feature_engineering(self, df: pd.DataFrame) -> bool:
        """
        Validate that feature engineering didn't introduce data leakage or corruption.
        This checks that derived features make logical sense.
        """
        print("Validating feature engineering...")
        
        # Check per-game calculations make sense
        if all(col in df.columns for col in ['Total_Yards', 'G', 'Total_Yards_Per_Game']):
            # Verify per-game calculations are correct
            calculated_ypg = df['Total_Yards'] / df['G']
            diff = abs(calculated_ypg - df['Total_Yards_Per_Game'])
            large_diffs = diff > 0.1  # Allow small floating point differences
            if large_diffs.sum() > 0:
                print(f"Warning: {large_diffs.sum()} players have incorrect yards per game calculations")
        
        # Check that efficiency ratios are within reasonable bounds
        if 'Catch_Rate' in df.columns:
            impossible_catch_rates = (df['Catch_Rate'] > 1.0) | (df['Catch_Rate'] < 0)
            if impossible_catch_rates.sum() > 0:
                print(f"Warning: {impossible_catch_rates.sum()} players have impossible catch rates")
        
        # Check yards per carry makes sense (should be between 0-15 typically)
        if 'Yards_Per_Carry' in df.columns:
            unrealistic_ypc = df['Yards_Per_Carry'] > 20
            if unrealistic_ypc.sum() > 0:
                print(f"Warning: {unrealistic_ypc.sum()} players have unrealistic yards per carry")
        
        # Validate injury features if present
        injury_features = [col for col in df.columns if 'injury' in col.lower()]
        if injury_features:
            for feature in injury_features:
                if 'risk' in feature.lower() or 'rate' in feature.lower():
                    # Risk scores should be between 0 and 1
                    invalid_risks = (df[feature] < 0) | (df[feature] > 1)
                    if invalid_risks.sum() > 0:
                        print(f"Warning: {invalid_risks.sum()} players have invalid {feature} values")
        
        print("Feature engineering validation complete")
        return True

    def detect_potential_data_leakage(self, df: pd.DataFrame) -> None:
        """
        Detect potential sources of data leakage that could inflate model performance.
        Data leakage occurs when future information accidentally gets included in training.
        """
        print("Checking for potential data leakage...")
        
        # Check if any features are too highly correlated with target (potential leakage)
        if 'FPPG' in df.columns:
            numeric_features = df.select_dtypes(include=[np.number]).columns
            correlations = df[numeric_features].corr()['FPPG'].abs().sort_values(ascending=False)
            
            # Perfect or near-perfect correlations might indicate leakage
            suspicious_correlations = correlations[correlations > 0.95]
            if len(suspicious_correlations) > 1:  # Exclude target itself
                print("Warning: Found features with suspiciously high correlation to target:")
                for feature, corr in suspicious_correlations.items():
                    if feature != 'FPPG':
                        print(f"  {feature}: {corr:.3f}")
        
        # Check for features that shouldn't exist in historical data
        future_looking_keywords = ['projection', 'predicted', 'forecast', 'expected']
        suspicious_features = []
        for col in df.columns:
            col_lower = col.lower()
            for keyword in future_looking_keywords:
                if keyword in col_lower:
                    suspicious_features.append(col)
                    break
        
        if suspicious_features:
            print(f"Warning: Found potentially future-looking features: {suspicious_features}")
        
        print("Data leakage check complete")

    def _get_data_hash(self) -> str:
        """
        Generate a hash of the training data to detect if model needs retraining.
        """
        if self.historical_data is None:
            return ""
        
        # Create hash from key data characteristics
        data_string = f"{len(self.historical_data)}_{self.historical_data.columns.tolist()}_{self.historical_data['Year'].max()}_{self.historical_data['FantPt'].sum()}"
        import hashlib
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _save_trained_model(self) -> None:
        """
        Save the trained model, scaler, features, and metadata to cache.
        """
        if self.model is None:
            print("No trained model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params,
            'training_timestamp': time.time(),
            'data_hash': self._get_data_hash()
        }
        
        try:
            with open(self.model_cache_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save data hash separately for quick checking
            with open(self.data_hash_file, 'w') as f:
                f.write(self._get_data_hash())
            
            print(f"Model cached successfully to {self.model_cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache model: {e}")
    
    def _load_cached_model(self) -> bool:
        """
        Load cached model if it exists and data hasn't changed.
        Returns True if model was loaded successfully, False otherwise.
        """
        if not self.model_cache_file.exists():
            print("No cached model found - will train new model")
            return False
        
        # Check if data has changed
        current_hash = self._get_data_hash()
        if self.data_hash_file.exists():
            try:
                with open(self.data_hash_file, 'r') as f:
                    cached_hash = f.read().strip()
                if cached_hash != current_hash:
                    print("Training data has changed - will retrain model")
                    return False
            except Exception as e:
                print(f"Warning: Could not read data hash: {e}")
                return False
        
        # Try to load the cached model
        try:
            with open(self.model_cache_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.feature_importance = model_data.get('feature_importance')
            self.best_params = model_data.get('best_params')
            
            training_time = model_data.get('training_timestamp', 0)
            hours_ago = (time.time() - training_time) / 3600
            
            print(f"+ Loaded cached model (trained {hours_ago:.1f} hours ago)")
            print(f"+ Model uses {len(self.features)} features")
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load cached model: {e}")
            print("Will train new model instead")
            return False
    
    def _should_retrain_model(self) -> bool:
        """
        Determine if model should be retrained based on cache age and data changes.
        """
        if not self.model_cache_file.exists():
            return True
        
        try:
            # Check cache age
            cache_age_hours = (time.time() - self.model_cache_file.stat().st_mtime) / 3600
            if cache_age_hours > 24:  # Retrain if cache is older than 24 hours
                print(f"Model cache is {cache_age_hours:.1f} hours old - will retrain")
                return True
            
            # Check if data has changed
            current_hash = self._get_data_hash()
            if self.data_hash_file.exists():
                with open(self.data_hash_file, 'r') as f:
                    cached_hash = f.read().strip()
                if cached_hash != current_hash:
                    print("Training data has changed - will retrain")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Warning: Error checking cache status: {e}")
            return True

    def get_enhanced_projections(self, force_refresh: bool = False, 
                               use_chemistry: bool = True, 
                               use_qb_multipliers: bool = True, 
                               use_injury_history: bool = True) -> Optional[pd.DataFrame]:
        """
        Get projections with all enhancements applied in a single optimized pass.
        Uses caching to prevent redundant calculations and web scraping.
        Includes comprehensive type checking and validation.
        """
        # Check if we can use cached results
        if not force_refresh and self._enhanced_projections_cache is not None:
            cache_age = time.time() - (self._cache_timestamp or 0)
            if cache_age < 300:  # Cache valid for 5 minutes
                print("Using cached enhanced projections (avoiding redundant calculations)")
                return self._enhanced_projections_cache.copy()
        
        print("Generating enhanced projections with all adjustments...")
        
        # Step 1: Get base projections (only scrape once)
        print("  -> Scraping current projections...")
        base_projections = self.scrape_all_positions()
        if base_projections is None:
            print("  X Failed to scrape projections")
            return None
        
        # Validate base projections structure
        required_columns = ['Player', 'Position']
        missing_cols = set(required_columns) - set(base_projections.columns)
        if missing_cols:
            print(f"  X Missing required columns: {missing_cols}")
            return None
        
        print(f"  + Scraped {len(base_projections)} players")
        
        # Step 2: Apply all enhancements in a single pass per position
        enhanced_positions = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = base_projections[base_projections['Position'] == position].copy()
            
            if len(pos_players) == 0:
                print(f"  ! No {position} players found")
                continue
            
            print(f"  -> Processing {len(pos_players)} {position} players...")
            
            # Type checking: Ensure we have numeric fantasy points
            # Look for fantasy points columns - prioritize MISC FPTS which is the main column
            fpts_columns = []
            if 'MISC FPTS' in pos_players.columns:
                fpts_columns.append('MISC FPTS')
            else:
                # Fallback to other fantasy point columns
                for col in pos_players.columns:
                    col_upper = col.upper()
                    if any(keyword in col_upper for keyword in ['FPTS', 'FANTASY', 'POINTS']):
                        fpts_columns.append(col)
            
            if not fpts_columns:
                print(f"  X No fantasy points column found for {position}")
                continue
                
            primary_fpts_col = fpts_columns[0]
            pos_players[primary_fpts_col] = pd.to_numeric(pos_players[primary_fpts_col], errors='coerce')
            
            # Remove players with invalid fantasy points
            before_count = len(pos_players)
            pos_players = pos_players.dropna(subset=[primary_fpts_col])
            after_count = len(pos_players)
            if before_count != after_count:
                print(f"  ! Removed {before_count - after_count} {position} players with invalid fantasy points")
            
            # Apply position-specific enhancements
            adjustments_applied = []
            
            # Chemistry adjustments for WR/TE
            if use_chemistry and position in ['WR', 'TE'] and self.qb_wr_chemistry_data:
                try:
                    pos_players = self._apply_chemistry_adjustments(pos_players)
                    adjustments_applied.append("QB-WR chemistry")
                except Exception as e:
                    print(f"  ! Chemistry adjustment failed for {position}: {e}")
            
            # QB support adjustments
            if use_qb_multipliers and position == 'QB' and self.qb_multiplier_data:
                try:
                    pos_players = self._apply_qb_support_adjustments(pos_players)
                    adjustments_applied.append("QB support multipliers")
                except Exception as e:
                    print(f"  ! QB support adjustment failed: {e}")
            
            # Injury risk adjustments for all positions
            if use_injury_history and self.player_injury_history:
                try:
                    pos_players = self._apply_injury_risk_adjustments(pos_players)
                    adjustments_applied.append("injury risk")
                except Exception as e:
                    print(f"  ! Injury adjustment failed for {position}: {e}")
            
            # Validate final results
            if len(pos_players) == 0:
                print(f"  X No valid {position} players after processing")
                continue
                
            # Determine best fantasy points column after adjustments
            final_fpts_col = None
            for col_priority in ['Injury_Adjusted_FPTS', 'Support_Adjusted_FPTS', 'Chemistry_Adjusted_FPTS', 'FPTS', 'Fantasy Points']:
                if col_priority in pos_players.columns:
                    final_fpts_col = col_priority
                    break
            
            if final_fpts_col is None:
                print(f"  X No valid fantasy points column found for {position}")
                continue
            
            # Final type check and sorting
            pos_players[final_fpts_col] = pd.to_numeric(pos_players[final_fpts_col], errors='coerce')
            pos_players = pos_players.dropna(subset=[final_fpts_col])
            pos_players = pos_players.sort_values(final_fpts_col, ascending=False)
            
            # Add position ranking
            pos_players[f'{position}_Rank'] = range(1, len(pos_players) + 1)
            
            enhanced_positions.append(pos_players)
            
            # Report results
            if adjustments_applied:
                adjustments_str = ", ".join(adjustments_applied)
                print(f"  + {position}: {len(pos_players)} players, applied {adjustments_str}")
            else:
                print(f"  + {position}: {len(pos_players)} players, no adjustments")
        
        # Step 3: Combine all positions and cache results
        if not enhanced_positions:
            print("  X No valid position data after enhancement")
            return None
        
        enhanced_projections = pd.concat(enhanced_positions, ignore_index=True)
        
        # Final validation
        total_players = len(enhanced_projections)
        positions_count = enhanced_projections['Position'].value_counts()
        print(f"  + Enhanced projections complete: {total_players} players")
        print(f"    Breakdown: {dict(positions_count)}")
        
        # Cache the results
        self._enhanced_projections_cache = enhanced_projections.copy()
        self._cache_timestamp = time.time()
        
        return enhanced_projections

    def create_temporal_split(self, df: pd.DataFrame, test_seasons: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/test split to prevent data leakage.
        Train on older seasons, test on more recent seasons.
        This ensures we don't accidentally use future information to predict the past.
        """
        print(f"Creating temporal split with {test_seasons} test season(s)...")
        
        # Ensure we have the Year column
        if 'Year' not in df.columns:
            raise ValueError("Year column is required for temporal splitting")
        
        # Get available years and sort them
        available_years = sorted(df['Year'].unique())
        print(f"Available years: {available_years}")
        
        # Calculate split point
        split_year = available_years[-test_seasons]
        train_data = df[df['Year'] < split_year]
        test_data = df[df['Year'] >= split_year]
        
        print(f"Training on years: {sorted(train_data['Year'].unique())}")
        print(f"Testing on years: {sorted(test_data['Year'].unique())}")
        print(f"Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Temporal split resulted in empty train or test set")
        
        return train_data, test_data

    def time_series_cross_validation(self, df: pd.DataFrame, n_splits: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Implement time series cross-validation (walk-forward validation).
        This creates multiple train/test splits moving forward through time.
        Each split trains on all previous years and tests on the next year.
        """
        print(f"Creating {n_splits} time series cross-validation splits...")
        
        available_years = sorted(df['Year'].unique())
        if len(available_years) < n_splits + 1:
            raise ValueError(f"Need at least {n_splits + 1} years for {n_splits} CV splits, got {len(available_years)}")
        
        splits = []
        
        # Start with enough years for training, then add one test year at a time
        min_train_years = len(available_years) - n_splits
        
        for i in range(n_splits):
            train_end_year = available_years[min_train_years + i - 1]  # Fix off-by-one error
            test_year = available_years[min_train_years + i]
            
            train_data = df[df['Year'] <= train_end_year]
            test_data = df[df['Year'] == test_year]
            
            print(f"  Split {i+1}: Train on {sorted(train_data['Year'].unique())} -> Test on {test_year}")
            splits.append((train_data, test_data))
        
        return splits

    def evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                 split_name: str = "Test") -> Dict[str, float]:
        """
        Comprehensive model performance evaluation with multiple metrics.
        This tracks all the key metrics data scientists care about.
        """
        predictions = model.predict(X_test)
        
        # Calculate standard regression metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Calculate percentage error metrics
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 0.1))) * 100  # Avoid division by zero
        
        # Calculate prediction accuracy within reasonable ranges
        # For fantasy football, being within 2 points is pretty good
        within_1_point = np.mean(np.abs(y_test - predictions) <= 1.0) * 100
        within_2_points = np.mean(np.abs(y_test - predictions) <= 2.0) * 100
        within_5_points = np.mean(np.abs(y_test - predictions) <= 5.0) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'within_1_point': within_1_point,
            'within_2_points': within_2_points,
            'within_5_points': within_5_points
        }
        
        # Print performance summary
        print(f"\n{split_name} Performance Metrics:")
        print(f"  MAE (Mean Absolute Error): {mae:.3f} fantasy points")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.3f}")
        print(f"  R² (Coefficient of Determination): {r2:.3f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape:.1f}%")
        print(f"  Predictions within 1 point: {within_1_point:.1f}%")
        print(f"  Predictions within 2 points: {within_2_points:.1f}%")
        print(f"  Predictions within 5 points: {within_5_points:.1f}%")
        
        return metrics

    def robust_cross_validation_training(self, df: pd.DataFrame, 
                                       optimize_hyperparameters: bool = True) -> Dict[str, any]:
        """
        Train model using proper time series cross-validation.
        This prevents data leakage and gives more realistic performance estimates.
        """
        print("Starting robust cross-validation training...")
        
        # Create time series cross-validation splits
        try:
            cv_splits = self.time_series_cross_validation(df, n_splits=3)
        except ValueError as e:
            print(f"Warning: Could not create time series CV: {e}")
            print("Falling back to temporal split...")
            train_data, test_data = self.create_temporal_split(df, test_seasons=1)
            cv_splits = [(train_data, test_data)]
        
        # Store results from each fold
        cv_results = []
        feature_importance_across_folds = []
        
        for fold, (train_data, test_data) in enumerate(cv_splits):
            print(f"\n--- Fold {fold + 1}/{len(cv_splits)} ---")
            
            # Prepare features for this fold
            potential_features = [
                'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1',
                'Yds', 'Yds.1', 'Yds.2', 'Total_Yards', 'Total_Yards_Per_Game',
                'TD', 'TD.1', 'Total_TDs', 'Total_TDs_Per_Game',
                'Attempts_Per_Game', 'Targets_Per_Game', 'Receptions_Per_Game',
                'Rush_TD_Per_Game', 'Rec_TD_Per_Game',
                'Yards_Per_Carry', 'Yards_Per_Target', 'Catch_Rate',
                'Rush_TD_Rate', 'Rec_TD_Rate',
                'Int', 'FL', 'Fmb', '2PM', '2PP',
                'injury_risk_score', 'recent_major_injury', 'career_injury_rate',
                'major_injuries_count', 'injury_multiplier'
            ]
            
            # Add position dummy variables
            pos_columns = [col for col in train_data.columns if col.startswith('Pos_')]
            potential_features.extend(pos_columns)
            
            # Use only features that exist in the data
            fold_features = [f for f in potential_features if f in train_data.columns]
            
            # Prepare train and test sets for this fold
            X_train = train_data[fold_features]
            y_train = train_data['FPPG']
            X_test = test_data[fold_features]
            y_test = test_data['FPPG']
            
            # Ensure numeric data and handle missing values
            for feature in fold_features:
                X_train[feature] = pd.to_numeric(X_train[feature], errors='coerce').fillna(0)
                X_test[feature] = pd.to_numeric(X_test[feature], errors='coerce').fillna(0)
            
            # Clean up infinite values
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
            # Scale features using only training data (prevent data leakage)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
            
            # Train model for this fold
            if optimize_hyperparameters and fold == 0:  # Only optimize on first fold to save time
                print("Optimizing hyperparameters...")
                best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
                self.best_params = best_params  # Store for future use
            else:
                # Use default or previously optimized parameters
                best_params = getattr(self, 'best_params', {
                    'objective': 'reg:squarederror',
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                })
            
            # Ensure we have valid parameters
            if best_params is None:
                best_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            
            # Train model
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            train_metrics = self.evaluate_model_performance(model, X_train_scaled, y_train, f"Fold {fold+1} Train")
            test_metrics = self.evaluate_model_performance(model, X_test_scaled, y_test, f"Fold {fold+1} Test")
            
            # Store results
            fold_result = {
                'fold': fold + 1,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_count': len(fold_features),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            cv_results.append(fold_result)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': fold_features,
                    'importance': model.feature_importances_
                })
                feature_importance_across_folds.append(importance_df)
        
        # Calculate average performance across folds
        print(f"\n=== Cross-Validation Summary ===")
        avg_train_mae = np.mean([result['train_metrics']['mae'] for result in cv_results])
        avg_test_mae = np.mean([result['test_metrics']['mae'] for result in cv_results])
        avg_test_r2 = np.mean([result['test_metrics']['r2'] for result in cv_results])
        
        print(f"Average Train MAE: {avg_train_mae:.3f}")
        print(f"Average Test MAE: {avg_test_mae:.3f}")
        print(f"Average Test R²: {avg_test_r2:.3f}")
        
        # Check for overfitting
        mae_gap = avg_test_mae - avg_train_mae
        if mae_gap > 1.0:
            print(f"Warning: Significant performance gap between train and test (MAE gap: {mae_gap:.3f})")
            print("This suggests the model may be overfitting. Consider:")
            print("  - Reducing model complexity")
            print("  - Adding regularization")
            print("  - Using fewer features")
        
        return {
            'cv_results': cv_results,
            'avg_train_mae': avg_train_mae,
            'avg_test_mae': avg_test_mae,
            'avg_test_r2': avg_test_r2,
            'feature_importance': feature_importance_across_folds,
            'best_params': best_params if 'best_params' in locals() else None
        }

    def load_historical_data(self, years=list(range(2015, 2025))):
        # Pro Football Reference has solid historical data - 10 years should be plenty
        print(f"Loading fantasy football data for years: {years}")

        all_data = []
        
        # Hit each year individually - takes a bit but gets us clean data
        for year in years:
            print(f"  Loading {year} data...")
            
            url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
            
            try:
                # pandas read_html is magic for scraping tables
                df = pd.read_html(url, header=1)[0]
                
                # pro-football-reference repeats headers throughout the table, annoying
                df = df[df['Rk'] != 'Rk']  
                df = df.fillna(0)  # zeros better than NaN for our purposes
                
                # convert everything to numbers - some cols might not exist in older years
                numeric_columns = [
                    'FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1', 'Yds', 'TD', 'Int', 
                    'Yds.1', 'TD.1', 'Yds.2', 'TD.2', 'FL', 'Fmb', 'Rush TD', 'Rec TD', 
                    'Ret TD', '2PM', '2PP', 'FantPt/G', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'
                ]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce') # coerce handles the weird values
                
                # this is where the magic happens - create all our advanced features
                df = self._engineer_features(df)
                
                # need at least 4 games to have meaningful stats
                df = df[df['G'] >= 4]
                
                df['Year'] = year
                all_data.append(df)
                print(f"Successfully loaded {len(df)} players from {year}")
                
                # be nice to their servers
                time.sleep(1)
                
            except Exception as e:
                print(f"Error loading {year} data: {e}")
                continue
        
        if all_data:
            # smash it all together
            self.historical_data = pd.concat(all_data, ignore_index=True)
            total_players = len(self.historical_data)
            print(f"\nCombined dataset: {total_players} total player-seasons")
            print(f"Years included: {sorted(self.historical_data['Year'].unique())}")
            return self.historical_data
        else:
            print("Uh oh, nothing loaded successfully")
            return None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This is where we create all the fancy features that actually matter
        """
        # main target variable - handle different column names
        if 'FantPt' in df.columns:
            df['FPPG'] = df['FantPt'] / df['G']
        elif 'FPTS' in df.columns:
            df['FPPG'] = df['FPTS'] / df['G']
        else:
            # Try to calculate from available scoring data
            df['FPPG'] = 0.0  # Will be overwritten if we find scoring data
        
        # efficiency is king in fantasy - yards per opportunity
        if 'Yds' in df.columns and 'Att' in df.columns:
            df['Yards_Per_Carry'] = df['Yds'] / df['Att'].replace(0, 1) # avoid divide by zero
        
        if 'Yds.1' in df.columns and 'Tgt' in df.columns:
            df['Yards_Per_Target'] = df['Yds.1'] / df['Tgt'].replace(0, 1)
        
        if 'Rec' in df.columns and 'Tgt' in df.columns:
            df['Catch_Rate'] = df['Rec'] / df['Tgt'].replace(0, 1) # crucial for WRs/TEs
        
        # volume is everything - normalize by games played
        df['Attempts_Per_Game'] = df['Att'] / df['G']
        df['Targets_Per_Game'] = df['Tgt'] / df['G'] # this usually predicts fantasy success
        df['Receptions_Per_Game'] = df['Rec'] / df['G']
        
        # touchdown rates - some guys just find the endzone
        if 'TD' in df.columns:
            df['Rush_TD_Per_Game'] = df['TD'] / df['G']
            df['Rush_TD_Rate'] = df['TD'] / df['Att'].replace(0, 1) # TDs per carry
        
        if 'TD.1' in df.columns:
            df['Rec_TD_Per_Game'] = df['TD.1'] / df['G']
            df['Rec_TD_Rate'] = df['TD.1'] / df['Rec'].replace(0, 1) # TDs per catch
        
        # combine rushing and receiving for dual-threat players
        if 'Yds' in df.columns and 'Yds.1' in df.columns:
            df['Total_Yards'] = df['Yds'].fillna(0) + df['Yds.1'].fillna(0)
            df['Total_Yards_Per_Game'] = df['Total_Yards'] / df['G']
        
        if 'TD' in df.columns and 'TD.1' in df.columns:
            df['Total_TDs'] = df['TD'].fillna(0) + df['TD.1'].fillna(0)
            df['Total_TDs_Per_Game'] = df['Total_TDs'] / df['G']
        
        # Remove this feature - it was causing data leakage by using the target variable
        # A proper consistency metric would require weekly fantasy points data
        # df['Fantasy_Points_Consistency'] = df['FPPG']  # REMOVED - data leakage!
        
        # position matters a lot - QBs vs RBs have totally different patterns
        pos_col = None
        if 'Pos' in df.columns:
            pos_col = 'Pos'
        elif 'FantPos' in df.columns:
            pos_col = 'FantPos'
        
        if pos_col:
            pos_dummies = pd.get_dummies(df[pos_col], prefix='Pos')
            df = pd.concat([df, pos_dummies], axis=1)
        
        # ADD INJURY HISTORY FEATURES - only if injury history is loaded
        if 'Player' in df.columns and self.player_injury_history:
            df = self._add_injury_features(df)
        
        # TODO: add age and team pace when I get around to scraping that
        
        # clean up any leftover NaNs
        df = df.fillna(0)
        
        return df
    
    def _add_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add injury history features to the dataframe
        This is where we account for season-ending injuries and injury-prone players
        """
        print("Adding injury history features to training data...")
        
        # initialize injury features - always add these even if no injury data
        df['injury_risk_score'] = 0.0
        df['recent_major_injury'] = 0
        df['career_injury_rate'] = 0.0
        df['major_injuries_count'] = 0
        df['injury_multiplier'] = 1.0
        
        # if no injury history data, return with default values
        if not self.player_injury_history:
            print("   No injury history data available - using default values")
            return df
        
        print(f"   Processing injury features for {len(self.player_injury_history)} players with injury data")
        
        # process each player
        for idx, row in df.iterrows():
            player_name = row.get('Player', '')
            
            # clean player name - remove team abbreviation
            clean_name = player_name
            if ' ' in player_name:
                parts = player_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                    clean_name = ' '.join(parts[:-1])
            
            # get injury multiplier and features
            injury_multiplier = self.get_player_injury_multiplier(clean_name)
            
            # find matching injury history
            injury_features = None
            if clean_name in self.player_injury_history:
                injury_features = self.player_injury_history[clean_name]
            else:
                # try fuzzy matching
                for stored_name, features in self.player_injury_history.items():
                    if (clean_name.lower() in stored_name.lower() or 
                        stored_name.lower() in clean_name.lower()):
                        injury_features = features
                        break
            
            # populate injury features
            if injury_features:
                df.at[idx, 'injury_risk_score'] = injury_features['injury_risk_score']
                df.at[idx, 'recent_major_injury'] = injury_features['recent_major_injury']
                df.at[idx, 'career_injury_rate'] = injury_features['career_injury_rate']
                df.at[idx, 'major_injuries_count'] = injury_features['major_injuries_count']
            
            df.at[idx, 'injury_multiplier'] = injury_multiplier
        
        # show some stats on injury impact
        injured_players = df[df['injury_risk_score'] > 0.1]
        if len(injured_players) > 0:
            print(f"Found injury history for {len(injured_players)} players in training data")
            print(f"Average injury risk score: {injured_players['injury_risk_score'].mean():.3f}")
            print(f"Players with recent major injuries: {injured_players['recent_major_injury'].sum()}")
        
        return df
    
    def scrape_qb_wr_connections(self, years=list(range(2020, 2025))):
        """
        This is where I scrape the data I need for QB-WR chemistry analysis
        PFR doesn't have direct QB-WR connection data, so I'm being clever here
        """
        print(f"Scraping QB-WR connection data for chemistry analysis...")
        
        all_connections = []
        
        for year in years:
            print(f"  Loading {year} QB-WR connection data...")
            
            try:
                # I'm using the same fantasy data but focusing on WRs/TEs this time
                receiving_url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                receiving_df = pd.read_html(receiving_url, header=1)[0]
                receiving_df = receiving_df[receiving_df['Rk'] != 'Rk']  # remove repeated headers
                receiving_df = receiving_df.fillna(0)
                
                # only want pass catchers for chemistry analysis
                pos_col = 'FantPos' if 'FantPos' in receiving_df.columns else 'Pos'
                if pos_col in receiving_df.columns:
                    wr_te_data = receiving_df[receiving_df[pos_col].isin(['WR', 'TE'])].copy()
                    
                    if 'Player' in wr_te_data.columns:
                        # use the Tm column for team abbreviations
                        wr_te_data['Team'] = wr_te_data['Tm'] if 'Tm' in wr_te_data.columns else None
                        wr_te_data['CleanPlayer'] = wr_te_data['Player'].str.replace(r'[*+]', '', regex=True)  # remove PFR markers
                        
                        wr_te_data['Year'] = year
                        
                        # convert the important stats to numbers - same as before
                        numeric_cols = ['Tgt', 'Rec', 'Yds.1', 'TD.1', 'G']
                        for col in numeric_cols:
                            if col in wr_te_data.columns:
                                wr_te_data[col] = pd.to_numeric(wr_te_data[col], errors='coerce').fillna(0)
                        
                        # calculate the metrics that matter for QB-WR chemistry
                        wr_te_data['Targets_Per_Game'] = wr_te_data['Tgt'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Receptions_Per_Game'] = wr_te_data['Rec'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Rec_Yards_Per_Game'] = wr_te_data['Yds.1'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Rec_TDs_Per_Game'] = wr_te_data['TD.1'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Catch_Rate'] = wr_te_data['Rec'] / wr_te_data['Tgt'].replace(0, 1)  # this is key for chemistry
                        
                        all_connections.append(wr_te_data)
                        
                        print(f"    Found {len(wr_te_data)} WR/TE connections for {year}")
                
                time.sleep(1.5)  # don't hammer their servers
                
            except Exception as e:
                print(f"Error loading {year} connection data: {e}")
                continue
        
        if all_connections:
            self.play_by_play_data = pd.concat(all_connections, ignore_index=True)
            print(f"\nTotal QB-WR connections loaded: {len(self.play_by_play_data)}")
            return self.play_by_play_data
        else:
            print("No connection data loaded successfully")
            return None
    
    def scrape_team_qb_data(self, years=list(range(2020, 2025))):
        """
        Need to grab QB data so I can match them up with the WRs by team
        This is how I figure out which QB was throwing to which WR each year
        """
        print("Scraping team QB information...")
        
        qb_team_data = {}
        
        for year in years:
            print(f"  Loading {year} QB data...")
            
            try:
                # same fantasy data source, just filtering for QBs now
                url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                df = pd.read_html(url, header=1)[0]
                df = df[df['Rk'] != 'Rk']  # get rid of header repeats
                df = df.fillna(0)
                
                # only want quarterbacks
                pos_col = 'FantPos' if 'FantPos' in df.columns else 'Pos'
                if pos_col in df.columns:
                    qb_data = df[df[pos_col] == 'QB'].copy()
                    
                    if 'Player' in qb_data.columns:
                        # use the Tm column for team abbreviations
                        qb_data['Team'] = qb_data['Tm'] if 'Tm' in qb_data.columns else None
                        qb_data['CleanPlayer'] = qb_data['Player'].str.replace(r'[*+]', '', regex=True)  # remove PFR markers
                        qb_data['Year'] = year
                        
                        # make sure key stats are numbers
                        for col in ['G', 'Cmp', 'Att.1', 'Yds.2', 'TD.2']:
                            if col in qb_data.columns:
                                qb_data[col] = pd.to_numeric(qb_data[col], errors='coerce').fillna(0)
                        
                        # store by team-year so I can easily match with WRs later
                        for _, qb in qb_data.iterrows():
                            team = qb['Team']
                            if team and not pd.isna(team):
                                key = f"{team}_{year}"
                                if key not in qb_team_data:
                                    qb_team_data[key] = []
                                qb_team_data[key].append(qb.to_dict())
                
                time.sleep(1)  # be nice to PFR
                
            except Exception as e:
                print(f"Error loading {year} QB data: {e}")
                continue
        
        print(f"QB data loaded for {len(qb_team_data)} team-year combinations")
        return qb_team_data
    
    def calculate_qb_wr_chemistry(self):
        """
        This is the magic - where I calculate how well QBs and WRs work together
        Using historical data to see which pairs have that special connection
        """
        if self.play_by_play_data is None or len(self.play_by_play_data) == 0:
            print("No connection data available. Please run scrape_qb_wr_connections() first.")
            return None
        
        print("Calculating QB-WR chemistry scores...")
        
        # need the QB data to match up with WRs - use same years as WR data
        wr_years = self.play_by_play_data['Year'].unique() if len(self.play_by_play_data) > 0 else [2023, 2024]
        qb_data = self.scrape_team_qb_data(wr_years)
        
        chemistry_scores = {}
        
        # go through all the WR data and match with their QBs
        wr_data = self.play_by_play_data.copy()
        
        for _, wr in wr_data.iterrows():
            team = wr['Team']
            year = wr['Year']
            wr_name = wr['CleanPlayer']
            
            if not team or pd.isna(team):
                continue
            
            # find the main QB for this team in this year
            qb_key = f"{team}_{year}"
            if qb_key not in qb_data:
                continue
            
            # pick the QB who played the most games (usually the starter)
            team_qbs = qb_data[qb_key]
            primary_qb = max(team_qbs, key=lambda x: x.get('G', 0))
            qb_name = primary_qb['CleanPlayer']
            
            # create unique key for this QB-WR pair
            chemistry_key = f"{qb_name}_{wr_name}"
            
            if chemistry_key not in chemistry_scores:
                chemistry_scores[chemistry_key] = {
                    'qb_name': qb_name,
                    'wr_name': wr_name,
                    'years_together': [],
                    'total_games': 0,
                    'total_targets': 0,
                    'total_receptions': 0,
                    'total_yards': 0,
                    'total_tds': 0,
                    'avg_catch_rate': 0,
                    'chemistry_score': 0,
                    'consistency_score': 0
                }
            
            # add up all their stats together over the years
            chem = chemistry_scores[chemistry_key]
            chem['years_together'].append(year)
            chem['most_recent_year'] = max(chem.get('most_recent_year', year), year)
            chem['total_games'] += wr.get('G', 0)
            chem['total_targets'] += wr.get('Tgt', 0)
            chem['total_receptions'] += wr.get('Rec', 0)
            chem['total_yards'] += wr.get('Yds.1', 0)
            chem['total_tds'] += wr.get('TD.1', 0)
        
        # now calculate the actual chemistry scores
        for key, chem in chemistry_scores.items():
            if chem['total_targets'] > 0 and chem['total_games'] > 0:
                # basic efficiency numbers - how well did they connect?
                chem['avg_catch_rate'] = chem['total_receptions'] / chem['total_targets']
                chem['yards_per_target'] = chem['total_yards'] / chem['total_targets']
                chem['tds_per_target'] = chem['total_tds'] / chem['total_targets']
                chem['targets_per_game'] = chem['total_targets'] / chem['total_games']
                
                # bonus points for playing together multiple years - chemistry builds over time
                years_together = len(set(chem['years_together']))
                longevity_bonus = min(years_together * 0.1, 0.3)  # cap at 30% bonus
                
                # high target share means QB trusts this WR
                volume_score = min(chem['targets_per_game'] / 8.0, 1.0)  # 8 targets/game is elite
                
                # catch rate is huge for chemistry - means they're on the same page
                catch_rate_score = chem['avg_catch_rate']  # already 0-1
                td_efficiency_score = min(chem['tds_per_target'] / 0.08, 1.0)  # 8% TD rate is solid
                
                # combine it all into final chemistry score
                base_score = (catch_rate_score * 0.4 +  # catch rate is most important
                             volume_score * 0.3 +       # volume shows trust
                             td_efficiency_score * 0.3)  # red zone chemistry matters
                
                chem['chemistry_score'] = base_score * (1 + longevity_bonus)
                
                # consistency bonus for guys who've been together multiple years
                if years_together >= 2:
                    chem['consistency_score'] = min(years_together / 3.0, 1.0)
                else:
                    chem['consistency_score'] = 0.5  # still decent for one year
        
        # filter out the noise - need at least 20 targets to be meaningful
        filtered_chemistry = {k: v for k, v in chemistry_scores.items() 
                            if v['total_targets'] >= 20}
        
        self.qb_wr_chemistry_data = filtered_chemistry
        
        print(f"Calculated chemistry scores for {len(filtered_chemistry)} QB-WR combinations")
        
        # show off the best chemistry pairs - this is the good stuff
        if filtered_chemistry:
            # Filter for current season (2024) relevant pairs only
            current_season_pairs = {}
            for key, chem in filtered_chemistry.items():
                # Check if this pair has recent activity (2023 or 2024)
                if chem.get('most_recent_year', 0) >= 2023:
                    current_season_pairs[key] = chem
            
            if current_season_pairs:
                sorted_pairs = sorted(current_season_pairs.items(), 
                                    key=lambda x: x[1]['chemistry_score'], 
                                    reverse=True)
                
                print(f"\nTop 10 Current Season QB-WR Chemistry Pairs:")
                print("-" * 50)
                for i, (key, chem) in enumerate(sorted_pairs[:10], 1):
                    print(f"{i:2d}. {chem['qb_name']} -> {chem['wr_name']}: "
                          f"{chem['chemistry_score']:.3f} "
                          f"({chem['total_targets']} targets, "
                          f"{chem['avg_catch_rate']:.1%} catch rate)")
            else:
                print("\nNo current season QB-WR chemistry pairs found")
        
        return self.qb_wr_chemistry_data
    
    def get_chemistry_multiplier(self, qb_name, wr_name):
        """
        Convert chemistry score to a projection multiplier
        Good chemistry = slight boost, bad chemistry = slight penalty
        """
        if not self.qb_wr_chemistry_data:
            return 1.0  # no data, no adjustment
        
        # try exact match first
        key = f"{qb_name}_{wr_name}"
        if key in self.qb_wr_chemistry_data:
            chem_score = self.qb_wr_chemistry_data[key]['chemistry_score']
            # convert chemistry score (0-1.0) to multiplier (0.9-1.2) - more balanced
            # Good chemistry (0.7+) gets boost, poor chemistry (0.5-) gets penalty
            if chem_score >= 0.7:  # Strong chemistry
                multiplier = 1.0 + ((chem_score - 0.7) * 0.67)  # 0.7->1.0, 1.0->1.2
            elif chem_score >= 0.5:  # Average chemistry  
                multiplier = 0.95 + ((chem_score - 0.5) * 0.25)  # 0.5->0.95, 0.7->1.0
            else:  # Poor chemistry
                multiplier = 0.9 + (chem_score * 0.1)  # 0->0.9, 0.5->0.95
            return min(max(multiplier, 0.9), 1.2)  # reasonable range
        
        # try fuzzy matching in case names don't match exactly
        for chem_key, chem_data in self.qb_wr_chemistry_data.items():
            stored_qb = chem_data['qb_name'].lower()
            stored_wr = chem_data['wr_name'].lower()
            
            # simple name matching - could be better but works for now
            if (qb_name.lower() in stored_qb or stored_qb in qb_name.lower()) and \
               (wr_name.lower() in stored_wr or stored_wr in wr_name.lower()):
                chem_score = chem_data['chemistry_score']
                # use same logic as exact match
                if chem_score >= 0.7:  # Strong chemistry
                    multiplier = 1.0 + ((chem_score - 0.7) * 0.67)  # 0.7->1.0, 1.0->1.2
                elif chem_score >= 0.5:  # Average chemistry  
                    multiplier = 0.95 + ((chem_score - 0.5) * 0.25)  # 0.5->0.95, 0.7->1.0
                else:  # Poor chemistry
                    multiplier = 0.9 + (chem_score * 0.1)  # 0->0.9, 0.5->0.95
                return min(max(multiplier, 0.9), 1.2)
        
        # no match found, neutral multiplier
        return 1.0
    
    def analyze_team_rb_support(self, years=list(range(2020, 2025))):
        """
        Analyze RB support quality for QB performance multipliers
        Good RBs take pressure off QBs and create better opportunities
        """
        print("Analyzing team RB support for QB multipliers...")
        
        team_rb_data = {}
        
        for year in years:
            print(f"  Analyzing {year} RB support...")
            
            try:
                # get the same fantasy data but focus on RBs this time
                url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                df = pd.read_html(url, header=1)[0]
                df = df[df['Rk'] != 'Rk'].fillna(0)
                
                pos_col = 'FantPos' if 'FantPos' in df.columns else 'Pos'
                if pos_col in df.columns:
                    rb_data = df[df[pos_col] == 'RB'].copy()
                    
                    if 'Player' in rb_data.columns:
                        # use the Tm column for team abbreviations
                        rb_data['Team'] = rb_data['Tm'] if 'Tm' in rb_data.columns else None
                        rb_data['CleanPlayer'] = rb_data['Player'].str.replace(r'[*+]', '', regex=True)  # remove PFR markers
                        rb_data['Year'] = year
                        
                        # make sure key stats are numeric
                        numeric_cols = ['G', 'Att', 'Yds', 'TD', 'Tgt', 'Rec', 'Yds.1', 'TD.1']
                        for col in numeric_cols:
                            if col in rb_data.columns:
                                rb_data[col] = pd.to_numeric(rb_data[col], errors='coerce').fillna(0)
                        
                        # calculate RB efficiency metrics that help QBs
                        rb_data['Rush_YPG'] = rb_data['Yds'] / rb_data['G'].replace(0, 1)
                        rb_data['Rush_YPC'] = rb_data['Yds'] / rb_data['Att'].replace(0, 1)
                        rb_data['Rush_TDs_Per_Game'] = rb_data['TD'] / rb_data['G'].replace(0, 1)
                        rb_data['Rec_YPG'] = rb_data['Yds.1'] / rb_data['G'].replace(0, 1)
                        rb_data['Total_YPG'] = rb_data['Rush_YPG'] + rb_data['Rec_YPG']
                        
                        # aggregate by team to get team RB support quality
                        for team in rb_data['Team'].dropna().unique():
                            team_rbs = rb_data[rb_data['Team'] == team]
                            
                            # find the primary back (most rushes)
                            if len(team_rbs) > 0:
                                primary_rb = team_rbs.loc[team_rbs['Att'].idxmax()]
                                
                                # calculate team RB support metrics
                                team_key = f"{team}_{year}"
                                team_rb_data[team_key] = {
                                    'primary_rb': primary_rb['CleanPlayer'],
                                    'primary_rb_ypg': primary_rb['Total_YPG'],
                                    'primary_rb_ypc': primary_rb['Rush_YPC'],
                                    'primary_rb_att_pg': primary_rb['Att'] / primary_rb['G'] if primary_rb['G'] > 0 else 0,
                                    'total_team_rush_ypg': team_rbs['Rush_YPG'].sum(),  # all RBs combined
                                    'total_team_rush_att': team_rbs['Att'].sum(),
                                    'rb_committee_score': len(team_rbs[team_rbs['Att'] >= 50]),  # how many significant contributors
                                    'team': team,
                                    'year': year
                                }
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error loading {year} RB data: {e}")
                continue
        
        print(f"RB support data loaded for {len(team_rb_data)} team-year combinations")
        return team_rb_data
    
    def analyze_team_oline_protection(self, years=list(range(2020, 2025))):
        """
        Analyze O-line protection quality using QB stats as proxy
        Better O-lines = fewer sacks, more time, better QB performance
        """
        print("Analyzing O-line protection for QB multipliers...")
        
        team_oline_data = {}
        
        for year in years:
            print(f"  Analyzing {year} O-line protection...")
            
            try:
                # get QB data to calculate sack rates and protection metrics
                url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                df = pd.read_html(url, header=1)[0]
                df = df[df['Rk'] != 'Rk'].fillna(0)
                
                pos_col = 'FantPos' if 'FantPos' in df.columns else 'Pos'
                if pos_col in df.columns:
                    qb_data = df[df[pos_col] == 'QB'].copy()
                    
                    if 'Player' in qb_data.columns:
                        qb_data['Team'] = qb_data['Tm'] if 'Tm' in qb_data.columns else None
                        qb_data['CleanPlayer'] = qb_data['Player'].str.replace(r'[*+]', '', regex=True)  # remove PFR markers
                        qb_data['Year'] = year
                        
                        # convert key stats - need passing attempts and any sack data if available
                        numeric_cols = ['G', 'Cmp', 'Att.1', 'Yds.2', 'TD.2', 'Int', 'Att', 'Yds']  # Att.1 = pass att, Att = rush att
                        for col in numeric_cols:
                            if col in qb_data.columns:
                                qb_data[col] = pd.to_numeric(qb_data[col], errors='coerce').fillna(0)
                        
                        # calculate protection metrics per team
                        for team in qb_data['Team'].dropna().unique():
                            team_qbs = qb_data[qb_data['Team'] == team]
                            primary_qb = team_qbs.loc[team_qbs['Att.1'].idxmax()]  # most pass attempts
                            
                            if primary_qb['Att.1'] > 0:  # must have passing attempts
                                team_key = f"{team}_{year}"
                                
                                # proxy metrics for O-line quality
                                rush_yards_per_att = float(primary_qb['Yds']) / max(1, float(primary_qb['Att']))  # QB rushing efficiency
                                completion_pct = float(primary_qb['Cmp']) / max(1, float(primary_qb['Att.1']))
                                
                                # high QB rushing usually means poor pocket protection
                                scramble_factor = float(primary_qb['Att']) / max(1, float(primary_qb['G']))
                                
                                team_oline_data[team_key] = {
                                    'primary_qb': primary_qb['CleanPlayer'],
                                    'qb_completion_pct': completion_pct,
                                    'qb_scramble_att_pg': scramble_factor,  # high = poor protection
                                    'qb_rush_ypc': rush_yards_per_att,
                                    'pass_attempts_pg': float(primary_qb['Att.1']) / max(1, float(primary_qb['G'])),
                                    'team': team,
                                    'year': year
                                }
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error loading {year} O-line data: {e}")
                continue
        
        print(f"O-line protection data loaded for {len(team_oline_data)} team-year combinations")
        return team_oline_data
    
    def calculate_qb_support_multipliers(self):
        """
        Calculate QB performance multipliers based on RB support and O-line protection
        This is where I give QBs credit for having good supporting cast
        """
        print("Calculating QB support multipliers...")
        
        # get the support data
        rb_data = self.analyze_team_rb_support()
        oline_data = self.analyze_team_oline_protection()
        
        qb_multipliers = {}
        
        # combine RB and O-line data by team-year
        all_team_years = set(rb_data.keys()) | set(oline_data.keys())
        
        for team_year in all_team_years:
            team, year = team_year.split('_')
            year = int(year)
            
            rb_support_score = 0.5  # default neutral
            oline_support_score = 0.5  # default neutral
            
            # calculate RB support score
            if team_year in rb_data:
                rb_info = rb_data[team_year]
                
                # elite RB = big help for QB
                primary_ypg = rb_info['primary_rb_ypg']
                primary_ypc = rb_info['primary_rb_ypc']
                primary_att_pg = rb_info['primary_rb_att_pg']
                committee_score = rb_info['rb_committee_score']
                
                # scoring logic - higher is better for QB
                if primary_ypg >= 100 and primary_ypc >= 4.5:  # elite RB
                    rb_support_score = 0.9
                elif primary_ypg >= 80 and primary_ypc >= 4.0:  # good RB
                    rb_support_score = 0.7
                elif primary_ypg >= 60:  # decent RB
                    rb_support_score = 0.6
                else:  # weak/committee backfield
                    rb_support_score = 0.3
                
                # committee penalty - QBs do better with a clear lead back
                if committee_score >= 3:  # too many cooks
                    rb_support_score *= 0.8
            
            # calculate O-line support score
            if team_year in oline_data:
                oline_info = oline_data[team_year]
                
                completion_pct = oline_info['qb_completion_pct']
                scramble_att_pg = oline_info['qb_scramble_att_pg']
                
                # high completion % usually means good protection/scheme
                if completion_pct >= 0.68:  # elite completion rate
                    oline_support_score = 0.8
                elif completion_pct >= 0.62:  # good completion rate
                    oline_support_score = 0.65
                elif completion_pct >= 0.58:  # average
                    oline_support_score = 0.5
                else:  # poor completion rate
                    oline_support_score = 0.35
                
                # scrambling penalty - high scrambles = poor protection
                if scramble_att_pg >= 6:  # lots of scrambling
                    oline_support_score *= 0.7
                elif scramble_att_pg >= 4:  # some scrambling
                    oline_support_score *= 0.85
            
            # combine RB and O-line scores into final multiplier
            combined_score = (rb_support_score * 0.4 + oline_support_score * 0.6)  # O-line slightly more important
            
            # convert to fantasy multiplier (0.9x to 1.2x range) - more meaningful spread
            qb_multiplier = 0.9 + (combined_score * 0.3)  # scales 0-1 to 0.9-1.2
            qb_multiplier = min(max(qb_multiplier, 0.9), 1.2)  # cap the range
            
            # get QB name for this team-year
            qb_name = None
            if team_year in oline_data:
                qb_name = oline_data[team_year]['primary_qb']
            
            if qb_name:
                qb_key = f"{qb_name}_{team}_{year}"
                qb_multipliers[qb_key] = {
                    'qb_name': qb_name,
                    'team': team,
                    'year': year,
                    'rb_support_score': rb_support_score,
                    'oline_support_score': oline_support_score,
                    'combined_score': combined_score,
                    'qb_multiplier': qb_multiplier,
                    'primary_rb': rb_data.get(team_year, {}).get('primary_rb', 'Unknown'),
                    'rb_ypg': rb_data.get(team_year, {}).get('primary_rb_ypg', 0)
                }
        
        self.qb_multiplier_data = qb_multipliers
        
        print(f"Calculated support multipliers for {len(qb_multipliers)} QB situations")
        
        # show the best and worst supported QBs
        if qb_multipliers:
            sorted_qbs = sorted(qb_multipliers.items(), 
                              key=lambda x: x[1]['qb_multiplier'], 
                              reverse=True)
            
            print("\nTop 10 Best Supported QBs:")
            print("-" * 60)
            for i, (key, qb) in enumerate(sorted_qbs[:10], 1):
                print(f"{i:2d}. {qb['qb_name']} ({qb['team']} {qb['year']}): "
                      f"{qb['qb_multiplier']:.3f}x "
                      f"(RB: {qb['primary_rb']}, {qb['rb_ypg']:.0f} ypg)")
            
            print("\nWorst 5 Supported QBs:")
            print("-" * 60)
            for i, (key, qb) in enumerate(sorted_qbs[-5:], 1):
                print(f"{i:2d}. {qb['qb_name']} ({qb['team']} {qb['year']}): "
                      f"{qb['qb_multiplier']:.3f}x "
                      f"(RB: {qb['primary_rb']}, {qb['rb_ypg']:.0f} ypg)")
        
        return self.qb_multiplier_data
    
    def get_qb_support_multiplier(self, qb_name, team=None, year=None):
        """
        Get the support multiplier for a specific QB
        Accounts for RB support and O-line protection
        """
        if not self.qb_multiplier_data:
            return 1.0  # neutral if no data
        
        # try exact match first
        if team and year:
            key = f"{qb_name}_{team}_{year}"
            if key in self.qb_multiplier_data:
                return self.qb_multiplier_data[key]['qb_multiplier']
        
        # try partial matching
        best_match_multiplier = 1.0
        for qb_key, qb_data in self.qb_multiplier_data.items():
            stored_qb_name = qb_data['qb_name'].lower()
            qb_name_lower = qb_name.lower()
            
            # simple name matching
            if (qb_name_lower in stored_qb_name or stored_qb_name in qb_name_lower):
                # prefer more recent years if multiple matches
                if year is None or qb_data['year'] >= (year - 2):  # within 2 years
                    return qb_data['qb_multiplier']
                else:
                    best_match_multiplier = qb_data['qb_multiplier']
        
        return best_match_multiplier
    
    def load_injury_data(self, years=list(range(2020, 2025))):
        """
        Load historical injury data by scraping Pro Football Reference injury reports
        This will help us track season-ending injuries and their impact
        """
        print(f"Loading injury data from Pro Football Reference for years: {years}")
        
        all_injury_data = []
        
        # Major injury keywords that indicate season-ending or significant injuries
        major_injury_keywords = [
            'ir', 'injured reserve', 'pup', 'physically unable to perform',
            'acl', 'achilles', 'surgery', 'torn', 'rupture', 'broken', 'fracture',
            'season', 'out for season', 'ended', 'surgery'
        ]
        
        for year in years:
            print(f"  Scraping {year} injury data...")
            
            try:
                # Use the same fantasy data but look for injury indicators in player status
                url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                df = pd.read_html(url, header=1)[0]
                df = df[df['Rk'] != 'Rk']  # remove repeated headers
                df = df.fillna('')
                
                # Extract player names and look for injury indicators
                if 'Player' in df.columns:
                    for _, player_row in df.iterrows():
                        player_name = str(player_row['Player']).strip()
                        
                        if player_name and player_name != '':
                            # Clean player name
                            clean_name = player_name
                            if ' ' in player_name:
                                parts = player_name.split()
                                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                                    clean_name = ' '.join(parts[:-1])
                                    team = parts[-1]
                                else:
                                    team = 'UNK'
                            else:
                                team = 'UNK'
                            
                            # Check games played vs season length to infer injuries
                            games_played = player_row.get('G', 0)
                            try:
                                games_played = int(games_played) if games_played != '' else 0
                            except:
                                games_played = 0
                            
                            # If player played significantly fewer games, likely injured
                            injury_severity = 'none'
                            if games_played <= 4 and games_played > 0:
                                injury_severity = 'major'  # Very few games, likely major injury
                            elif games_played <= 8:
                                injury_severity = 'moderate'  # Half season or less
                            elif games_played <= 12:
                                injury_severity = 'minor'  # Missed some games
                            
                            # Create injury record
                            if injury_severity != 'none':
                                injury_record = {
                                    'player_name': clean_name,
                                    'season': year,
                                    'team': team,
                                    'games_played': games_played,
                                    'injury_severity': injury_severity,
                                    'estimated_games_missed': max(0, 17 - games_played),
                                    'position': player_row.get('FantPos', player_row.get('Pos', 'UNK'))
                                }
                                all_injury_data.append(injury_record)
                
                time.sleep(1.5)  # Be respectful to PFR servers
                
            except Exception as e:
                print(f"  Error scraping {year} injury data: {e}")
                continue
        
        if all_injury_data:
            self.injury_data = pd.DataFrame(all_injury_data)
            print(f"Processed {len(all_injury_data)} potential injury cases from {len(years)} years")
            
            # Show injury severity distribution
            severity_counts = self.injury_data['injury_severity'].value_counts()
            print(f"Injury severity distribution: {dict(severity_counts)}")
            
            return self.injury_data
        else:
            print("No injury data could be processed")
            return None
    
    def analyze_player_injury_history(self):
        """
        Process injury data to create meaningful features for each player
        Focus on season-ending injuries and their impact on future performance
        """
        if not hasattr(self, 'injury_data') or self.injury_data is None or len(self.injury_data) == 0:
            print("No injury data available. Please run load_injury_data() first.")
            return None
        
        print("Analyzing player injury histories for season-ending injuries...")
        
        injury_df = self.injury_data.copy()
        player_injury_features = {}
        
        # group by player to analyze their injury history
        if 'player_name' in injury_df.columns:
            grouped = injury_df.groupby('player_name')
            
            for player_name, player_injuries in grouped:
                injury_features = {
                    'player_name': player_name,
                    'total_injury_incidents': len(player_injuries),
                    'years_with_injuries': len(player_injuries['season'].unique()),
                    'major_injuries_count': len(player_injuries[player_injuries['injury_severity'] == 'major']),
                    'moderate_injuries_count': len(player_injuries[player_injuries['injury_severity'] == 'moderate']),
                    'recent_major_injury': 0,  # binary flag for major injury in last 2 years
                    'career_injury_rate': 0.0,
                    'injury_risk_score': 0.0,
                    'total_games_missed': player_injuries['estimated_games_missed'].sum()
                }
                
                # check for recent major injuries (last 2 seasons)
                current_year = max(injury_df['season'])
                recent_injuries = player_injuries[
                    (player_injuries['season'] >= (current_year - 2)) & 
                    (player_injuries['injury_severity'] == 'major')
                ]
                if len(recent_injuries) > 0:
                    injury_features['recent_major_injury'] = 1
                
                # calculate injury risk score (0-1 scale)
                years_active = max(1, injury_features['years_with_injuries'])
                
                # Weight different injury types
                major_weight = injury_features['major_injuries_count'] * 3  # Major injuries heavily weighted
                moderate_weight = injury_features['moderate_injuries_count'] * 1.5  # Moderate injuries less weighted
                recent_weight = injury_features['recent_major_injury'] * 2  # Recent injuries matter more
                
                # Combined risk score - normalize to 0-1 scale
                raw_score = (major_weight + moderate_weight + recent_weight) / years_active
                injury_features['injury_risk_score'] = min(1.0, raw_score / 5.0)  # Scale to 0-1
                
                # career injury rate as percentage of games missed
                total_possible_games = years_active * 17
                injury_features['career_injury_rate'] = min(100.0, 
                    (injury_features['total_games_missed'] / max(1, total_possible_games)) * 100)
                
                player_injury_features[player_name] = injury_features
        
        self.player_injury_history = player_injury_features
        
        print(f"Processed injury history for {len(player_injury_features)} players")
        
        # show players with highest injury risk
        if player_injury_features:
            sorted_players = sorted(player_injury_features.items(), 
                                  key=lambda x: x[1]['injury_risk_score'], 
                                  reverse=True)
            
            print("\nTop 10 Highest Injury Risk Players:")
            print("-" * 70)
            for i, (name, features) in enumerate(sorted_players[:10], 1):
                major = features['major_injuries_count']
                moderate = features['moderate_injuries_count'] 
                recent = 'Yes' if features['recent_major_injury'] else 'No'
                print(f"{i:2d}. {name:25} Risk: {features['injury_risk_score']:.3f} "
                      f"(Major: {major}, Mod: {moderate}, Recent: {recent})")
        
        return self.player_injury_history
    
    def get_player_injury_multiplier(self, player_name):
        """
        Get injury risk multiplier for a player's projection
        Players with injury history get penalized slightly
        """
        if not self.player_injury_history:
            return 1.0  # neutral if no injury data
        
        # try exact match first
        if player_name in self.player_injury_history:
            injury_info = self.player_injury_history[player_name]
            risk_score = injury_info['injury_risk_score']
            
            # convert risk score to multiplier (high risk = lower multiplier)
            # range from 0.75 (high injury risk) to 1.0 (no injury history)
            multiplier = 1.0 - (risk_score * 0.25)
            return max(0.75, multiplier)
        
        # try fuzzy matching
        for stored_name, injury_info in self.player_injury_history.items():
            if (player_name.lower() in stored_name.lower() or 
                stored_name.lower() in player_name.lower()):
                risk_score = injury_info['injury_risk_score']
                multiplier = 1.0 - (risk_score * 0.25)
                return max(0.75, multiplier)
        
        # no injury history found - assume neutral
        return 1.0
    
    def prepare_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Get our data ready for the XGBoost model
        """
        if self.historical_data is None:
            print("No historical data loaded. Please run load_historical_data() first.")
            return None, None
        
        df = self.historical_data.copy()
        
        # Run comprehensive data validation before training
        # This prevents training on corrupted or invalid data
        try:
            self.validate_training_data(df)
            self.validate_feature_engineering(df)
            self.detect_potential_data_leakage(df)
        except ValueError as e:
            print(f"Data validation failed: {e}")
            return None, None
        
        # all the features we might want to use - some might not exist in all years
        potential_features = [
            # basic counting stats
            'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1',
            # yardage totals and per game
            'Yds', 'Yds.1', 'Yds.2', 'Total_Yards', 'Total_Yards_Per_Game',
            # touchdown stats
            'TD', 'TD.1', 'Total_TDs', 'Total_TDs_Per_Game',
            # per game usage - usually the most predictive
            'Attempts_Per_Game', 'Targets_Per_Game', 'Receptions_Per_Game',
            'Rush_TD_Per_Game', 'Rec_TD_Per_Game',
            # efficiency ratios - separates good players from great ones
            'Yards_Per_Carry', 'Yards_Per_Target', 'Catch_Rate',
            'Rush_TD_Rate', 'Rec_TD_Rate',
            # negative plays and special stuff
            'Int', 'FL', 'Fmb', '2PM', '2PP',
            # 'Fantasy_Points_Consistency',  # REMOVED - was causing data leakage
            # NEW: injury history features - accounts for season-ending injuries
            'injury_risk_score', 'recent_major_injury', 'career_injury_rate',
            'major_injuries_count', 'injury_multiplier'
        ]
        
        # grab any position dummy variables we created
        pos_columns = [col for col in df.columns if col.startswith('Pos_')]
        potential_features.extend(pos_columns)
        
        # add injury features if we have injury history data but features are missing
        if self.player_injury_history and 'injury_risk_score' not in df.columns:
            print("Re-engineering features with injury history data...")
            df = self._add_injury_features(df)
        
        # only use features that actually exist in our dataset
        self.features = [f for f in potential_features if f in df.columns]
        
        # debug: check if injury features are present
        injury_features_present = [f for f in self.features if 'injury' in f.lower()]
        if injury_features_present:
            print(f"Injury features included in training: {injury_features_present}")
        elif self.player_injury_history:
            print(f"Debug: Expected injury features but only found: {[f for f in df.columns if 'injury' in f.lower()]}")
            print(f"Debug: Selected features: {self.features}")
        else:
            print("Info: No injury history data loaded - injury features not available")
        
        # make sure everything is numeric
        for feature in self.features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # filter out any weird FPPG values
        df = df[df['FPPG'].notna() & (df['FPPG'] >= 0)]
        
        # X is our features, y is what we're trying to predict (FPPG)
        X = df[self.features]
        y = df['FPPG']
        
        # clean up any infinite values that snuck through
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Training data prepared with {len(X)} samples and {len(self.features)} features")
        print(f"Features: {self.features[:10]}{'...' if len(self.features) > 10 else ''}")
        return X, y
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Let optuna find the best hyperparameters - this takes a while but worth it
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            # test these params with cross-validation
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            return -scores.mean() # return positive MAE
        
        print("Optimizing hyperparameters... grab a coffee, this takes a few minutes")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"Best MAE found: {study.best_value:.3f}")
        return study.best_params
    
    def train_model(self, optimize_hyperparameters: bool = True, use_temporal_validation: bool = True, force_retrain: bool = False) -> Optional[xgb.XGBRegressor]:
        """
        Train our XGBoost model with proper temporal validation to prevent data leakage.
        Now with intelligent caching - loads cached model if data hasn't changed.
        
        Args:
            optimize_hyperparameters: Whether to run hyperparameter optimization
            use_temporal_validation: Whether to use temporal splits (recommended) or random splits
            force_retrain: Whether to force retraining even if cached model exists
        """
        # Try to load cached model first (unless forced to retrain)
        if not force_retrain and self._load_cached_model():
            return self.model
        
        # If we reach here, we need to train a new model
        print("Training new XGBoost model...")
        start_time = time.time()
        
        X, y = self.prepare_training_data()
        if X is None:
            return None
        
        # Use temporal validation if we have historical data with years
        if use_temporal_validation and 'Year' in self.historical_data.columns:
            print("Using temporal validation to prevent data leakage...")
            
            # Perform robust cross-validation training
            cv_results = self.robust_cross_validation_training(
                self.historical_data, 
                optimize_hyperparameters=optimize_hyperparameters
            )
            
            # Now train final model on all available data except most recent year for final test
            try:
                train_data, test_data = self.create_temporal_split(self.historical_data, test_seasons=1)
                
                # Prepare final training data
                potential_features = [
                    'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1',
                    'Yds', 'Yds.1', 'Yds.2', 'Total_Yards', 'Total_Yards_Per_Game',
                    'TD', 'TD.1', 'Total_TDs', 'Total_TDs_Per_Game',
                    'Attempts_Per_Game', 'Targets_Per_Game', 'Receptions_Per_Game',
                    'Rush_TD_Per_Game', 'Rec_TD_Per_Game',
                    'Yards_Per_Carry', 'Yards_Per_Target', 'Catch_Rate',
                    'Rush_TD_Rate', 'Rec_TD_Rate',
                    'Int', 'FL', 'Fmb', '2PM', '2PP',
                    'injury_risk_score', 'recent_major_injury', 'career_injury_rate',
                    'major_injuries_count', 'injury_multiplier'
                ]
                
                pos_columns = [col for col in train_data.columns if col.startswith('Pos_')]
                potential_features.extend(pos_columns)
                final_features = [f for f in potential_features if f in train_data.columns]
                
                X_train = train_data[final_features]
                y_train = train_data['FPPG']
                X_test = test_data[final_features]
                y_test = test_data['FPPG']
                
                # Clean and prepare data
                for feature in final_features:
                    X_train[feature] = pd.to_numeric(X_train[feature], errors='coerce').fillna(0)
                    X_test[feature] = pd.to_numeric(X_test[feature], errors='coerce').fillna(0)
                
                X_train = X_train.replace([np.inf, -np.inf], 0)
                X_test = X_test.replace([np.inf, -np.inf], 0)
                
                # Scale features using training data only
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
                
            except ValueError as e:
                print(f"Temporal split failed: {e}, falling back to random split")
                use_temporal_validation = False
        
        # Fallback to random split if temporal validation not available or requested
        if not use_temporal_validation:
            print("Using random train/test split...")
            # Scale features so they're all on similar ranges
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            
            # 80/20 train/test split
            X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        # For temporal validation, we may already have optimized hyperparameters from CV
        if not hasattr(self, 'best_params') or self.best_params is None:
            if optimize_hyperparameters and len(X_train_scaled) > 100:  
                # run optuna optimization if we have enough data
                self.best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            else:
                # decent default params if we skip optimization
                self.best_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 1,
                    'reg_lambda': 1,
                    'random_state': 42
                }
        
        # train the final model with our best params
        print("Training final XGBoost model...")
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate final model performance using our comprehensive metrics
        final_metrics = self.evaluate_model_performance(self.model, X_test_scaled, y_test, "Final Test")
        
        # If we used temporal validation, we already have robust CV estimates
        if use_temporal_validation and 'Year' in self.historical_data.columns:
            print("\n" + "="*60)
            print("TEMPORAL VALIDATION MODEL TRAINING COMPLETE")
            print("="*60)
            print("Final model trained with temporal validation to prevent data leakage")
            if 'cv_results' in locals():
                print(f"Cross-validation MAE: {cv_results['avg_test_mae']:.3f}")
                print(f"Cross-validation R²: {cv_results['avg_test_r2']:.3f}")
        else:
            # For random split, add traditional cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            print("\n" + "="*50)
            print("MODEL TRAINING RESULTS")
            print("="*50)
            print(f"Test MAE: {final_metrics['mae']:.3f} fantasy points")
            print(f"Test RMSE: {final_metrics['rmse']:.3f}")
            print(f"Test R²: {final_metrics['r2']:.3f} (higher is better)")
            print(f"CV MAE: {cv_mae:.3f} (±{cv_scores.std():.3f})")
        
        # Store feature importance for analysis
        # Update self.features to match what was actually used
        if use_temporal_validation and 'final_features' in locals():
            self.features = final_features
        
        self.feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (what drives fantasy success):")
        print(self.feature_importance.head(10))
        
        # Save the trained model to cache for next time
        self._save_trained_model()
        
        training_time = time.time() - start_time
        print(f"\n+ Model training completed in {training_time:.1f} seconds")
        print("+ Model cached for future runs")
        
        return self.model
    
    def scrape_fantasy_projections(self, position='rb'):
        """
        Scrape player projections from FantasyPros
        """
        print(f"Scraping {position.upper()} projections from FantasyPros...")
        
        url = f"https://www.fantasypros.com/nfl/projections/{position}.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            dfs = pd.read_html(response.text, header=[0, 1])
            df = dfs[0]
            
            # Flatten multi-level column headers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns]
            
            # Find and rename player column
            player_cols = [col for col in df.columns if 'Player' in col]
            if player_cols:
                df = df.rename(columns={player_cols[0]: 'Player'})
            
            # Add position information
            df['Position'] = position.upper()
            
            # Clean player names (remove team info, defense/kicker designations)
            df = df[~df['Player'].str.contains("Defense|Kicker", na=False)]
            
            self.projections_data[position] = df
            print(f"Successfully scraped {len(df)} {position.upper()} players")
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
            
            return df
            
        except Exception as e:
            print(f"Error scraping {position} projections: {e}")
            return None
    
    def scrape_all_positions(self):
        """
        Scrape projections for all relevant fantasy positions
        """
        positions = ['qb', 'rb', 'wr', 'te']
        all_projections = []
        
        for position in positions:
            df = self.scrape_fantasy_projections(position)
            if df is not None:
                all_projections.append(df)
        
        if all_projections:
            combined_df = pd.concat(all_projections, ignore_index=True)
            print(f"\nTotal players scraped: {len(combined_df)}")
            return combined_df
        else:
            return None
    
    def predict_fantasy_points(self, player_stats: Dict) -> Optional[float]:
        """
        Predict FPPG for a single player
        """
        if self.model is None:
            print("Model not trained yet. Please run train_model() first.")
            return None
        
        # build feature array in the same order as training
        stats_array = []
        for feature in self.features:
            stats_array.append(player_stats.get(feature, 0))
        
        # apply the same scaling we used in training
        stats_scaled = self.scaler.transform([stats_array])
        
        prediction = self.model.predict(stats_scaled)[0]
        return max(0, prediction)  # can't have negative fantasy points
    
    def generate_draft_recommendations(self, projections_df=None, top_n=20, use_chemistry=True, use_qb_multipliers=True, use_injury_history=True):
        """
        Generate draft recommendations using optimized enhanced projections.
        Now eliminates redundant calculations and uses caching for efficiency.
        """
        # Use optimized enhanced projections instead of raw projections
        if projections_df is None:
            enhanced_projections = self.get_enhanced_projections(
                use_chemistry=use_chemistry,
                use_qb_multipliers=use_qb_multipliers, 
                use_injury_history=use_injury_history
            )
        else:
            # If projections provided, assume they're already enhanced (backward compatibility)
            enhanced_projections = projections_df
            
        if enhanced_projections is None:
            print("No enhanced projections available")
            return None
        
        print(f"Generating top {top_n} recommendations from {len(enhanced_projections)} enhanced players...")
        
        recommendations = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = enhanced_projections[enhanced_projections['Position'] == position].copy()
            
            if len(pos_players) > 0:
                # Players are already enhanced and sorted by the optimized method
                top_pos = pos_players.head(min(top_n//4, len(pos_players)))
                recommendations.append(top_pos)
                print(f"  -> {position}: Top {len(top_pos)} players selected")
        
        if recommendations:
            final_recommendations = pd.concat(recommendations, ignore_index=True)
            print(f"+ Generated {len(final_recommendations)} total recommendations")
            return final_recommendations
        else:
            print("X No recommendations generated")
            return None
    
    def create_draft_guide(self, recommendations_df=None):
        """
        Create realistic draft guide using optimized enhanced projections.
        Eliminates redundant scraping and calculations by using cached data.
        """
        print("\nCreating realistic draft guide based on actual draft patterns...")
        
        # Use cached enhanced projections instead of re-scraping and re-calculating
        expanded_projections = self.get_enhanced_projections(force_refresh=False)
        if expanded_projections is None:
            print("X No enhanced projections available for draft guide")
            return None
        
        print(f"+ Using enhanced projections with {len(expanded_projections)} players")
        
        # Group by position (data is already enhanced and sorted)
        all_enhanced_players = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = expanded_projections[expanded_projections['Position'] == position].copy()
            
            if len(pos_players) > 0:
                all_enhanced_players.append(pos_players)
                print(f"  -> {position}: {len(pos_players)} enhanced players ready")
        
        if not all_enhanced_players:
            print("X No enhanced players available")
            return None
        
        # sort players within each position, then apply realistic ADP logic
        all_positions = {}
        sort_col = None
        
        for pos_df in all_enhanced_players:
            if len(pos_df) > 0:
                position = pos_df['Position'].iloc[0]
                
                # figure out which fantasy points column to use for this position
                pos_sort_col = None
                for col in ['Injury_Adjusted_FPTS', 'Support_Adjusted_FPTS', 'Chemistry_Adjusted_FPTS', 'FPTS', 'Fantasy Points', 'MISC FPTS']:
                    if col in pos_df.columns:
                        pos_sort_col = col
                        break
                
                if pos_sort_col:
                    # rank players within their position by fantasy points
                    pos_df[pos_sort_col] = pd.to_numeric(pos_df[pos_sort_col], errors='coerce').fillna(0)
                    pos_df = pos_df.sort_values(pos_sort_col, ascending=False).reset_index(drop=True)
                    all_positions[position] = pos_df
                    # remember the sort column for later use
                    if sort_col is None:
                        sort_col = pos_sort_col
        
        if not sort_col or not all_positions:
            return None
        
        # here's where the magic happens - realistic draft order
        draft_guide_data = []
        realistic_draft_board = self._create_realistic_adp_order(all_positions, sort_col)
        
        # define round ranges (12-team league assumption)
        round_ranges = {
            1: (1, 12),      # Round 1
            2: (13, 24),     # Round 2  
            3: (25, 36),     # Round 3
            4: (37, 48),     # Round 4
            5: (49, 60),     # Round 5
            6: (61, 72),     # Round 6
            7: (73, 84),     # Round 7
            8: (85, 96),     # Round 8
        }
        
        for overall_rank, player_info in enumerate(realistic_draft_board, 1):
            # determine round
            draft_round = None
            for round_num, (start, end) in round_ranges.items():
                if start <= overall_rank <= end:
                    draft_round = round_num
                    break
            
            if draft_round is None or draft_round > 8:  # only include first 8 rounds
                continue
            
            player = player_info['player_data']
            position = player_info['position']
            position_rank = player_info['position_rank']
            
            # get player info
            player_name = player.get('Player', '').strip()
            
            # find the best available fantasy points column for this player
            player_fpts = 0
            for col in ['Injury_Adjusted_FPTS', 'Support_Adjusted_FPTS', 'Chemistry_Adjusted_FPTS', 'FPTS', 'Fantasy Points', 'MISC FPTS']:
                if col in player.index and pd.notna(player.get(col, None)):
                    player_fpts = player.get(col, 0)
                    break
            base_fpts = player_fpts
            
            # determine if adjusted
            is_chemistry_adjusted = 'Chemistry_Adjusted_FPTS' in player.index and player.get('Chemistry_Adjusted_FPTS', 0) != player.get('FPTS', 0)
            is_support_adjusted = 'Support_Adjusted_FPTS' in player.index and player.get('Support_Adjusted_FPTS', 0) != player.get('FPTS', 0)
            
            # create adjustment note
            adjustment_note = "Standard projection"
            if is_chemistry_adjusted:
                multiplier = player.get('Chemistry_Multiplier', 1.0)
                qb_match = player.get('Best_QB_Match', 'Unknown')
                if multiplier > 1.02:
                    adjustment_note = f"Chemistry boost with {qb_match}"
                elif multiplier < 0.98:
                    adjustment_note = f"Chemistry concern with {qb_match}"
            elif is_support_adjusted:
                multiplier = player.get('Support_Multiplier', 1.0)
                primary_rb = player.get('Primary_RB', 'Unknown')
                if multiplier > 1.02:
                    adjustment_note = f"Strong supporting cast (RB: {primary_rb})"
                elif multiplier < 0.98:
                    adjustment_note = f"Weak supporting cast (RB: {primary_rb})"
            
            # create round recommendation
            round_strategy = self._get_round_strategy(draft_round, position, overall_rank)
            
            draft_guide_data.append({
                'Overall_Rank': overall_rank,
                'Draft_Round': draft_round,
                'Player': player_name,
                'Position': position,
                'Position_Rank': f"{position}{position_rank}",
                'Projected_FPPG': round(base_fpts, 1),
                'Round_Strategy': round_strategy,
                'Adjustment_Note': adjustment_note,
                'Target_Round': f"Round {draft_round}"
            })
        
        draft_guide_df = pd.DataFrame(draft_guide_data)
        
        print(f"Created realistic draft guide with {len(draft_guide_df)} players across 8 rounds")
        print("Draft order finally looks realistic - no more QBs in the first round!")
        return draft_guide_df
    
    def _create_realistic_adp_order(self, all_positions, sort_col):
        """
        This is the secret sauce - makes draft order actually look like real drafts
        Based on years of looking at actual ADP data from various sites
        """
        realistic_order = []
        
        # how many of each position get drafted in each round (12-team leagues)
        adp_patterns = {
            # round 1: all RBs and WRs, that's it
            (1, 12): {'RB': 6, 'WR': 6, 'QB': 0, 'TE': 0},
            # round 2: still mostly RBs/WRs, maybe kelce/andrews
            (13, 24): {'RB': 5, 'WR': 6, 'QB': 0, 'TE': 1},
            # round 3: first QBs show up here (mahomes, allen, etc)
            (25, 36): {'RB': 3, 'WR': 6, 'QB': 2, 'TE': 1},
            # round 4: QB run usually starts for real here
            (37, 48): {'RB': 3, 'WR': 5, 'QB': 3, 'TE': 1},
            # round 5: more QB pickings, still some solid WRs
            (49, 60): {'RB': 2, 'WR': 5, 'QB': 4, 'TE': 1},
            # round 6: tier 2 QBs, people start grabbing TEs
            (61, 72): {'RB': 2, 'WR': 4, 'QB': 4, 'TE': 2},
            # round 7: backup RBs, late QBs
            (73, 84): {'RB': 3, 'WR': 4, 'QB': 3, 'TE': 2},
            # round 8: handcuffs and depth
            (85, 96): {'RB': 4, 'WR': 4, 'QB': 2, 'TE': 2}
        }
        
        # keep track of where we are in each position rankings
        position_counters = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}
        
        current_pick = 1
        
        for (start_pick, end_pick), quotas in adp_patterns.items():
            # grab the right number of each position for this round range
            range_players = []
            
            for position, quota in quotas.items():
                if position in all_positions:
                    pos_df = all_positions[position]
                    start_idx = position_counters[position]
                    end_idx = min(start_idx + quota, len(pos_df))
                    
                    for i in range(start_idx, end_idx):
                        if i < len(pos_df):
                            player_data = pos_df.iloc[i]
                            player_score = player_data.get(sort_col, 0)
                            
                            range_players.append({
                                'player_data': player_data,
                                'position': position,
                                'position_rank': i + 1,
                                'score': player_score,
                                'adp_adjusted_score': self._calculate_adp_adjusted_score(player_score, position, current_pick + len(range_players))
                            })
                    
                    position_counters[position] = end_idx
            
            # within each round range, sort by adjusted scores
            range_players.sort(key=lambda x: x['adp_adjusted_score'], reverse=True)
            
            # Add to the realistic order
            realistic_order.extend(range_players)
            current_pick = end_pick + 1
        
        return realistic_order
    
    def _calculate_adp_adjusted_score(self, base_score, position, pick_number):
        """
        Penalize positions that don't get drafted early (QBs, most TEs)
        Give RBs a boost since they go fast due to scarcity
        """
        # general position adjustments (RBs scarce, QBs replaceable)
        position_modifiers = {
            'RB': 1.0,   # RBs hold their value
            'WR': 0.95,  # WRs slightly behind RBs
            'QB': 0.6,   # QBs way overvalued by raw points
            'TE': 0.8    # most TEs suck, only a few worth early picks
        }
        
        # adjust more based on when they actually get picked
        if pick_number <= 24:  # first 2 rounds
            pick_modifiers = {
                'RB': 1.1,   # RB premium early
                'WR': 1.05,  # WRs good early too
                'QB': 0.3,   # nobody takes QBs this early
                'TE': 0.6    # only kelce/andrews territory
            }
        elif pick_number <= 48:  # rounds 3-4
            pick_modifiers = {
                'RB': 1.0,
                'WR': 1.0,
                'QB': 0.8,   # QBs start being reasonable here
                'TE': 0.8
            }
        else:  # rounds 5+
            pick_modifiers = {
                'RB': 0.9,   # less RB premium later
                'WR': 0.95,
                'QB': 1.0,   # QB sweet spot rounds 4-6
                'TE': 0.9
            }
        
        # multiply both adjustments together
        total_modifier = position_modifiers.get(position, 1.0) * pick_modifiers.get(position, 1.0)
        
        return base_score * total_modifier
    
    def _get_round_strategy(self, round_num, position, overall_rank):
        """
        Provide round-specific draft strategy advice for non-technical users
        """
        if round_num == 1:
            if position == 'RB':
                return "Premium RB - High floor, workload secure"
            elif position == 'WR':
                return "Elite WR - Target share locked in"
            else:
                return "Top-tier talent - Safe pick"
        
        elif round_num == 2:
            if position == 'RB':
                return "RB1 upside - Strong weekly starter"
            elif position == 'WR':
                return "WR1 potential - Reliable target"
            elif position == 'TE':
                return "Premium TE - Positional advantage"
            else:
                return "High-upside pick"
        
        elif round_num == 3:
            if position == 'QB':
                return "Top QB - Wait paid off"
            elif position == 'RB':
                return "Solid RB2 - Good value here"
            else:
                return "Strong starter - Good value"
        
        elif round_num == 4:
            if position == 'QB':
                return "Quality QB - Safe option"
            else:
                return "Flex starter - Reliable contributor"
        
        elif round_num == 5:
            return "Depth piece - Injury insurance"
        
        elif round_num == 6:
            return "Bench depth - Upside play"
        
        elif round_num == 7:
            return "Late value - Potential sleeper"
        
        elif round_num == 8:
            return "Depth/handcuff - End of starters"
        
        else:
            return "Deep sleeper pick"
    
    def _get_position_rank(self, player_name, position, all_players_df):
        """
        Calculate position rank within the draft guide
        """
        pos_players = all_players_df[all_players_df['Position'] == position]
        for i, (_, player) in enumerate(pos_players.iterrows(), 1):
            if player.get('Player', '') == player_name:
                return f"{position}{i}"
        return f"{position}?"
    
    def _apply_chemistry_adjustments(self, pos_players):
        """
        Apply QB-WR chemistry adjustments to WR/TE projections
        """
        pos_players_adjusted = pos_players.copy()
        
        # Find FPTS column
        fpts_col = None
        for col in ['FPTS', 'Fantasy Points', 'MISC FPTS']:
            if col in pos_players.columns:
                fpts_col = col
                break
        
        if not fpts_col:
            print("No fantasy points column found for chemistry adjustment")
            return pos_players
        
        # Convert to numeric
        pos_players_adjusted[fpts_col] = pd.to_numeric(pos_players_adjusted[fpts_col], errors='coerce').fillna(0)
        
        chemistry_adjustments = []
        
        for _, player in pos_players_adjusted.iterrows():
            player_name = player.get('Player', '').strip()
            
            # Extract clean player name (remove team designation)
            clean_player = player_name
            if ' ' in player_name:
                # Try to extract team abbreviation and remove it
                parts = player_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                    clean_player = ' '.join(parts[:-1])
                    team_abbr = parts[-1]
                else:
                    team_abbr = None
            
            # Get current season QB projections to find likely QB-WR matchups
            base_fpts = player[fpts_col]
            chemistry_multiplier = 1.0
            best_qb_match = None
            
            # Try to find chemistry data for this receiver
            if self.qb_wr_chemistry_data:
                best_chemistry = 0
                for chem_key, chem_data in self.qb_wr_chemistry_data.items():
                    wr_name_stored = chem_data['wr_name'].lower()
                    clean_player_lower = clean_player.lower()
                    
                    # Simple name matching
                    if (clean_player_lower in wr_name_stored or 
                        wr_name_stored in clean_player_lower or
                        any(part in wr_name_stored for part in clean_player_lower.split() if len(part) > 2)):
                        
                        if chem_data['chemistry_score'] > best_chemistry:
                            best_chemistry = chem_data['chemistry_score']
                            chemistry_multiplier = self.get_chemistry_multiplier(chem_data['qb_name'], clean_player)
                            best_qb_match = chem_data['qb_name']
            
            # Apply chemistry adjustment
            chemistry_adjusted_fpts = base_fpts * chemistry_multiplier
            
            chemistry_adjustments.append({
                'player': player_name,
                'base_fpts': base_fpts,
                'chemistry_multiplier': chemistry_multiplier,
                'adjusted_fpts': chemistry_adjusted_fpts,
                'best_qb_match': best_qb_match
            })
        
        # Add adjusted column
        pos_players_adjusted['Chemistry_Adjusted_FPTS'] = [adj['adjusted_fpts'] for adj in chemistry_adjustments]
        pos_players_adjusted['Chemistry_Multiplier'] = [adj['chemistry_multiplier'] for adj in chemistry_adjustments]
        pos_players_adjusted['Best_QB_Match'] = [adj['best_qb_match'] for adj in chemistry_adjustments]
        
        # Show some adjustment examples
        significant_adjustments = [adj for adj in chemistry_adjustments 
                                 if abs(adj['chemistry_multiplier'] - 1.0) > 0.05]
        
        if significant_adjustments:
            print(f"\nChemistry Adjustments Applied to {len(significant_adjustments)} players:")
            print("-" * 60)
            for adj in significant_adjustments[:5]:  # Show top 5
                direction = "UP" if adj['chemistry_multiplier'] > 1.0 else "DOWN"
                print(f"{adj['player']:25} {direction} {adj['base_fpts']:5.1f} -> {adj['adjusted_fpts']:5.1f} "
                      f"(x{adj['chemistry_multiplier']:.2f}) w/ {adj['best_qb_match']}")
        
        return pos_players_adjusted
    
    def _apply_qb_support_adjustments(self, pos_players):
        """
        Apply support multipliers to QB projections based on RB help and O-line protection
        """
        pos_players_adjusted = pos_players.copy()
        
        # find fantasy points column
        fpts_col = None
        for col in ['FPTS', 'Fantasy Points', 'MISC FPTS']:
            if col in pos_players.columns:
                fpts_col = col
                break
        
        if not fpts_col:
            print("No fantasy points column found for QB support adjustment")
            return pos_players
        
        # convert to numeric
        pos_players_adjusted[fpts_col] = pd.to_numeric(pos_players_adjusted[fpts_col], errors='coerce').fillna(0)
        
        support_adjustments = []
        
        for _, qb in pos_players_adjusted.iterrows():
            qb_name = qb.get('Player', '').strip()
            
            # clean QB name - remove team designation
            clean_qb = qb_name
            if ' ' in qb_name:
                parts = qb_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                    clean_qb = ' '.join(parts[:-1])
                    team_abbr = parts[-1]
                else:
                    team_abbr = None
            
            base_fpts = qb[fpts_col]
            support_multiplier = self.get_qb_support_multiplier(clean_qb)
            
            # find the best support data for this QB
            best_support_info = None
            for qb_key, qb_data in self.qb_multiplier_data.items():
                stored_qb = qb_data['qb_name'].lower()
                clean_qb_lower = clean_qb.lower()
                
                if (clean_qb_lower in stored_qb or stored_qb in clean_qb_lower):
                    if best_support_info is None or qb_data['year'] > best_support_info['year']:
                        best_support_info = qb_data
                        support_multiplier = qb_data['qb_multiplier']
            
            # apply support adjustment
            support_adjusted_fpts = base_fpts * support_multiplier
            
            support_adjustments.append({
                'qb': qb_name,
                'base_fpts': base_fpts,
                'support_multiplier': support_multiplier,
                'adjusted_fpts': support_adjusted_fpts,
                'support_info': best_support_info
            })
        
        # add adjusted columns
        pos_players_adjusted['Support_Adjusted_FPTS'] = [adj['adjusted_fpts'] for adj in support_adjustments]
        pos_players_adjusted['Support_Multiplier'] = [adj['support_multiplier'] for adj in support_adjustments]
        pos_players_adjusted['Primary_RB'] = [adj['support_info']['primary_rb'] if adj['support_info'] else 'Unknown' for adj in support_adjustments]
        
        # show significant adjustments
        significant_adjustments = [adj for adj in support_adjustments 
                                 if abs(adj['support_multiplier'] - 1.0) > 0.03]
        
        if significant_adjustments:
            print(f"\nQB Support Adjustments Applied to {len(significant_adjustments)} players:")
            print("-" * 65)
            for adj in significant_adjustments[:8]:  # show top 8
                direction = "UP" if adj['support_multiplier'] > 1.0 else "DOWN"
                support_info = adj['support_info']
                primary_rb = support_info['primary_rb'] if support_info else 'Unknown'
                print(f"{adj['qb']:20} {direction} {adj['base_fpts']:5.1f} -> {adj['adjusted_fpts']:5.1f} "
                      f"(x{adj['support_multiplier']:.3f}) w/ {primary_rb}")
        
        return pos_players_adjusted
    
    def _apply_injury_risk_adjustments(self, pos_players):
        """
        Apply injury risk adjustments to all player projections
        Players with season-ending injury history get penalized
        """
        pos_players_adjusted = pos_players.copy()
        
        # find fantasy points column
        fpts_col = None
        for col in ['Support_Adjusted_FPTS', 'Chemistry_Adjusted_FPTS', 'FPTS', 'Fantasy Points', 'MISC FPTS']:
            if col in pos_players.columns:
                fpts_col = col
                break
        
        if not fpts_col:
            print("No fantasy points column found for injury adjustment")
            return pos_players
        
        # convert to numeric
        pos_players_adjusted[fpts_col] = pd.to_numeric(pos_players_adjusted[fpts_col], errors='coerce').fillna(0)
        
        injury_adjustments = []
        
        for _, player in pos_players_adjusted.iterrows():
            player_name = player.get('Player', '').strip()
            
            # clean player name - remove team designation
            clean_player = player_name
            if ' ' in player_name:
                parts = player_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                    clean_player = ' '.join(parts[:-1])
            
            base_fpts = player[fpts_col]
            injury_multiplier = self.get_player_injury_multiplier(clean_player)
            
            # find detailed injury info for display
            injury_info = None
            for stored_name, features in self.player_injury_history.items():
                if (clean_player.lower() in stored_name.lower() or 
                    stored_name.lower() in clean_player.lower()):
                    injury_info = features
                    break
            
            # apply injury adjustment
            injury_adjusted_fpts = base_fpts * injury_multiplier
            
            injury_adjustments.append({
                'player': player_name,
                'base_fpts': base_fpts,
                'injury_multiplier': injury_multiplier,
                'adjusted_fpts': injury_adjusted_fpts,
                'injury_info': injury_info
            })
        
        # add adjusted columns
        pos_players_adjusted['Injury_Adjusted_FPTS'] = [adj['adjusted_fpts'] for adj in injury_adjustments]
        pos_players_adjusted['Injury_Multiplier'] = [adj['injury_multiplier'] for adj in injury_adjustments]
        pos_players_adjusted['Injury_Risk_Level'] = [
            'High' if adj['injury_multiplier'] < 0.9 else
            'Medium' if adj['injury_multiplier'] < 0.95 else
            'Low' for adj in injury_adjustments
        ]
        
        # show significant injury adjustments
        significant_adjustments = [adj for adj in injury_adjustments 
                                 if adj['injury_multiplier'] < 0.95]
        
        if significant_adjustments:
            print(f"\nInjury Risk Adjustments Applied to {len(significant_adjustments)} players:")
            print("-" * 70)
            for adj in significant_adjustments[:8]:  # show top 8
                direction = "DOWN" if adj['injury_multiplier'] < 1.0 else "SAME"
                injury_info = adj['injury_info']
                risk_detail = ""
                if injury_info:
                    recent = "Recent injury" if injury_info['recent_major_injury'] else "No recent injury"
                    major_count = injury_info['major_injuries_count']
                    risk_detail = f"({recent}, {major_count} major)"
                
                print(f"{adj['player']:25} {direction} {adj['base_fpts']:5.1f} -> {adj['adjusted_fpts']:5.1f} "
                      f"(x{adj['injury_multiplier']:.3f}) {risk_detail}")
        
        return pos_players_adjusted
    
    def display_draft_board(self, recommendations_df):
        """
        Display a formatted draft board
        """
        if recommendations_df is None:
            print("No recommendations available")
            return
        
        print("\n" + "="*80)
        print("NFL FANTASY DRAFT RECOMMENDATIONS")
        print("="*80)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = recommendations_df[recommendations_df['Position'] == position]
            
            if len(pos_players) > 0:
                print(f"\nTOP {position}s:")
                print("-" * 40)
                
                # Find the best fantasy points column to sort by
                fpts_cols = [col for col in pos_players.columns if 'FPTS' in col or 'Fantasy' in col]
                if fpts_cols:
                    # Sort by fantasy points in descending order (highest first)
                    sort_col = fpts_cols[0]
                    pos_players_sorted = pos_players.sort_values(sort_col, ascending=False)
                else:
                    pos_players_sorted = pos_players
                
                for idx, (_, player) in enumerate(pos_players_sorted.head(10).iterrows(), 1):
                    player_name = player['Player']
                    if fpts_cols:
                        fpts = player[fpts_cols[0]]
                        print(f"{idx:2d}. {player_name:<25} ({fpts:.1f} proj. pts)")
                    else:
                        print(f"{idx:2d}. {player_name}")
    
    def run_complete_analysis(self, use_chemistry=True, use_qb_multipliers=True, use_injury_history=True):
        """
        The full pipeline - load data, train model, scrape projections, generate rankings
        Now with QB-WR chemistry, QB support multipliers, AND season-ending injury analysis!
        """
        print("Starting NFL Fantasy Football Analysis with Advanced Multipliers + Injury History\n")
        
        # step 1: load historical data 
        print("Loading Historical Data")
        print("-" * 30)
        self.load_historical_data(list(range(2015, 2025)))  # 10 years should be enough
        
        # step 2: load injury history BEFORE training (needed for features)
        if use_injury_history:
            print(f"\nAnalyzing Player Injury History (Season-Ending Injuries)")
            print("-" * 60)
            self.load_injury_data()  # load injury data
            self.analyze_player_injury_history()  # process injury features
        
        # step 3: train model WITH injury features included
        print("\nTraining Advanced XGBoost Model with All Features")
        print("-" * 55)
        self.train_model(optimize_hyperparameters=True)
        
        # step 4: QB-WR chemistry analysis
        if use_chemistry:
            print(f"\nAnalyzing QB-WR Chemistry")
            print("-" * 35)
            self.scrape_qb_wr_connections()  # get the connection data
            self.calculate_qb_wr_chemistry()  # crunch the chemistry numbers
        
        # step 5: QB support multipliers - RB support and O-line protection
        if use_qb_multipliers:
            print(f"\nAnalyzing QB Support Systems (RB Help + O-Line Protection)")
            print("-" * 65)
            self.calculate_qb_support_multipliers()  # analyze supporting cast
        
        # step 6: generate enhanced projections and recommendations in optimized way
        print(f"\nGenerating Optimized Enhanced Projections & Recommendations")
        print("-" * 65)
        recommendations = self.generate_draft_recommendations(
            projections_df=None,  # Force use of optimized enhanced projections
            use_chemistry=use_chemistry,
            use_qb_multipliers=use_qb_multipliers,
            use_injury_history=use_injury_history
        )
        
        # step 8: show the results
        print(f"\nYour Multi-Factor Enhanced Draft Board (w/ Injury Analysis)")
        print("-" * 65)
        self.display_draft_board(recommendations)
        
        return recommendations

# run the whole thing
if __name__ == "__main__":
    predictor = NFLFantasyPredictor()
    
    # do the full analysis
    draft_recommendations = predictor.run_complete_analysis()
    
    # save results
    if draft_recommendations is not None:
        # create comprehensive draft guide CSV
        draft_guide = predictor.create_draft_guide(draft_recommendations)
        draft_guide.to_csv('fantasy_draft_guide.csv', index=False)
        print(f"\nComprehensive draft guide saved to 'fantasy_draft_guide.csv'")
        print(f"\nThese are PPR projections with QB-WR chemistry and QB support adjustments - adjust for your league scoring.")
        
        # save chemistry data for reference
        if predictor.qb_wr_chemistry_data:
            chemistry_df = pd.DataFrame([
                {
                    'QB': data['qb_name'],
                    'WR': data['wr_name'],
                    'Chemistry_Score': data['chemistry_score'],
                    'Total_Targets': data['total_targets'],
                    'Catch_Rate': data['avg_catch_rate'],
                    'Years_Together': len(set(data['years_together']))
                }
                for data in predictor.qb_wr_chemistry_data.values()
            ])
            chemistry_df.to_csv('qb_wr_chemistry_scores.csv', index=False)
            print(f"QB-WR chemistry data saved to 'qb_wr_chemistry_scores.csv'")
        
        # save QB support multiplier data too
        if predictor.qb_multiplier_data:
            qb_support_df = pd.DataFrame([
                {
                    'QB': data['qb_name'],
                    'Team': data['team'],
                    'Year': data['year'],
                    'Support_Multiplier': data['qb_multiplier'],
                    'Primary_RB': data['primary_rb'],
                    'RB_YPG': data['rb_ypg'],
                    'RB_Support_Score': data['rb_support_score'],
                    'OLine_Support_Score': data['oline_support_score']
                }
                for data in predictor.qb_multiplier_data.values()
            ])
            qb_support_df.to_csv('qb_support_multipliers.csv', index=False)
            print(f"QB support multiplier data saved to 'qb_support_multipliers.csv'")
    
    print(f"\nBooyah. Good luck drafting with your multi-factor edge, friends. \n - Kevin Veeder")

# bonus functions for nerds who want to dig deeper
def compare_players(predictor, player1_stats, player2_stats, player1_name="Player 1", player2_name="Player 2"):
    """
    Head to head player comparison - now with chemistry awareness for WRs
    """
    pred1 = predictor.predict_fantasy_points(player1_stats)
    pred2 = predictor.predict_fantasy_points(player2_stats)
    
    print(f"{player1_name} projected FPPG: {pred1:.2f}")
    print(f"{player2_name} projected FPPG: {pred2:.2f}")
    
    if pred1 > pred2:
        print(f"{player1_name} is projected to score {pred1-pred2:.2f} more points per game")
    elif pred2 > pred1:
        print(f"{player2_name} is projected to score {pred2-pred1:.2f} more points per game")
    else:
        print("Both players have similar projections")

def analyze_qb_wr_chemistry(predictor, qb_name, wr_name):
    """
    Get detailed chemistry analysis for a specific QB-WR pair
    """
    if not predictor.qb_wr_chemistry_data:
        print("No chemistry data loaded. Run the analysis first.")
        return
    
    multiplier = predictor.get_chemistry_multiplier(qb_name, wr_name)
    
    # find the exact match if it exists
    key = f"{qb_name}_{wr_name}"
    if key in predictor.qb_wr_chemistry_data:
        chem = predictor.qb_wr_chemistry_data[key]
        print(f"\n{qb_name} -> {wr_name} Chemistry Report:")
        print("-" * 50)
        print(f"Chemistry Score: {chem['chemistry_score']:.3f}")
        print(f"Fantasy Multiplier: {multiplier:.2f}x")
        print(f"Years Together: {len(set(chem['years_together']))}")
        print(f"Total Targets: {chem['total_targets']}")
        print(f"Catch Rate: {chem['avg_catch_rate']:.1%}")
        print(f"Targets/Game: {chem['targets_per_game']:.1f}")
        print(f"TDs/Target: {chem['tds_per_target']:.1%}")
    else:
        print(f"No specific chemistry data found for {qb_name} -> {wr_name}")
        print(f"Using default multiplier: {multiplier:.2f}x")

def analyze_qb_support_system(predictor, qb_name):
    """
    Get detailed support analysis for a specific QB - RB help and O-line protection
    """
    if not predictor.qb_multiplier_data:
        print("No QB support data loaded. Run the analysis first.")
        return
    
    # find matches for this QB
    qb_support_data = []
    for qb_key, qb_data in predictor.qb_multiplier_data.items():
        stored_qb = qb_data['qb_name'].lower()
        qb_name_lower = qb_name.lower()
        
        if qb_name_lower in stored_qb or stored_qb in qb_name_lower:
            qb_support_data.append(qb_data)
    
    if not qb_support_data:
        print(f"No support data found for {qb_name}")
        return
    
    # show the most recent data first
    qb_support_data.sort(key=lambda x: x['year'], reverse=True)
    
    print(f"\n{qb_name} Support System Analysis:")
    print("=" * 60)
    
    for data in qb_support_data[:3]:  # show last 3 years max
        print(f"\n{data['team']} {data['year']}:")
        print("-" * 30)
        print(f"Support Multiplier: {data['qb_multiplier']:.3f}x")
        print(f"Primary RB: {data['primary_rb']} ({data['rb_ypg']:.1f} ypg)")
        print(f"RB Support Score: {data['rb_support_score']:.2f}/1.0")
        print(f"O-Line Support Score: {data['oline_support_score']:.2f}/1.0")
        
        # interpretation
        if data['qb_multiplier'] >= 1.05:
            print("🟢 Excellent supporting cast")
        elif data['qb_multiplier'] >= 0.98:
            print("🟡 Average supporting cast")  
        else:
            print("🔴 Poor supporting cast")

def analyze_position_depth(projections_df, position, threshold=10.0):
    """
    See how deep the talent goes at each position - helps with draft strategy
    """
    if projections_df is None:
        return
    
    pos_players = projections_df[projections_df['Position'] == position]
    
    if len(pos_players) == 0:
        print(f"No {position} players found in projections")
        return
    
    # how many guys are projected above the threshold?
    fpts_cols = [col for col in pos_players.columns if 'FPTS' in col or 'Fantasy' in col]
    if fpts_cols:
        high_value = pos_players[pos_players[fpts_cols[0]] >= threshold]
        print(f"{position} players projected for {threshold}+ points: {len(high_value)}")
        print(f"Position depth score: {len(high_value)/len(pos_players)*100:.1f}%")