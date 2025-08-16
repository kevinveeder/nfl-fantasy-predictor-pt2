"""
Vectorized feature engineering for maximum performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from performance_utils import timer

logger = logging.getLogger(__name__)

class VectorizedFeatureEngine:
    """High-performance vectorized feature engineering"""
    
    @staticmethod
    @timer("Vectorized feature engineering")
    def engineer_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized feature engineering - replaces slow loops with numpy operations
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure numeric columns are actually numeric (vectorized)
        numeric_columns = ['FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1', 'Yds', 'TD', 'Int', 
                          'Yds.1', 'TD.1', 'Yds.2', 'TD.2', 'FL', 'Fmb']
        
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Vectorized per-game calculations (much faster than loops)
        games_played = np.maximum(df['G'], 1)  # Avoid division by zero
        
        # Basic per-game stats
        df['FPPG'] = df['FantPt'] / games_played
        df['Attempts_Per_Game'] = df.get('Att', 0) / games_played
        df['Targets_Per_Game'] = df.get('Tgt', 0) / games_played
        df['Receptions_Per_Game'] = df.get('Rec', 0) / games_played
        
        # Vectorized total calculations
        df['Total_Yards'] = df.get('Yds', 0) + df.get('Yds.1', 0) + df.get('Yds.2', 0)
        df['Total_TDs'] = df.get('TD', 0) + df.get('TD.1', 0) + df.get('TD.2', 0)
        df['Total_Yards_Per_Game'] = df['Total_Yards'] / games_played
        df['Total_TDs_Per_Game'] = df['Total_TDs'] / games_played
        
        # Vectorized efficiency calculations with safe division
        df['Yards_Per_Carry'] = np.where(
            df.get('Att', 0) > 0,
            df.get('Yds', 0) / df.get('Att', 0),
            0
        )
        
        df['Yards_Per_Target'] = np.where(
            df.get('Tgt', 0) > 0,
            df.get('Yds.1', 0) / df.get('Tgt', 0),
            0
        )
        
        df['Catch_Rate'] = np.where(
            df.get('Tgt', 0) > 0,
            df.get('Rec', 0) / df.get('Tgt', 0),
            0
        )
        
        # Vectorized TD rates
        df['Rush_TD_Per_Game'] = df.get('TD', 0) / games_played
        df['Rec_TD_Per_Game'] = df.get('TD.1', 0) / games_played
        
        df['Rush_TD_Rate'] = np.where(
            df.get('Att', 0) > 0,
            df.get('TD', 0) / df.get('Att', 0),
            0
        )
        
        df['Rec_TD_Rate'] = np.where(
            df.get('Rec', 0) > 0,
            df.get('TD.1', 0) / df.get('Rec', 0),
            0
        )
        
        # Vectorized position encoding (much faster than loops)
        position_dummies = pd.get_dummies(df.get('Pos', ''), prefix='Pos', dummy_na=False)
        df = pd.concat([df, position_dummies], axis=1)
        
        logger.info(f"Engineered features for {len(df)} players in vectorized operations")
        return df
    
    @staticmethod
    @timer("Vectorized chemistry adjustments")
    def apply_chemistry_adjustments_vectorized(df: pd.DataFrame, 
                                            chemistry_data: Dict) -> pd.DataFrame:
        """
        Vectorized QB-WR chemistry adjustments using pandas operations
        """
        if df.empty or not chemistry_data:
            return df
        
        df = df.copy()
        
        # Create chemistry lookup Series for vectorized operations
        chemistry_series = pd.Series(chemistry_data)
        
        # Extract player names and clean them for matching
        player_names = df['Player'].str.strip().str.upper()
        
        # Vectorized chemistry score lookup
        df['Chemistry_Score'] = player_names.map(chemistry_series).fillna(1.0)
        
        # Vectorized multiplier calculation
        df['Chemistry_Multiplier'] = np.clip(df['Chemistry_Score'], 0.9, 1.2)
        
        # Apply adjustments to fantasy points columns
        fantasy_cols = [col for col in df.columns if 'FPTS' in col or 'Fantasy' in col]
        for col in fantasy_cols:
            df[f'Chemistry_Adjusted_{col}'] = df[col] * df['Chemistry_Multiplier']
        
        logger.info(f"Applied chemistry adjustments to {len(df)} players")
        return df
    
    @staticmethod
    @timer("Vectorized injury adjustments") 
    def apply_injury_adjustments_vectorized(df: pd.DataFrame, 
                                          injury_data: Dict) -> pd.DataFrame:
        """
        Vectorized injury risk adjustments
        """
        if df.empty or not injury_data:
            return df
        
        df = df.copy()
        
        # Create injury lookup Series
        injury_series = pd.Series(injury_data)
        
        # Vectorized injury risk lookup
        player_names = df['Player'].str.strip().str.upper()
        df['Injury_Risk'] = player_names.map(injury_series).fillna(0.0)
        
        # Vectorized multiplier calculation (higher risk = lower multiplier)
        df['Injury_Multiplier'] = np.clip(1.0 - (df['Injury_Risk'] * 0.25), 0.75, 1.0)
        
        # Apply to fantasy points
        fantasy_cols = [col for col in df.columns if 'FPTS' in col or 'Fantasy' in col]
        for col in fantasy_cols:
            df[f'Injury_Adjusted_{col}'] = df[col] * df['Injury_Multiplier']
        
        logger.info(f"Applied injury adjustments to {len(df)} players")
        return df
    
    @staticmethod
    @timer("Batch data processing")
    def process_data_batch(data_chunks: List[pd.DataFrame], 
                          feature_functions: List[callable]) -> pd.DataFrame:
        """
        Process data in chunks to optimize memory usage
        """
        processed_chunks = []
        
        for i, chunk in enumerate(data_chunks):
            logger.info(f"Processing chunk {i+1}/{len(data_chunks)} ({len(chunk)} rows)")
            
            # Apply all feature functions to chunk
            processed_chunk = chunk.copy()
            for func in feature_functions:
                processed_chunk = func(processed_chunk)
            
            processed_chunks.append(processed_chunk)
        
        # Concatenate all processed chunks
        result = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Batch processing complete: {len(result)} total rows")
        
        return result

class DataProcessor:
    """Memory-efficient data processing pipeline"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.feature_engine = VectorizedFeatureEngine()
    
    @timer("Data chunking and processing")
    def process_large_dataset(self, df: pd.DataFrame, 
                            chemistry_data: Dict = None,
                            injury_data: Dict = None) -> pd.DataFrame:
        """
        Process large datasets in memory-efficient chunks
        """
        if len(df) <= self.chunk_size:
            # Small dataset, process normally
            return self._process_single_chunk(df, chemistry_data, injury_data)
        
        # Split into chunks
        chunks = [df[i:i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        logger.info(f"Processing {len(chunks)} chunks of ~{self.chunk_size} rows each")
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            processed_chunk = self._process_single_chunk(chunk, chemistry_data, injury_data)
            processed_chunks.append(processed_chunk)
        
        # Combine results
        result = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Chunk processing complete: {len(result)} total rows")
        
        return result
    
    def _process_single_chunk(self, df: pd.DataFrame, 
                            chemistry_data: Dict = None,
                            injury_data: Dict = None) -> pd.DataFrame:
        """Process a single chunk with all transformations"""
        
        # Basic feature engineering
        df = self.feature_engine.engineer_features_vectorized(df)
        
        # Chemistry adjustments if data available
        if chemistry_data:
            df = self.feature_engine.apply_chemistry_adjustments_vectorized(df, chemistry_data)
        
        # Injury adjustments if data available  
        if injury_data:
            df = self.feature_engine.apply_injury_adjustments_vectorized(df, injury_data)
        
        return df

def create_feature_pipeline() -> List[callable]:
    """Create optimized feature engineering pipeline"""
    engine = VectorizedFeatureEngine()
    
    return [
        engine.engineer_features_vectorized,
        # Add more feature functions as needed
    ]

# Utility functions for common operations
@timer("Dataframe memory optimization")
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types
    """
    df = df.copy()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns to categories where beneficial
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df

@timer("Data validation vectorized")
def validate_data_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized data validation and cleaning
    """
    if df.empty:
        return df
    
    # Remove rows with all-NaN values (vectorized)
    df = df.dropna(how='all')
    
    # Remove duplicate rows (if any)
    df = df.drop_duplicates()
    
    # Vectorized outlier detection for fantasy points
    if 'FantPt' in df.columns:
        q1 = df['FantPt'].quantile(0.01)
        q99 = df['FantPt'].quantile(0.99)
        df = df[(df['FantPt'] >= q1) & (df['FantPt'] <= q99)]
    
    # Fill remaining NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df