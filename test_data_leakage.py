#!/usr/bin/env python3
"""
Test script to identify and fix data leakage in the model
"""

from nfl_fantasy_predictor import NFLFantasyPredictor
import pandas as pd
import numpy as np

def test_data_leakage():
    print("Testing for data leakage in the fantasy predictor...")
    
    predictor = NFLFantasyPredictor()
    
    # Load minimal data for testing
    predictor.load_historical_data([2023, 2024])
    
    # Prepare training data
    X, y = predictor.prepare_training_data()
    
    if X is None or y is None:
        print("Error: Could not prepare training data")
        return
    
    print(f"\nTarget variable (FPPG) statistics:")
    print(f"Mean: {y.mean():.3f}")
    print(f"Std: {y.std():.3f}")
    print(f"Min: {y.min():.3f}")
    print(f"Max: {y.max():.3f}")
    
    print(f"\nFeatures that might contain data leakage:")
    suspicious_features = []
    
    for feature in X.columns:
        # Check correlation with target
        correlation = X[feature].corr(y)
        if abs(correlation) > 0.95:  # Very high correlation
            suspicious_features.append((feature, correlation))
            print(f"  {feature}: correlation = {correlation:.6f}")
    
    if suspicious_features:
        print(f"\n[ALERT] FOUND {len(suspicious_features)} SUSPICIOUS FEATURES!")
        
        # Check if Fantasy_Points_Consistency is essentially the same as target
        if 'Fantasy_Points_Consistency' in X.columns:
            consistency_values = X['Fantasy_Points_Consistency']
            target_values = y
            
            print(f"\nFantasy_Points_Consistency vs FPPG comparison:")
            print(f"Are they identical? {np.allclose(consistency_values, target_values)}")
            print(f"Mean difference: {abs(consistency_values - target_values).mean():.10f}")
            print(f"Max difference: {abs(consistency_values - target_values).max():.10f}")
            
            if np.allclose(consistency_values, target_values):
                print("[CONFIRMED] Fantasy_Points_Consistency is identical to target variable!")
                print("This is severe data leakage - the model is essentially predicting FPPG using FPPG")
    
    else:
        print("[OK] No obvious data leakage detected")
    
    return suspicious_features

def test_model_without_leakage():
    print("\n" + "="*60)
    print("TESTING MODEL PERFORMANCE WITHOUT DATA LEAKAGE")
    print("="*60)
    
    predictor = NFLFantasyPredictor()
    predictor.load_historical_data([2023, 2024])
    
    # Get training data
    X, y = predictor.prepare_training_data()
    
    # Remove the leaky feature
    if 'Fantasy_Points_Consistency' in X.columns:
        print("Removing Fantasy_Points_Consistency feature...")
        X_clean = X.drop(columns=['Fantasy_Points_Consistency'])
        
        # Update the predictor's features list
        predictor.features = [f for f in predictor.features if f != 'Fantasy_Points_Consistency']
        
        # Train model with clean data
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        import xgboost as xgb
        
        # Scale and split
        X_scaled = predictor.scaler.fit_transform(X_clean)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train simple model
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nMODEL PERFORMANCE WITHOUT DATA LEAKAGE:")
        print(f"MAE: {mae:.3f} fantasy points")
        print(f"R²: {r2:.3f}")
        
        print(f"\nThis is much more realistic performance!")
        print(f"The original R² of 0.997 was due to data leakage.")
        
        return mae, r2
    
    return None, None

if __name__ == "__main__":
    suspicious = test_data_leakage()
    
    if suspicious:
        test_model_without_leakage()
    else:
        print("No data leakage detected - model performance might be legitimate")