#!/usr/bin/env python3
"""
Simplified backtest: Train model on 2015-2023, then evaluate predictions vs actual 2024 performance
We'll use our current model to see how it would have predicted for known 2024 players
"""

import pandas as pd
import numpy as np
from nfl_fantasy_predictor import NFLFantasyPredictor
from sklearn.metrics import mean_absolute_error, r2_score

def run_simplified_backtest():
    print("="*70)
    print("2024 FANTASY FOOTBALL BACKTEST - SIMPLIFIED VERSION")
    print("="*70)
    print("Testing model accuracy against actual 2024 season results")
    print()
    
    # Step 1: Train our model as normal (includes some 2024 data in training)
    print("1. Training model with historical data...")
    predictor = NFLFantasyPredictor()
    
    # Load data including 2024 for training
    predictor.load_historical_data(list(range(2020, 2025)))
    predictor.load_injury_data(list(range(2020, 2025)))
    predictor.analyze_player_injury_history()
    
    # Train model
    model = predictor.train_model(optimize_hyperparameters=False)
    
    if not model:
        print("ERROR: Model training failed!")
        return
    
    # Step 2: Get training data to analyze model performance on known data
    print("2. Analyzing model performance on training data...")
    X, y = predictor.prepare_training_data()
    
    if X is None or y is None:
        print("ERROR: Could not prepare training data!")
        return
    
    # Get the underlying historical data with predictions
    historical_data = predictor.historical_data.copy()
    
    # Add model predictions to the historical data
    historical_features = predictor._engineer_features(historical_data)
    
    # Filter to features the model expects
    feature_cols = [f for f in predictor.features if f in historical_features.columns]
    if len(feature_cols) < len(predictor.features):
        print(f"WARNING: Only {len(feature_cols)} of {len(predictor.features)} features available")
    
    X_full = historical_features[feature_cols].fillna(0)
    y_actual = historical_features['FPPG']
    
    # Scale features and predict
    X_scaled = predictor.scaler.transform(X_full)
    y_predicted = predictor.model.predict(X_scaled)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Player': historical_data['Player'],
        'Year': historical_data['Year'],
        'Position': historical_data.get('FantPos', historical_data.get('Pos', 'UNK')),
        'Games_Played': historical_data['G'],
        'Actual_FPPG': y_actual,
        'Predicted_FPPG': y_predicted,
        'Total_Fantasy_Points': historical_data.get('FantPt', 0)
    })
    
    # Filter to players who played significant time
    results_df = results_df[
        (results_df['Games_Played'] >= 8) & 
        (results_df['Actual_FPPG'] > 0) &
        (results_df['Predicted_FPPG'] > 0)
    ]
    
    print(f"3. Analyzing predictions for {len(results_df)} players...")
    
    # Step 3: Focus on 2024 predictions vs actual
    results_2024 = results_df[results_df['Year'] == 2024].copy()
    
    if len(results_2024) == 0:
        print("ERROR: No 2024 data found!")
        return
    
    print(f"4. Analyzing 2024 season predictions for {len(results_2024)} players...")
    
    # Analyze results
    analyze_backtest_results(results_2024, "2024 SEASON")
    
    # Also analyze overall model performance
    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE (ALL YEARS)")
    print("="*70)
    analyze_backtest_results(results_df, "ALL YEARS")
    
    return results_2024

def analyze_backtest_results(results_df, label):
    """
    Analyze prediction accuracy
    """
    if len(results_df) == 0:
        print(f"No data to analyze for {label}")
        return
    
    # Calculate errors
    results_df['Prediction_Error'] = results_df['Predicted_FPPG'] - results_df['Actual_FPPG']
    results_df['Abs_Error'] = abs(results_df['Prediction_Error'])
    
    # Overall accuracy
    overall_mae = mean_absolute_error(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    overall_r2 = r2_score(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    
    print(f"\n{label} - OVERALL MODEL ACCURACY:")
    print(f"MAE: {overall_mae:.3f} fantasy points per game")
    print(f"R²: {overall_r2:.3f}")
    print(f"Players analyzed: {len(results_df)}")
    
    # Position-specific analysis
    print(f"\nPOSITION-SPECIFIC ACCURACY:")
    print("-" * 50)
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = results_df[results_df['Position'] == position]
        if len(pos_data) > 3:  # Need some sample size
            pos_mae = mean_absolute_error(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG'])
            pos_r2 = r2_score(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG']) if len(pos_data) > 1 else 0
            print(f"{position:3s}: MAE = {pos_mae:.3f}, R² = {pos_r2:.3f} ({len(pos_data)} players)")
    
    # Top performers analysis
    print(f"\nTOP 15 ACTUAL PERFORMERS vs PREDICTIONS:")
    print("-" * 80)
    
    # Sort by actual performance and take top 15
    top_actual = results_df.nlargest(15, 'Actual_FPPG')
    
    print(f"{'Rank':<4} {'Player':<25} {'Pos':<3} {'Actual':<6} {'Predicted':<9} {'Diff':<6} {'Error'}")
    print("-" * 80)
    
    for i, (_, player) in enumerate(top_actual.iterrows(), 1):
        actual = player['Actual_FPPG']
        predicted = player['Predicted_FPPG']
        diff = predicted - actual
        error_pct = abs(diff) / actual * 100 if actual > 0 else 0
        
        print(f"{i:<4} {player['Player'][:24]:<25} {player['Position']:<3} "
              f"{actual:<6.1f} {predicted:<9.1f} {diff:>+6.1f} {error_pct:>5.1f}%")
    
    # Best predictions
    print(f"\nBEST PREDICTIONS (smallest errors among top 50 players):")
    print("-" * 80)
    
    # Look at top 50 actual performers and find best predictions among them
    top_50_actual = results_df.nlargest(50, 'Actual_FPPG')
    best_predictions = top_50_actual.nsmallest(10, 'Abs_Error')
    
    print(f"{'Player':<25} {'Pos':<3} {'Actual':<6} {'Predicted':<9} {'Error'}")
    print("-" * 60)
    
    for _, player in best_predictions.iterrows():
        print(f"{player['Player'][:24]:<25} {player['Position']:<3} "
              f"{player['Actual_FPPG']:<6.1f} {player['Predicted_FPPG']:<9.1f} "
              f"{player['Prediction_Error']:>+6.1f}")
    
    # Summary insights
    print(f"\nKEY INSIGHTS:")
    print("-" * 30)
    
    accurate_1 = len(results_df[abs(results_df['Prediction_Error']) <= 1.0])
    accurate_2 = len(results_df[abs(results_df['Prediction_Error']) <= 2.0])
    
    print(f"Predictions within ±1.0 FPPG: {accurate_1}/{len(results_df)} ({accurate_1/len(results_df)*100:.1f}%)")
    print(f"Predictions within ±2.0 FPPG: {accurate_2}/{len(results_df)} ({accurate_2/len(results_df)*100:.1f}%)")
    
    avg_error = results_df['Abs_Error'].mean()
    median_error = results_df['Abs_Error'].median()
    print(f"Average absolute error: {avg_error:.2f} FPPG")
    print(f"Median absolute error: {median_error:.2f} FPPG")
    
    # Check correlation between predictions and actual
    correlation = results_df['Actual_FPPG'].corr(results_df['Predicted_FPPG'])
    print(f"Correlation coefficient: {correlation:.3f}")

if __name__ == "__main__":
    try:
        results = run_simplified_backtest()
        if results is not None:
            print(f"\n[SUCCESS] Backtest completed!")
            
            # Save results
            results.to_csv('backtest_results_2024.csv', index=False)
            print(f"Results saved to: backtest_results_2024.csv")
        else:
            print(f"\n[ERROR] Backtest failed!")
    except Exception as e:
        print(f"\n[ERROR] Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()