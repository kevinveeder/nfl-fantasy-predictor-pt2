#!/usr/bin/env python3
"""
Analyze how well our model predicts 2024 fantasy performance
Compare model predictions vs actual 2024 results for top players
"""

import pandas as pd
import numpy as np
from nfl_fantasy_predictor import NFLFantasyPredictor
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_2024_performance():
    print("="*70)
    print("2024 FANTASY SEASON - MODEL PREDICTION ANALYSIS")
    print("="*70)
    print("Analyzing how accurately our model predicted 2024 fantasy performance")
    print()
    
    # Initialize and train our model
    print("1. Training model with full dataset...")
    predictor = NFLFantasyPredictor()
    
    # Load full dataset
    predictor.load_historical_data(list(range(2020, 2025)))
    predictor.load_injury_data(list(range(2020, 2025)))
    predictor.analyze_player_injury_history()
    
    # Train model
    model = predictor.train_model(optimize_hyperparameters=False)
    
    if not model:
        print("ERROR: Model training failed!")
        return
    
    # Get the training data and split by year
    print("2. Extracting 2024 season data...")
    training_data = predictor.historical_data.copy()
    
    # Focus on 2024 season
    data_2024 = training_data[training_data['Year'] == 2024].copy()
    
    if len(data_2024) == 0:
        print("ERROR: No 2024 data found!")
        return
    
    print(f"Found {len(data_2024)} players from 2024 season")
    
    # Get model predictions for all data
    print("3. Generating model predictions...")
    X, y = predictor.prepare_training_data()
    
    if X is None or y is None:
        print("ERROR: Could not prepare training data!")
        return
    
    # Get predictions for all data
    y_pred = predictor.model.predict(X)
    
    # Map predictions back to the historical data
    predictions_df = pd.DataFrame({
        'Player': training_data['Player'].iloc[:len(y_pred)],
        'Year': training_data['Year'].iloc[:len(y_pred)],
        'Position': training_data.get('FantPos', training_data.get('Pos', 'UNK')).iloc[:len(y_pred)],
        'Team': training_data.get('Tm', 'UNK').iloc[:len(y_pred)],
        'Games_Played': training_data['G'].iloc[:len(y_pred)],
        'Actual_FPPG': y,
        'Predicted_FPPG': y_pred,
        'Total_Fantasy_Points': training_data.get('FantPt', 0).iloc[:len(y_pred)]
    })
    
    # Filter to 2024 and players with significant playing time
    results_2024 = predictions_df[
        (predictions_df['Year'] == 2024) & 
        (predictions_df['Games_Played'] >= 8) &
        (predictions_df['Actual_FPPG'] > 0)
    ].copy()
    
    print(f"4. Analyzing {len(results_2024)} 2024 players (8+ games played)...")
    
    # Calculate prediction errors
    results_2024['Prediction_Error'] = results_2024['Predicted_FPPG'] - results_2024['Actual_FPPG']
    results_2024['Abs_Error'] = abs(results_2024['Prediction_Error'])
    results_2024['Error_Percentage'] = (results_2024['Abs_Error'] / results_2024['Actual_FPPG']) * 100
    
    # Analyze results
    analyze_prediction_accuracy(results_2024)
    
    return results_2024

def analyze_prediction_accuracy(results_df):
    """
    Comprehensive analysis of 2024 prediction accuracy
    """
    print("\n" + "="*70)
    print("2024 SEASON PREDICTION ACCURACY ANALYSIS")
    print("="*70)
    
    # Overall accuracy metrics
    overall_mae = mean_absolute_error(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    overall_r2 = r2_score(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    correlation = results_df['Actual_FPPG'].corr(results_df['Predicted_FPPG'])
    
    print(f"\nOVERALL MODEL PERFORMANCE ON 2024 SEASON:")
    print(f"Mean Absolute Error (MAE): {overall_mae:.3f} fantasy points per game")
    print(f"R² Score: {overall_r2:.3f}")
    print(f"Correlation: {correlation:.3f}")
    print(f"Players analyzed: {len(results_df)}")
    
    # Accuracy breakdown
    within_1 = len(results_df[results_df['Abs_Error'] <= 1.0])
    within_2 = len(results_df[results_df['Abs_Error'] <= 2.0])
    within_3 = len(results_df[results_df['Abs_Error'] <= 3.0])
    
    print(f"\nPREDICTION ACCURACY BREAKDOWN:")
    print(f"Within ±1.0 FPPG: {within_1}/{len(results_df)} ({within_1/len(results_df)*100:.1f}%)")
    print(f"Within ±2.0 FPPG: {within_2}/{len(results_df)} ({within_2/len(results_df)*100:.1f}%)")
    print(f"Within ±3.0 FPPG: {within_3}/{len(results_df)} ({within_3/len(results_df)*100:.1f}%)")
    
    # Position-specific analysis
    print(f"\nPOSITION-SPECIFIC ACCURACY:")
    print("-" * 55)
    print(f"{'Pos':<3} {'Players':<7} {'MAE':<6} {'R²':<6} {'Corr':<6} {'±1.0':<6} {'±2.0'}")
    print("-" * 55)
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = results_df[results_df['Position'] == position]
        if len(pos_data) >= 3:
            pos_mae = mean_absolute_error(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG'])
            pos_r2 = r2_score(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG']) if len(pos_data) > 1 else 0
            pos_corr = pos_data['Actual_FPPG'].corr(pos_data['Predicted_FPPG']) if len(pos_data) > 1 else 0
            pos_within_1 = len(pos_data[pos_data['Abs_Error'] <= 1.0])
            pos_within_2 = len(pos_data[pos_data['Abs_Error'] <= 2.0])
            
            print(f"{position:<3} {len(pos_data):<7} {pos_mae:<6.2f} {pos_r2:<6.2f} {pos_corr:<6.2f} "
                  f"{pos_within_1}/{len(pos_data):<4} {pos_within_2}/{len(pos_data)}")
    
    # Top 20 Fantasy Performers Analysis
    print(f"\nTOP 20 FANTASY PERFORMERS - PREDICTION vs ACTUAL:")
    print("-" * 85)
    print(f"{'Rank':<4} {'Player':<22} {'Pos':<3} {'Team':<3} {'Actual':<6} {'Predicted':<8} {'Error':<6} {'%Err'}")
    print("-" * 85)
    
    top_20_actual = results_df.nlargest(20, 'Actual_FPPG')
    
    for i, (_, player) in enumerate(top_20_actual.iterrows(), 1):
        actual = player['Actual_FPPG']
        predicted = player['Predicted_FPPG']
        error = player['Prediction_Error']
        error_pct = player['Error_Percentage']
        
        # Clean player name
        player_name = player['Player']
        if len(player_name) > 21:
            player_name = player_name[:21]
        
        print(f"{i:<4} {player_name:<22} {player['Position']:<3} {str(player['Team'])[:3]:<3} "
              f"{actual:<6.1f} {predicted:<8.1f} {error:>+6.1f} {error_pct:>5.1f}%")
    
    # Best Predictions (smallest errors among top performers)
    print(f"\nBEST PREDICTIONS (smallest errors among top 50 performers):")
    print("-" * 85)
    print(f"{'Player':<22} {'Pos':<3} {'Team':<3} {'Actual':<6} {'Predicted':<8} {'Error':<6} {'%Err'}")
    print("-" * 85)
    
    top_50 = results_df.nlargest(50, 'Actual_FPPG')
    best_predictions = top_50.nsmallest(10, 'Abs_Error')
    
    for _, player in best_predictions.iterrows():
        actual = player['Actual_FPPG']
        predicted = player['Predicted_FPPG']
        error = player['Prediction_Error']
        error_pct = player['Error_Percentage']
        
        player_name = player['Player']
        if len(player_name) > 21:
            player_name = player_name[:21]
        
        print(f"{player_name:<22} {player['Position']:<3} {str(player['Team'])[:3]:<3} "
              f"{actual:<6.1f} {predicted:<8.1f} {error:>+6.1f} {error_pct:>5.1f}%")
    
    # Biggest Misses (largest errors among top performers)
    print(f"\nBIGGEST PREDICTION ERRORS (among top 50 performers):")
    print("-" * 85)
    print(f"{'Player':<22} {'Pos':<3} {'Team':<3} {'Actual':<6} {'Predicted':<8} {'Error':<6} {'%Err'}")
    print("-" * 85)
    
    biggest_errors = top_50.nlargest(10, 'Abs_Error')
    
    for _, player in biggest_errors.iterrows():
        actual = player['Actual_FPPG']
        predicted = player['Predicted_FPPG']
        error = player['Prediction_Error']
        error_pct = player['Error_Percentage']
        
        player_name = player['Player']
        if len(player_name) > 21:
            player_name = player_name[:21]
        
        print(f"{player_name:<22} {player['Position']:<3} {str(player['Team'])[:3]:<3} "
              f"{actual:<6.1f} {predicted:<8.1f} {error:>+6.1f} {error_pct:>5.1f}%")
    
    # Summary
    print(f"\nSUMMARY INSIGHTS:")
    print("-" * 40)
    
    median_error = results_df['Abs_Error'].median()
    mean_error_pct = results_df['Error_Percentage'].mean()
    
    print(f"Median absolute error: {median_error:.2f} FPPG")
    print(f"Average error percentage: {mean_error_pct:.1f}%")
    
    # Check if model tends to over or under predict
    over_predictions = len(results_df[results_df['Prediction_Error'] > 0])
    under_predictions = len(results_df[results_df['Prediction_Error'] < 0])
    
    print(f"Over-predictions: {over_predictions} ({over_predictions/len(results_df)*100:.1f}%)")
    print(f"Under-predictions: {under_predictions} ({under_predictions/len(results_df)*100:.1f}%)")
    
    # Performance on different tiers of players
    print(f"\nACCURACY BY PLAYER TIER:")
    print("-" * 30)
    
    # Divide players into tiers based on actual performance
    results_sorted = results_df.sort_values('Actual_FPPG', ascending=False)
    tier_size = len(results_sorted) // 4
    
    tiers = {
        'Elite (Top 25%)': results_sorted.iloc[:tier_size],
        'Good (25-50%)': results_sorted.iloc[tier_size:2*tier_size],
        'Average (50-75%)': results_sorted.iloc[2*tier_size:3*tier_size],
        'Below Avg (75-100%)': results_sorted.iloc[3*tier_size:]
    }
    
    for tier_name, tier_data in tiers.items():
        if len(tier_data) > 0:
            tier_mae = mean_absolute_error(tier_data['Actual_FPPG'], tier_data['Predicted_FPPG'])
            tier_r2 = r2_score(tier_data['Actual_FPPG'], tier_data['Predicted_FPPG']) if len(tier_data) > 1 else 0
            print(f"{tier_name}: MAE = {tier_mae:.2f}, R² = {tier_r2:.3f} ({len(tier_data)} players)")

if __name__ == "__main__":
    try:
        results = analyze_2024_performance()
        if results is not None:
            print(f"\n[SUCCESS] 2024 season analysis completed!")
            
            # Save results
            results.to_csv('2024_prediction_analysis.csv', index=False)
            print(f"Detailed results saved to: 2024_prediction_analysis.csv")
        else:
            print(f"\n[ERROR] Analysis failed!")
    except Exception as e:
        print(f"\n[ERROR] Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()