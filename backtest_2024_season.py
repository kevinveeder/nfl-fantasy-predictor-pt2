#!/usr/bin/env python3
"""
Backtest the model by training on 2015-2023 data and predicting 2024 results
Compare predictions vs actual 2024 fantasy performance
"""

import pandas as pd
import numpy as np
from nfl_fantasy_predictor import NFLFantasyPredictor
from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib.pyplot as plt  # Not needed for this analysis
# import seaborn as sns  # Not needed for this analysis

def create_backtest():
    print("="*70)
    print("2024 NFL FANTASY SEASON BACKTEST")
    print("="*70)
    print("Training on 2015-2023 data, predicting 2024 performance")
    print()
    
    predictor = NFLFantasyPredictor()
    
    # Step 1: Load training data (2015-2023, excluding 2024)
    print("1. Loading historical training data (2015-2023)...")
    training_years = list(range(2015, 2024))  # 2015-2023
    predictor.load_historical_data(training_years)
    
    # Step 2: Load injury data for training years
    print("2. Loading injury data for training years...")
    injury_years = list(range(2020, 2024))  # 2020-2023 for injury patterns
    predictor.load_injury_data(injury_years)
    predictor.analyze_player_injury_history()
    
    # Step 3: Load chemistry and support data
    print("3. Loading QB-WR chemistry and support data...")
    predictor.scrape_qb_wr_connections(injury_years)
    predictor.calculate_qb_wr_chemistry()
    predictor.calculate_qb_support_multipliers()
    
    # Step 4: Train model on historical data
    print("4. Training model on 2015-2023 data...")
    model = predictor.train_model(optimize_hyperparameters=False)  # Skip optimization for speed
    
    if not model:
        print("ERROR: Model training failed!")
        return
    
    # Step 5: Load actual 2024 data for comparison
    print("5. Loading actual 2024 season data...")
    actual_2024_predictor = NFLFantasyPredictor()
    actual_2024_predictor.load_historical_data([2024])
    
    if actual_2024_predictor.historical_data is None:
        print("ERROR: Could not load 2024 data!")
        return
    
    actual_2024_data = actual_2024_predictor.historical_data.copy()
    
    # Step 6: Generate predictions for 2024 players
    print("6. Generating predictions for 2024 players...")
    predictions = []
    
    for _, player_row in actual_2024_data.iterrows():
        try:
            # Create feature vector for this player (using 2023 stats to predict 2024)
            player_features = prepare_prediction_features(player_row, predictor)
            if player_features is not None:
                predicted_fppg = predictor.model.predict([player_features])[0]
                
                predictions.append({
                    'Player': player_row['Player'],
                    'Position': player_row.get('FantPos', player_row.get('Pos', 'UNK')),
                    'Predicted_FPPG': predicted_fppg,
                    'Actual_FPPG': player_row.get('FantPt', 0) / max(1, player_row.get('G', 1)),
                    'Games_Played': player_row.get('G', 0),
                    'Total_Fantasy_Points': player_row.get('FantPt', 0)
                })
        except Exception as e:
            # Skip players we can't predict for
            continue
    
    if not predictions:
        print("ERROR: No predictions generated!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)
    results_df = results_df[results_df['Games_Played'] >= 8]  # Filter to players who played most of season
    
    print(f"7. Analyzing predictions for {len(results_df)} players (8+ games)...")
    
    # Step 7: Analyze results by position
    analyze_backtest_results(results_df)
    
    return results_df

def prepare_prediction_features(player_row, predictor):
    """
    Prepare feature vector for a 2024 player using the same features the model was trained on
    """
    try:
        # Create a temporary dataframe with this player
        temp_df = pd.DataFrame([player_row])
        
        # Engineer features using the same process
        temp_df = predictor.engineer_features(temp_df)
        
        # Extract only the features the model expects
        if predictor.features and all(f in temp_df.columns for f in predictor.features):
            feature_vector = temp_df[predictor.features].iloc[0].values
            
            # Scale features
            feature_vector_scaled = predictor.scaler.transform([feature_vector])[0]
            return feature_vector_scaled
        
    except Exception as e:
        pass  # Skip problematic players
    
    return None

def analyze_backtest_results(results_df):
    """
    Analyze how well the model predicted 2024 performance
    """
    print("\n" + "="*70)
    print("BACKTEST RESULTS ANALYSIS")
    print("="*70)
    
    # Overall accuracy
    overall_mae = mean_absolute_error(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    overall_r2 = r2_score(results_df['Actual_FPPG'], results_df['Predicted_FPPG'])
    
    print(f"\nOVERALL MODEL ACCURACY:")
    print(f"MAE: {overall_mae:.3f} fantasy points per game")
    print(f"R²: {overall_r2:.3f}")
    print(f"Players analyzed: {len(results_df)}")
    
    # Position-specific analysis
    print(f"\nPOSITION-SPECIFIC ACCURACY:")
    print("-" * 50)
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = results_df[results_df['Position'] == position]
        if len(pos_data) > 5:  # Need decent sample size
            pos_mae = mean_absolute_error(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG'])
            pos_r2 = r2_score(pos_data['Actual_FPPG'], pos_data['Predicted_FPPG'])
            print(f"{position:3s}: MAE = {pos_mae:.3f}, R² = {pos_r2:.3f} ({len(pos_data)} players)")
    
    # Top performers analysis
    print(f"\nTOP 20 ACTUAL PERFORMERS vs PREDICTIONS:")
    print("-" * 80)
    
    # Sort by actual performance
    top_actual = results_df.nlargest(20, 'Actual_FPPG')
    
    print(f"{'Rank':<4} {'Player':<25} {'Pos':<3} {'Actual':<6} {'Predicted':<9} {'Diff':<6} {'Error'}")
    print("-" * 80)
    
    for i, (_, player) in enumerate(top_actual.iterrows(), 1):
        actual = player['Actual_FPPG']
        predicted = player['Predicted_FPPG']
        diff = predicted - actual
        error_pct = abs(diff) / actual * 100 if actual > 0 else 0
        
        print(f"{i:<4} {player['Player'][:24]:<25} {player['Position']:<3} "
              f"{actual:<6.1f} {predicted:<9.1f} {diff:>+6.1f} {error_pct:>5.1f}%")
    
    # Biggest prediction errors
    print(f"\nBIGGEST PREDICTION ERRORS (over-predictions):")
    print("-" * 80)
    
    results_df['Prediction_Error'] = results_df['Predicted_FPPG'] - results_df['Actual_FPPG']
    biggest_errors = results_df.nlargest(10, 'Prediction_Error')
    
    print(f"{'Player':<25} {'Pos':<3} {'Actual':<6} {'Predicted':<9} {'Error'}")
    print("-" * 60)
    
    for _, player in biggest_errors.iterrows():
        print(f"{player['Player'][:24]:<25} {player['Position']:<3} "
              f"{player['Actual_FPPG']:<6.1f} {player['Predicted_FPPG']:<9.1f} "
              f"{player['Prediction_Error']:>+6.1f}")
    
    # Best predictions
    print(f"\nBEST PREDICTIONS (smallest errors):")
    print("-" * 80)
    
    results_df['Abs_Error'] = abs(results_df['Prediction_Error'])
    best_predictions = results_df.nsmallest(10, 'Abs_Error')
    
    print(f"{'Player':<25} {'Pos':<3} {'Actual':<6} {'Predicted':<9} {'Error'}")
    print("-" * 60)
    
    for _, player in best_predictions.iterrows():
        print(f"{player['Player'][:24]:<25} {player['Position']:<3} "
              f"{player['Actual_FPPG']:<6.1f} {player['Predicted_FPPG']:<9.1f} "
              f"{player['Prediction_Error']:>+6.1f}")
    
    # Summary insights
    print(f"\nKEY INSIGHTS:")
    print("-" * 30)
    
    overestimated = len(results_df[results_df['Prediction_Error'] > 1.0])
    underestimated = len(results_df[results_df['Prediction_Error'] < -1.0])
    accurate = len(results_df[abs(results_df['Prediction_Error']) <= 1.0])
    
    print(f"Predictions within ±1.0 FPPG: {accurate}/{len(results_df)} ({accurate/len(results_df)*100:.1f}%)")
    print(f"Over-predicted by >1.0 FPPG: {overestimated} players")
    print(f"Under-predicted by >1.0 FPPG: {underestimated} players")
    
    avg_error = results_df['Abs_Error'].mean()
    print(f"Average absolute error: {avg_error:.2f} FPPG")
    
    # Save results
    results_df.to_csv('backtest_2024_results.csv', index=False)
    print(f"\nResults saved to: backtest_2024_results.csv")

if __name__ == "__main__":
    try:
        results = create_backtest()
        if results is not None:
            print(f"\n[SUCCESS] Backtest completed!")
        else:
            print(f"\n[ERROR] Backtest failed!")
    except Exception as e:
        print(f"\n[ERROR] Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()