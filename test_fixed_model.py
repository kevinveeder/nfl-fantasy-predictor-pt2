#!/usr/bin/env python3
"""
Test the fixed model without data leakage
"""

from nfl_fantasy_predictor import NFLFantasyPredictor

def test_fixed_model():
    print("Testing fixed model performance (without data leakage)...")
    
    predictor = NFLFantasyPredictor()
    
    # Load data and run the full analysis
    print("\n1. Loading historical data...")
    predictor.load_historical_data([2022, 2023, 2024])  # Use 3 years for better training
    
    print("\n2. Loading injury data...")
    predictor.load_injury_data([2022, 2023, 2024])
    predictor.analyze_player_injury_history()
    
    print("\n3. Training model...")
    model = predictor.train_model(optimize_hyperparameters=False)  # Skip optimization for speed
    
    if model:
        print("\n[SUCCESS] Model training successful!")
        print("The performance metrics should now be realistic (not 0.997 R²)")
    else:
        print("\n[ERROR] Model training failed")
        
    # Check features to make sure Fantasy_Points_Consistency is not included
    X, y = predictor.prepare_training_data()
    if X is not None:
        print(f"\n4. Feature verification:")
        print(f"   Total features: {len(X.columns)}")
        if 'Fantasy_Points_Consistency' in X.columns:
            print("   [ERROR] Fantasy_Points_Consistency still present (BAD)")
        else:
            print("   [SUCCESS] Fantasy_Points_Consistency successfully removed")
        
        print(f"\n   Current features: {list(X.columns)}")

if __name__ == "__main__":
    test_fixed_model()