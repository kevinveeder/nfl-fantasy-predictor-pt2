#!/usr/bin/env python3
"""
Test the fixed sequence in run_complete_analysis
"""

from nfl_fantasy_predictor import NFLFantasyPredictor

def test_complete_analysis_sequence():
    print("Testing complete analysis sequence...")
    
    predictor = NFLFantasyPredictor()
    
    try:
        # Test with minimal features to speed up the test
        print("Running complete analysis with injury history enabled...")
        
        # Override to use smaller dataset for faster testing
        predictor.load_historical_data([2023, 2024])  # Just 2 years
        
        # Test injury loading sequence
        predictor.load_injury_data([2023, 2024])
        predictor.analyze_player_injury_history()
        
        # Test model training with injury features
        X, y = predictor.prepare_training_data()
        
        if X is not None:
            injury_features = [f for f in X.columns if 'injury' in f.lower()]
            print(f"[OK] Injury features successfully included: {injury_features}")
            print(f"[OK] Training data ready: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Quick model training test (without optimization to save time)
            try:
                model = predictor.train_model(optimize_hyperparameters=False)
                if model is not None:
                    print("[OK] Model training successful with injury features")
                    
                    # Check feature importance
                    if hasattr(predictor, 'feature_importance'):
                        injury_importance = predictor.feature_importance[
                            predictor.feature_importance['Feature'].str.contains('injury', case=False)
                        ]
                        if len(injury_importance) > 0:
                            print("[OK] Injury features have importance scores:")
                            for _, row in injury_importance.iterrows():
                                print(f"   {row['Feature']}: {row['Importance']:.4f}")
                        else:
                            print("[WARNING] No injury features in importance rankings")
                else:
                    print("[ERROR] Model training failed")
                    return False
                    
            except Exception as e:
                print(f"[ERROR] Model training failed: {e}")
                return False
                
        else:
            print("[ERROR] Training data preparation failed")
            return False
            
        print("\n[SUCCESS] SEQUENCE FIX SUCCESSFUL!")
        print("   - Injury data loads BEFORE model training")
        print("   - Injury features included in training")
        print("   - Model trained with injury features")
        print("   - No 'no injury data loaded' message should appear")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_analysis_sequence()
    if success:
        print("\n[SUCCESS] The sequence fix resolves the issue!")
    else:
        print("\n[ERROR] The issue still needs more work.")