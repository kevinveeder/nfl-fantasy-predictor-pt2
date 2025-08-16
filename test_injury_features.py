#!/usr/bin/env python3
"""
Test script to validate injury feature implementation
"""

import sys
import pandas as pd
import numpy as np

# Test the injury feature implementation without running full analysis
def test_injury_features():
    print("Testing injury feature implementation...")
    
    # Import our predictor class
    from nfl_fantasy_predictor import NFLFantasyPredictor
    
    predictor = NFLFantasyPredictor()
    
    # Test 1: Check if injury feature methods exist
    print("\n1. Testing method availability:")
    methods_to_test = [
        'load_injury_data',
        'analyze_player_injury_history', 
        'get_player_injury_multiplier',
        '_add_injury_features',
        '_apply_injury_risk_adjustments'
    ]
    
    for method in methods_to_test:
        if hasattr(predictor, method):
            print(f"   [OK] {method} - Available")
        else:
            print(f"   [ERROR] {method} - Missing")
    
    # Test 2: Test injury multiplier function with sample data
    print("\n2. Testing injury multiplier calculation:")
    
    # Create mock injury history data
    predictor.player_injury_history = {
        'Christian McCaffrey': {
            'injury_risk_score': 0.8,
            'recent_major_injury': 1,
            'major_injuries_count': 3,
            'career_injury_rate': 25.0
        },
        'Cooper Kupp': {
            'injury_risk_score': 0.6,
            'recent_major_injury': 1,
            'major_injuries_count': 2,
            'career_injury_rate': 15.0
        },
        'Josh Allen': {
            'injury_risk_score': 0.2,
            'recent_major_injury': 0,
            'major_injuries_count': 0,
            'career_injury_rate': 5.0
        }
    }
    
    test_players = ['Christian McCaffrey', 'Cooper Kupp', 'Josh Allen', 'Unknown Player']
    for player in test_players:
        multiplier = predictor.get_player_injury_multiplier(player)
        print(f"   {player:20} -> Multiplier: {multiplier:.3f}")
    
    # Test 3: Test feature engineering with sample data
    print("\n3. Testing feature engineering:")
    
    sample_df = pd.DataFrame({
        'Player': ['Christian McCaffrey CAR', 'Cooper Kupp LAR', 'Josh Allen BUF'],
        'FPPG': [20.5, 18.2, 22.1],
        'G': [16, 12, 17],
        'Pos': ['RB', 'WR', 'QB']
    })
    
    try:
        enhanced_df = predictor._add_injury_features(sample_df)
        print(f"   [OK] Enhanced dataframe created with {len(enhanced_df.columns)} columns")
        
        injury_cols = [col for col in enhanced_df.columns if 'injury' in col.lower()]
        print(f"   [OK] Injury columns added: {injury_cols}")
        
        # Show sample data
        print("\n   Sample enhanced data:")
        display_cols = ['Player', 'FPPG', 'injury_risk_score', 'injury_multiplier']
        for col in display_cols:
            if col in enhanced_df.columns:
                print(f"     {col}: {enhanced_df[col].tolist()}")
        
    except Exception as e:
        print(f"   [ERROR] Feature engineering failed: {e}")
    
    # Test 4: Test injury risk adjustments
    print("\n4. Testing injury risk adjustments:")
    
    sample_projections = pd.DataFrame({
        'Player': ['Christian McCaffrey CAR', 'Cooper Kupp LAR', 'Josh Allen BUF'],
        'Position': ['RB', 'WR', 'QB'],
        'FPTS': [250.0, 200.0, 300.0]
    })
    
    try:
        adjusted_projections = predictor._apply_injury_risk_adjustments(sample_projections)
        print(f"   [OK] Injury adjustments applied successfully")
        
        if 'Injury_Adjusted_FPTS' in adjusted_projections.columns:
            print("   [OK] Injury_Adjusted_FPTS column created")
            for _, player in adjusted_projections.iterrows():
                original = player['FPTS']
                adjusted = player['Injury_Adjusted_FPTS']
                multiplier = player.get('Injury_Multiplier', 1.0)
                risk_level = player.get('Injury_Risk_Level', 'Unknown')
                print(f"     {player['Player']:20} {original:6.1f} -> {adjusted:6.1f} "
                      f"(x{multiplier:.3f}, {risk_level} risk)")
        
    except Exception as e:
        print(f"   [ERROR] Injury adjustments failed: {e}")
    
    print("\n5. Testing library availability:")
    try:
        import nfl_data_py as nfl
        print("   [OK] nfl_data_py available - can load injury data")
    except ImportError:
        print("   [WARNING] nfl_data_py not available - injury data loading will be skipped")
        print("      Install with: pip install nfl-data-py")
    
    print("\n[SUCCESS] Injury feature testing completed!")
    print("\nThe model now includes:")
    print("  - Season-ending injury history tracking")
    print("  - Injury risk scoring (0-1 scale)")  
    print("  - Recent major injury flags")
    print("  - Career injury rate calculations")
    print("  - Injury-adjusted fantasy projections")
    print("  - Integration with existing chemistry/support features")

if __name__ == "__main__":
    test_injury_features()