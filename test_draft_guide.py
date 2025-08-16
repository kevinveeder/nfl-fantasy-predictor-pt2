#!/usr/bin/env python3
"""
Test script to reproduce the Support_Adjusted_FPTS KeyError
"""

import sys
import pandas as pd
import numpy as np

def test_draft_guide_creation():
    print("Testing draft guide creation...")
    
    from nfl_fantasy_predictor import NFLFantasyPredictor
    
    predictor = NFLFantasyPredictor()
    
    try:
        # Load a small amount of data for testing
        print("1. Loading historical data...")
        historical_data = predictor.load_historical_data([2024])
        print(f"   Loaded {len(historical_data)} player-seasons")
        
        print("2. Loading injury data...")
        injury_data = predictor.load_injury_data([2024])
        predictor.analyze_player_injury_history()
        print(f"   Processed injury data")
        
        print("3. Loading chemistry data...")
        connections = predictor.scrape_qb_wr_connections([2024])
        chemistry = predictor.calculate_qb_wr_chemistry()
        print(f"   Calculated chemistry for {len(chemistry) if chemistry else 0} pairs")
        
        print("4. Loading QB support data...")
        support = predictor.calculate_qb_support_multipliers()
        print(f"   Calculated support for {len(support) if support else 0} QBs")
        
        print("5. Scraping current projections...")
        projections = predictor.scrape_all_positions()
        print(f"   Scraped {len(projections) if projections is not None else 0} players")
        
        print("6. Generating recommendations...")
        recommendations = predictor.generate_draft_recommendations(
            projections, 
            use_chemistry=True,
            use_qb_multipliers=True,
            use_injury_history=True
        )
        print(f"   Generated recommendations for {len(recommendations) if recommendations is not None else 0} players")
        
        print("7. Creating draft guide...")
        draft_guide = predictor.create_draft_guide(recommendations)
        if draft_guide is not None:
            print(f"   [SUCCESS] Created draft guide with {len(draft_guide)} players")
        else:
            print("   [ERROR] Draft guide creation failed")
            
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_draft_guide_creation()
    if success:
        print("\n[SUCCESS] Draft guide creation test completed!")
    else:
        print("\n[ERROR] Draft guide creation test failed!")