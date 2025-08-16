#!/usr/bin/env python3
"""
Test script to demonstrate model caching optimization
Shows the dramatic speed improvement from model caching
"""

import time
from nfl_fantasy_predictor import NFLFantasyPredictor

def test_model_caching():
    """Test the model caching functionality"""
    print("=" * 60)
    print("MODEL CACHING OPTIMIZATION TEST")
    print("=" * 60)
    
    # First run - should train the model
    print("\n1. FIRST RUN (Training Model)")
    print("-" * 40)
    predictor1 = NFLFantasyPredictor()
    
    # Load some minimal data for testing
    predictor1.load_historical_data([2023, 2024])  # Just 2 years for speed
    
    start_time = time.time()
    
    # Train model (this will cache it)
    model = predictor1.train_model(optimize_hyperparameters=False)  # Skip optimization for speed
    
    first_run_time = time.time() - start_time
    print(f"First run completed in {first_run_time:.1f} seconds")
    
    # Second run - should load from cache
    print("\n2. SECOND RUN (Loading Cached Model)")
    print("-" * 40)
    predictor2 = NFLFantasyPredictor()
    
    # Load the same data
    predictor2.load_historical_data([2023, 2024])
    
    start_time = time.time()
    
    # This should load from cache
    model = predictor2.train_model(optimize_hyperparameters=False)
    
    second_run_time = time.time() - start_time
    print(f"Second run completed in {second_run_time:.1f} seconds")
    
    # Calculate speedup
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"\n" + "=" * 60)
        print("RESULTS:")
        print(f"First run (training):  {first_run_time:.1f} seconds")
        print(f"Second run (cached):   {second_run_time:.1f} seconds")
        print(f"SPEEDUP: {speedup:.1f}x faster!")
        print("=" * 60)
        
        if speedup > 10:
            print("+ EXCELLENT: Model caching provides significant speedup!")
        elif speedup > 3:
            print("+ GOOD: Model caching provides measurable speedup!")
        else:
            print("? Model caching working but speedup is minimal")
    else:
        print("+ INSTANT: Second run was so fast it couldn't be measured!")

if __name__ == "__main__":
    test_model_caching()