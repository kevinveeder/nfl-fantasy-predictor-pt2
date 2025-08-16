#!/usr/bin/env python3
"""
Simple analysis of how our current model performs on 2024 data
Compare actual vs predicted for recent performance
"""

import pandas as pd
import numpy as np
from nfl_fantasy_predictor import NFLFantasyPredictor
from sklearn.metrics import mean_absolute_error, r2_score

def quick_2024_analysis():
    print("="*70)
    print("QUICK 2024 FANTASY PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Load recent data for model training
    predictor = NFLFantasyPredictor()
    predictor.load_historical_data([2023, 2024])  # Just recent years for speed
    
    # Get the raw 2024 data to see actual performance
    actual_2024_data = predictor.historical_data[predictor.historical_data['Year'] == 2024].copy()
    
    if len(actual_2024_data) == 0:
        print("No 2024 data found!")
        return
    
    print(f"Found {len(actual_2024_data)} players from 2024 season")
    
    # Filter to players with significant playing time
    significant_players = actual_2024_data[actual_2024_data['G'] >= 8].copy()
    print(f"Analyzing {len(significant_players)} players with 8+ games")
    
    # Calculate actual FPPG
    significant_players['Actual_FPPG'] = significant_players.get('FantPt', 0) / significant_players['G'].replace(0, 1)
    
    # Sort by actual fantasy performance
    significant_players = significant_players.sort_values('Actual_FPPG', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 25 FANTASY PERFORMERS - 2024 SEASON")
    print("="*70)
    print(f"{'Rank':<4} {'Player':<25} {'Pos':<3} {'G':<2} {'FPPG':<6} {'Total':<6} {'Key Stats'}")
    print("-" * 70)
    
    top_25 = significant_players.head(25)
    
    for i, (_, player) in enumerate(top_25.iterrows(), 1):
        # Get key stats based on position
        pos = player.get('FantPos', player.get('Pos', 'UNK'))
        games = int(player['G'])
        fppg = player['Actual_FPPG']
        total_pts = player.get('FantPt', 0)
        
        # Position-specific key stats
        if pos == 'QB':
            pass_yds = player.get('Yds.1', 0) 
            pass_tds = player.get('TD.1', 0)
            key_stats = f"{int(pass_yds)} pass yds, {int(pass_tds)} pass TDs"
        elif pos == 'RB':
            rush_yds = player.get('Yds', 0)
            rush_tds = player.get('TD', 0)
            rec_yds = player.get('Yds.1', 0)
            key_stats = f"{int(rush_yds)} rush yds, {int(rush_tds)} rush TDs, {int(rec_yds)} rec yds"
        elif pos in ['WR', 'TE']:
            rec = player.get('Rec', 0)
            rec_yds = player.get('Yds.1', 0)
            rec_tds = player.get('TD.1', 0)
            key_stats = f"{int(rec)} rec, {int(rec_yds)} yds, {int(rec_tds)} TDs"
        else:
            key_stats = "N/A"
        
        # Truncate long names/stats
        player_name = player['Player'][:24]
        if len(key_stats) > 35:
            key_stats = key_stats[:32] + "..."
        
        print(f"{i:<4} {player_name:<25} {pos:<3} {games:<2} {fppg:<6.1f} {total_pts:<6.0f} {key_stats}")
    
    # Position breakdown
    print(f"\nPOSITION BREAKDOWN - TOP PERFORMERS:")
    print("-" * 40)
    
    pos_stats = {}
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_players = top_25[top_25.get('FantPos', top_25.get('Pos')) == pos]
        if len(pos_players) > 0:
            avg_fppg = pos_players['Actual_FPPG'].mean()
            max_fppg = pos_players['Actual_FPPG'].max()
            pos_stats[pos] = {
                'count': len(pos_players),
                'avg_fppg': avg_fppg,
                'max_fppg': max_fppg,
                'top_player': pos_players.iloc[0]['Player']
            }
            print(f"{pos}: {len(pos_players)} in top 25, avg {avg_fppg:.1f} FPPG, best: {pos_players.iloc[0]['Player'][:20]} ({max_fppg:.1f})")
    
    # Injury analysis for top performers
    print(f"\nINJURY ANALYSIS - TOP 25 PERFORMERS:")
    print("-" * 50)
    
    games_analysis = {
        '16-17 games (healthy)': len(top_25[top_25['G'] >= 16]),
        '13-15 games (mostly healthy)': len(top_25[(top_25['G'] >= 13) & (top_25['G'] < 16)]),
        '10-12 games (some missed time)': len(top_25[(top_25['G'] >= 10) & (top_25['G'] < 13)]),
        '8-9 games (significant missed time)': len(top_25[(top_25['G'] >= 8) & (top_25['G'] < 10)])
    }
    
    for category, count in games_analysis.items():
        print(f"{category}: {count} players")
    
    # Fantasy points distribution
    print(f"\nFANTASY POINTS DISTRIBUTION (Top 25):")
    print("-" * 40)
    
    fppg_ranges = {
        '20+ FPPG (Elite)': len(top_25[top_25['Actual_FPPG'] >= 20]),
        '15-19.9 FPPG (Excellent)': len(top_25[(top_25['Actual_FPPG'] >= 15) & (top_25['Actual_FPPG'] < 20)]),
        '12-14.9 FPPG (Very Good)': len(top_25[(top_25['Actual_FPPG'] >= 12) & (top_25['Actual_FPPG'] < 15)]),
        '10-11.9 FPPG (Good)': len(top_25[(top_25['Actual_FPPG'] >= 10) & (top_25['Actual_FPPG'] < 12)]),
        'Under 10 FPPG': len(top_25[top_25['Actual_FPPG'] < 10])
    }
    
    for category, count in fppg_ranges.items():
        print(f"{category}: {count} players")
    
    # Show some notable performances
    print(f"\nNOTABLE 2024 PERFORMANCES:")
    print("-" * 50)
    
    # Elite QB performances
    qb_top = significant_players[significant_players.get('FantPos', significant_players.get('Pos')) == 'QB'].head(3)
    print("Top QBs:")
    for _, qb in qb_top.iterrows():
        print(f"  {qb['Player']}: {qb['Actual_FPPG']:.1f} FPPG ({int(qb['G'])} games)")
    
    # Elite RB performances  
    rb_top = significant_players[significant_players.get('FantPos', significant_players.get('Pos')) == 'RB'].head(3)
    print("Top RBs:")
    for _, rb in rb_top.iterrows():
        print(f"  {rb['Player']}: {rb['Actual_FPPG']:.1f} FPPG ({int(rb['G'])} games)")
    
    # Elite WR performances
    wr_top = significant_players[significant_players.get('FantPos', significant_players.get('Pos')) == 'WR'].head(3)
    print("Top WRs:")
    for _, wr in wr_top.iterrows():
        print(f"  {wr['Player']}: {wr['Actual_FPPG']:.1f} FPPG ({int(wr['G'])} games)")
    
    # Elite TE performances
    te_top = significant_players[significant_players.get('FantPos', significant_players.get('Pos')) == 'TE'].head(2)
    print("Top TEs:")
    for _, te in te_top.iterrows():
        print(f"  {te['Player']}: {te['Actual_FPPG']:.1f} FPPG ({int(te['G'])} games)")
    
    print(f"\n[SUCCESS] 2024 season analysis completed!")
    print(f"This shows the actual top performers our model would need to predict accurately.")
    
    # Save results for reference
    top_25.to_csv('2024_top_performers.csv', index=False)
    print(f"Top 25 performers saved to: 2024_top_performers.csv")
    
    return top_25

if __name__ == "__main__":
    try:
        results = quick_2024_analysis()
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()