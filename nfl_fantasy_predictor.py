"""
NFL Fantasy Football Draft Predictor
====================================
A comprehensive tool for scraping NFL data, training predictive models,
and generating draft recommendations based on historical performance and projections.

Author: Kevin Veeder
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from bs4 import BeautifulSoup
import lxml
import time
import warnings
warnings.filterwarnings('ignore')

class NFLFantasyPredictor:
    def __init__(self):
        self.model = None
        self.features = ['Att', 'Tgt', 'Rec', 'Cmp', 'Att.1']
        self.historical_data = None
        self.projections_data = {}
        
    def load_historical_data(self, years=[2022, 2023, 2024]):
        """
        Load and prepare historical fantasy data from Pro Football Reference for multiple years
        """
        print(f"Loading fantasy football data for years: {years}")
        all_data = []
        
        for year in years:
            print(f"  Loading {year} data...")
            
            # URL for Pro Football Reference fantasy data
            url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
            
            try:
                # Read the HTML table
                df = pd.read_html(url, header=1)[0]
                
                # Clean up the data
                df = df[df['Rk'] != 'Rk']  # Remove repeated header rows
                df = df.fillna(0)  # Fill missing values
                
                # Convert relevant columns to numeric
                numeric_columns = ['FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate Fantasy Points Per Game (FPPG)
                df['FPPG'] = df['FantPt'] / df['G']
                
                # Filter out players with 0 games (they don't help in training)
                df = df[df['G'] > 0]
                
                # Add year column for reference
                df['Year'] = year
                
                all_data.append(df)
                print(f"    Successfully loaded {len(df)} players from {year}")
                
                # Add small delay between requests to be respectful
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error loading {year} data: {e}")
                continue
        
        if all_data:
            # Combine all years into one dataset
            self.historical_data = pd.concat(all_data, ignore_index=True)
            total_players = len(self.historical_data)
            print(f"\nCombined dataset: {total_players} total player-seasons")
            print(f"Years included: {sorted(self.historical_data['Year'].unique())}")
            return self.historical_data
        else:
            print("No historical data successfully loaded")
            return None
    
    def prepare_training_data(self):
        """
        Prepare features and target variables for model training
        """
        if self.historical_data is None:
            print("No historical data loaded. Please run load_historical_data() first.")
            return None, None
        
        df = self.historical_data.copy()
        
        # Ensure all feature columns exist and are numeric
        for feature in self.features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
            else:
                print(f"Warning: Feature '{feature}' not found in data")
                self.features.remove(feature)
        
        # Prepare features (X) and target (y)
        X = df[self.features]
        y = df['FPPG']
        
        print(f"Training data prepared with {len(X)} samples and {len(self.features)} features")
        return X, y
    
    def train_model(self):
        """
        Train the linear regression model
        """
        X, y = self.prepare_training_data()
        if X is None:
            return None
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("Model Training Complete!")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Display feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.model
    
    def scrape_fantasy_projections(self, position='rb'):
        """
        Scrape player projections from FantasyPros
        """
        print(f"Scraping {position.upper()} projections from FantasyPros...")
        
        url = f"https://www.fantasypros.com/nfl/projections/{position}.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            dfs = pd.read_html(response.text, header=[0, 1])
            df = dfs[0]
            
            # Flatten multi-level column headers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns]
            
            # Find and rename player column
            player_cols = [col for col in df.columns if 'Player' in col]
            if player_cols:
                df = df.rename(columns={player_cols[0]: 'Player'})
            
            # Add position information
            df['Position'] = position.upper()
            
            # Clean player names (remove team info, defense/kicker designations)
            df = df[~df['Player'].str.contains("Defense|Kicker", na=False)]
            
            self.projections_data[position] = df
            print(f"Successfully scraped {len(df)} {position.upper()} players")
            
            # Add a small delay to be respectful to the server
            time.sleep(1)
            
            return df
            
        except Exception as e:
            print(f"Error scraping {position} projections: {e}")
            return None
    
    def scrape_all_positions(self):
        """
        Scrape projections for all relevant fantasy positions
        """
        positions = ['qb', 'rb', 'wr', 'te']
        all_projections = []
        
        for position in positions:
            df = self.scrape_fantasy_projections(position)
            if df is not None:
                all_projections.append(df)
        
        if all_projections:
            combined_df = pd.concat(all_projections, ignore_index=True)
            print(f"\nTotal players scraped: {len(combined_df)}")
            return combined_df
        else:
            return None
    
    def predict_fantasy_points(self, player_stats):
        """
        Predict fantasy points for a player given their stats
        """
        if self.model is None:
            print("Model not trained yet. Please run train_model() first.")
            return None
        
        # Ensure player_stats has all required features
        stats_array = []
        for feature in self.features:
            stats_array.append(player_stats.get(feature, 0))
        
        prediction = self.model.predict([stats_array])[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def generate_draft_recommendations(self, projections_df, top_n=20):
        """
        Generate draft recommendations based on projected fantasy points
        """
        if projections_df is None:
            print("No projections data available")
            return None
        
        # This is a simplified version - you might want to add more sophisticated
        # analysis like value over replacement, positional scarcity, etc.
        
        recommendations = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = projections_df[projections_df['Position'] == position].copy()
            
            if len(pos_players) > 0:
                # Sort by a key metric (you might want to customize this)
                if 'FPTS' in pos_players.columns:
                    pos_players = pos_players.sort_values('FPTS', ascending=False)
                elif 'Fantasy Points' in pos_players.columns:
                    pos_players = pos_players.sort_values('Fantasy Points', ascending=False)
                
                top_pos = pos_players.head(min(top_n//4, len(pos_players)))
                recommendations.append(top_pos)
        
        if recommendations:
            final_recommendations = pd.concat(recommendations, ignore_index=True)
            return final_recommendations
        else:
            return None
    
    def display_draft_board(self, recommendations_df):
        """
        Display a formatted draft board
        """
        if recommendations_df is None:
            print("No recommendations available")
            return
        
        print("\n" + "="*80)
        print("NFL FANTASY DRAFT RECOMMENDATIONS ")
        print("="*80)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = recommendations_df[recommendations_df['Position'] == position]
            
            if len(pos_players) > 0:
                print(f"\n TOP {position}s:")
                print("-" * 40)
                
                for idx, (_, player) in enumerate(pos_players.head(10).iterrows(), 1):
                    player_name = player['Player']
                    # Try to find a fantasy points column
                    fpts_cols = [col for col in pos_players.columns if 'FPTS' in col or 'Fantasy' in col]
                    if fpts_cols:
                        fpts = player[fpts_cols[0]]
                        print(f"{idx:2d}. {player_name:<25} ({fpts:.1f} proj. pts)")
                    else:
                        print(f"{idx:2d}. {player_name}")
    
    def run_complete_analysis(self):
        """
        Run the complete fantasy football analysis pipeline
        """
        print(" Starting NFL Fantasy Football Analysis \n")
        
        # Step 1: Load historical data and train model
        print("STEP 1: Loading Historical Data and Training Model")
        print("-" * 50)
        self.load_historical_data([2022, 2023, 2024])
        self.train_model()
        
        # Step 2: Scrape current projections
        print(f"\nSTEP 2: Scraping Current Projections")
        print("-" * 50)
        projections = self.scrape_all_positions()
        
        # Step 3: Generate recommendations
        print(f"\nSTEP 3: Generating Draft Recommendations")
        print("-" * 50)
        recommendations = self.generate_draft_recommendations(projections)
        
        # Step 4: Display results
        print(f"\nSTEP 4: Draft Board")
        print("-" * 50)
        self.display_draft_board(recommendations)
        
        return recommendations

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = NFLFantasyPredictor()
    
    # Run complete analysis
    draft_recommendations = predictor.run_complete_analysis()
    
    # Optional: Save results to CSV
    if draft_recommendations is not None:
        draft_recommendations.to_csv('fantasy_draft_recommendations.csv', index=False)
        print(f"\n Draft recommendations saved to 'fantasy_draft_recommendations.csv'")
    
    print(f"\n Analysis complete! Good luck with your draft!")

# Additional utility functions
def compare_players(predictor, player1_stats, player2_stats, player1_name="Player 1", player2_name="Player 2"):
    """
    Compare projected fantasy points between two players
    """
    pred1 = predictor.predict_fantasy_points(player1_stats)
    pred2 = predictor.predict_fantasy_points(player2_stats)
    
    print(f"{player1_name} projected FPPG: {pred1:.2f}")
    print(f"{player2_name} projected FPPG: {pred2:.2f}")
    
    if pred1 > pred2:
        print(f"{player1_name} is projected to score {pred1-pred2:.2f} more points per game")
    elif pred2 > pred1:
        print(f"{player2_name} is projected to score {pred2-pred1:.2f} more points per game")
    else:
        print("Both players have similar projections")

def analyze_position_depth(projections_df, position, threshold=10.0):
    """
    Analyze the depth of talent at a specific position
    """
    if projections_df is None:
        return
    
    pos_players = projections_df[projections_df['Position'] == position]
    
    if len(pos_players) == 0:
        print(f"No {position} players found in projections")
        return
    
    # Find players above threshold
    fpts_cols = [col for col in pos_players.columns if 'FPTS' in col or 'Fantasy' in col]
    if fpts_cols:
        high_value = pos_players[pos_players[fpts_cols[0]] >= threshold]
        print(f"{position} players projected for {threshold}+ points: {len(high_value)}")
        print(f"Position depth score: {len(high_value)/len(pos_players)*100:.1f}%")
