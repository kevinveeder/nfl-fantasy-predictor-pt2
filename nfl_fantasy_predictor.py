"""
NFL Fantasy Football Draft Predictor
====================================
A comprehensive tool for scraping NFL data, training predictive models,
and generating draft recommendations based on historical performance and projections.

Author: Kevin Veeder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import requests
from bs4 import BeautifulSoup
import lxml
import time
import warnings
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

class NFLFantasyPredictor:
    # initalize the predictor with default settings
    def __init__(self):
        self.model = None # placeholder for the trained XGBoost model
        self.scaler = StandardScaler() # for feature scaling
        self.features = [] # will be populated with comprehensive features
        self.historical_data = None # placeholder for historical data
        self.projections_data = {} # empty dict to hold projections data for different positions
        self.feature_importance = None # store feature importance
        self.best_params = None # store optimal hyperparameters

    # Load historical fantasy data from Pro Football Reference   
    def load_historical_data(self, years=list(range(2015, 2025))):
        print(f"Loading fantasy football data for years: {years}")

        # initatialize storage for all years' data
        all_data = []
        
        # Loop through each year and scrape data
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
                
                # Convert relevant columns to numeric - expanded feature set
                numeric_columns = [
                    'FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1', 'Yds', 'TD', 'Int', 
                    'Yds.1', 'TD.1', 'Yds.2', 'TD.2', 'FL', 'Fmb', 'Rush TD', 'Rec TD', 
                    'Ret TD', '2PM', '2PP', 'FantPt/G', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'
                ]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce') # helps handle messy data
                
                # Calculate comprehensive features
                df = self._engineer_features(df)
                
                # Filter out players with minimal games (less informative)
                df = df[df['G'] >= 4]  # At least 4 games for meaningful stats
                
                # Add year column for reference
                df['Year'] = year
                
                all_data.append(df)
                print(f"Successfully loaded {len(df)} players from {year}")
                
                # Add small delay between requests to be respectful
                time.sleep(1)
                
            except Exception as e:
                print(f"Error loading {year} data: {e}")
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
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for fantasy football prediction
        """
        # Calculate Fantasy Points Per Game (FPPG)
        df['FPPG'] = df['FantPt'] / df['G']
        
        # Efficiency metrics
        if 'Yds' in df.columns and 'Att' in df.columns:
            df['Yards_Per_Carry'] = df['Yds'] / df['Att'].replace(0, 1)
        
        if 'Yds.1' in df.columns and 'Tgt' in df.columns:
            df['Yards_Per_Target'] = df['Yds.1'] / df['Tgt'].replace(0, 1)
        
        if 'Rec' in df.columns and 'Tgt' in df.columns:
            df['Catch_Rate'] = df['Rec'] / df['Tgt'].replace(0, 1)
        
        # Usage metrics (attempts/targets per game)
        df['Attempts_Per_Game'] = df['Att'] / df['G']
        df['Targets_Per_Game'] = df['Tgt'] / df['G']
        df['Receptions_Per_Game'] = df['Rec'] / df['G']
        
        # Touchdown efficiency
        if 'TD' in df.columns:
            df['Rush_TD_Per_Game'] = df['TD'] / df['G']
            df['Rush_TD_Rate'] = df['TD'] / df['Att'].replace(0, 1)
        
        if 'TD.1' in df.columns:
            df['Rec_TD_Per_Game'] = df['TD.1'] / df['G']
            df['Rec_TD_Rate'] = df['TD.1'] / df['Rec'].replace(0, 1)
        
        # Total production metrics
        if 'Yds' in df.columns and 'Yds.1' in df.columns:
            df['Total_Yards'] = df['Yds'].fillna(0) + df['Yds.1'].fillna(0)
            df['Total_Yards_Per_Game'] = df['Total_Yards'] / df['G']
        
        if 'TD' in df.columns and 'TD.1' in df.columns:
            df['Total_TDs'] = df['TD'].fillna(0) + df['TD.1'].fillna(0)
            df['Total_TDs_Per_Game'] = df['Total_TDs'] / df['G']
        
        # Consistency metric (will be calculated later with weekly data if available)
        df['Fantasy_Points_Consistency'] = df['FPPG']  # Placeholder
        
        # Position-specific features
        if 'Pos' in df.columns:
            # Create position dummy variables
            pos_dummies = pd.get_dummies(df['Pos'], prefix='Pos')
            df = pd.concat([df, pos_dummies], axis=1)
        
        # Age feature (if available in future scraping)
        # Team offensive pace (if available)
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        return df
    
    def prepare_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features and target variables for model training with comprehensive feature set
        """
        if self.historical_data is None:
            print("No historical data loaded. Please run load_historical_data() first.")
            return None, None
        
        df = self.historical_data.copy()
        
        # Define comprehensive feature set
        potential_features = [
            # Basic stats
            'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1',
            # Yardage
            'Yds', 'Yds.1', 'Yds.2', 'Total_Yards', 'Total_Yards_Per_Game',
            # Touchdowns
            'TD', 'TD.1', 'Total_TDs', 'Total_TDs_Per_Game',
            # Per game metrics
            'Attempts_Per_Game', 'Targets_Per_Game', 'Receptions_Per_Game',
            'Rush_TD_Per_Game', 'Rec_TD_Per_Game',
            # Efficiency metrics
            'Yards_Per_Carry', 'Yards_Per_Target', 'Catch_Rate',
            'Rush_TD_Rate', 'Rec_TD_Rate',
            # Other stats
            'Int', 'FL', 'Fmb', '2PM', '2PP',
            # Position dummies (will be added if they exist)
            'Fantasy_Points_Consistency'
        ]
        
        # Add position dummy variables if they exist
        pos_columns = [col for col in df.columns if col.startswith('Pos_')]
        potential_features.extend(pos_columns)
        
        # Filter features that actually exist in the data
        self.features = [f for f in potential_features if f in df.columns]
        
        # Ensure all feature columns are numeric
        for feature in self.features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Remove rows with invalid FPPG
        df = df[df['FPPG'].notna() & (df['FPPG'] >= 0)]
        
        # Prepare features (X) and target (y)
        X = df[self.features]
        y = df['FPPG']
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Training data prepared with {len(X)} samples and {len(self.features)} features")
        print(f"Features: {self.features[:10]}{'...' if len(self.features) > 10 else ''}")
        return X, y
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Use Optuna to find optimal XGBoost hyperparameters
        """
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            # Cross-validation
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            return -scores.mean()
        
        print("Optimizing hyperparameters...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"Best MAE: {study.best_value:.3f}")
        return study.best_params
    
    def train_model(self, optimize_hyperparameters: bool = True) -> Optional[xgb.XGBRegressor]:
        """
        Train XGBoost model with optional hyperparameter tuning
        """
        X, y = self.prepare_training_data()
        if X is None:
            return None
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        if optimize_hyperparameters and len(X_train) > 100:  # Only optimize with sufficient data
            self.best_params = self._optimize_hyperparameters(X_train, y_train)
        else:
            # Default parameters
            self.best_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'random_state': 42
            }
        
        # Train final model
        print("Training final XGBoost model...")
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate comprehensive metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS")
        print("="*50)
        print(f"Test MAE: {mae:.3f}")
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test R²: {r2:.3f}")
        print(f"CV MAE: {cv_mae:.3f} (±{cv_scores.std():.3f})")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
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
    
    def predict_fantasy_points(self, player_stats: Dict) -> Optional[float]:
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
        
        # Scale the input features
        stats_scaled = self.scaler.transform([stats_array])
        
        prediction = self.model.predict(stats_scaled)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def generate_draft_recommendations(self, projections_df, top_n=20):
        """
        Generate draft recommendations based on projected fantasy points
        """
        if projections_df is None:
            print("No projections data available")
            return None
        
        # This needs much more ...work... or as some might say: more sophisticated
        # analysis like value over replacement, positional scarcity, etc. I got lazy. 
        
        recommendations = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = projections_df[projections_df['Position'] == position].copy()
            
            if len(pos_players) > 0:
                # Sort by a key metric
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
        print("NFL FANTASY DRAFT RECOMMENDATIONS")
        print("="*80)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = recommendations_df[recommendations_df['Position'] == position]
            
            if len(pos_players) > 0:
                print(f"\nTOP {position}s:")
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
        print("Starting NFL Fantasy Football Analysis \n")
        
        # Load historical data and train model
        print("Loading Historical Data and Training Advanced XGBoost Model")
        print("-" * 60)
        self.load_historical_data(list(range(2015, 2025)))  # 10 years of data
        self.train_model(optimize_hyperparameters=True)
        
        # Scrape current projections
        print(f"\nScraping Current Projections")
        print("-" * 50)
        projections = self.scrape_all_positions()
        
        # Generate recommendations
        print(f"\nGenerating Draft Recommendations")
        print("-" * 50)
        recommendations = self.generate_draft_recommendations(projections)
        
        # Display results
        print(f"\nDraft Board")
        print("-" * 50)
        self.display_draft_board(recommendations)
        
        return recommendations

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = NFLFantasyPredictor()
    
    # Run complete analysis
    draft_recommendations = predictor.run_complete_analysis()
    
    # Save results to CSV
    if draft_recommendations is not None:
        draft_recommendations.to_csv('fantasy_draft_recommendations.csv', index=False)
        print(f"\nDraft recommendations saved to 'fantasy_draft_recommendations.csv'")
        print(f"\nThe projected points are standard ppr, so adjust accordingly for your league settings.")
    
    print(f"\nBooyah. Good luck drafting, friends. \n - Kevin Veeder")

# more fun functions to play with
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