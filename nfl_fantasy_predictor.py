"""
NFL Fantasy Football Draft Predictor
====================================
Scrapes NFL data, trains an XGBoost model, and generates draft recommendations.
Upgraded from basic linear regression to something actually useful.

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
    def __init__(self):
        self.model = None # XGBoost model - way better than linear regression
        self.scaler = StandardScaler() # gotta normalize the features
        self.features = [] # starts empty, gets populated with all the good stuff
        self.historical_data = None 
        self.projections_data = {} # holds current year projections by position
        self.feature_importance = None # helps us see what actually matters
        self.best_params = None # hyperparams found by optuna

    def load_historical_data(self, years=list(range(2015, 2025))):
        # Pro Football Reference has solid historical data - 10 years should be plenty
        print(f"Loading fantasy football data for years: {years}")

        all_data = []
        
        # Hit each year individually - takes a bit but gets us clean data
        for year in years:
            print(f"  Loading {year} data...")
            
            url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
            
            try:
                # pandas read_html is magic for scraping tables
                df = pd.read_html(url, header=1)[0]
                
                # pro-football-reference repeats headers throughout the table, annoying
                df = df[df['Rk'] != 'Rk']  
                df = df.fillna(0)  # zeros better than NaN for our purposes
                
                # convert everything to numbers - some cols might not exist in older years
                numeric_columns = [
                    'FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1', 'Yds', 'TD', 'Int', 
                    'Yds.1', 'TD.1', 'Yds.2', 'TD.2', 'FL', 'Fmb', 'Rush TD', 'Rec TD', 
                    'Ret TD', '2PM', '2PP', 'FantPt/G', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'
                ]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce') # coerce handles the weird values
                
                # this is where the magic happens - create all our advanced features
                df = self._engineer_features(df)
                
                # need at least 4 games to have meaningful stats
                df = df[df['G'] >= 4]
                
                df['Year'] = year
                all_data.append(df)
                print(f"Successfully loaded {len(df)} players from {year}")
                
                # be nice to their servers
                time.sleep(1)
                
            except Exception as e:
                print(f"Error loading {year} data: {e}")
                continue
        
        if all_data:
            # smash it all together
            self.historical_data = pd.concat(all_data, ignore_index=True)
            total_players = len(self.historical_data)
            print(f"\nCombined dataset: {total_players} total player-seasons")
            print(f"Years included: {sorted(self.historical_data['Year'].unique())}")
            return self.historical_data
        else:
            print("Uh oh, nothing loaded successfully")
            return None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This is where we create all the fancy features that actually matter
        """
        # main target variable
        df['FPPG'] = df['FantPt'] / df['G']
        
        # efficiency is king in fantasy - yards per opportunity
        if 'Yds' in df.columns and 'Att' in df.columns:
            df['Yards_Per_Carry'] = df['Yds'] / df['Att'].replace(0, 1) # avoid divide by zero
        
        if 'Yds.1' in df.columns and 'Tgt' in df.columns:
            df['Yards_Per_Target'] = df['Yds.1'] / df['Tgt'].replace(0, 1)
        
        if 'Rec' in df.columns and 'Tgt' in df.columns:
            df['Catch_Rate'] = df['Rec'] / df['Tgt'].replace(0, 1) # crucial for WRs/TEs
        
        # volume is everything - normalize by games played
        df['Attempts_Per_Game'] = df['Att'] / df['G']
        df['Targets_Per_Game'] = df['Tgt'] / df['G'] # this usually predicts fantasy success
        df['Receptions_Per_Game'] = df['Rec'] / df['G']
        
        # touchdown rates - some guys just find the endzone
        if 'TD' in df.columns:
            df['Rush_TD_Per_Game'] = df['TD'] / df['G']
            df['Rush_TD_Rate'] = df['TD'] / df['Att'].replace(0, 1) # TDs per carry
        
        if 'TD.1' in df.columns:
            df['Rec_TD_Per_Game'] = df['TD.1'] / df['G']
            df['Rec_TD_Rate'] = df['TD.1'] / df['Rec'].replace(0, 1) # TDs per catch
        
        # combine rushing and receiving for dual-threat players
        if 'Yds' in df.columns and 'Yds.1' in df.columns:
            df['Total_Yards'] = df['Yds'].fillna(0) + df['Yds.1'].fillna(0)
            df['Total_Yards_Per_Game'] = df['Total_Yards'] / df['G']
        
        if 'TD' in df.columns and 'TD.1' in df.columns:
            df['Total_TDs'] = df['TD'].fillna(0) + df['TD.1'].fillna(0)
            df['Total_TDs_Per_Game'] = df['Total_TDs'] / df['G']
        
        # placeholder for now - would need weekly data for real consistency calc
        df['Fantasy_Points_Consistency'] = df['FPPG']
        
        # position matters a lot - QBs vs RBs have totally different patterns
        if 'Pos' in df.columns:
            pos_dummies = pd.get_dummies(df['Pos'], prefix='Pos')
            df = pd.concat([df, pos_dummies], axis=1)
        
        # TODO: add age and team pace when I get around to scraping that
        
        # clean up any leftover NaNs
        df = df.fillna(0)
        
        return df
    
    def prepare_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Get our data ready for the XGBoost model
        """
        if self.historical_data is None:
            print("No historical data loaded. Please run load_historical_data() first.")
            return None, None
        
        df = self.historical_data.copy()
        
        # all the features we might want to use - some might not exist in all years
        potential_features = [
            # basic counting stats
            'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1',
            # yardage totals and per game
            'Yds', 'Yds.1', 'Yds.2', 'Total_Yards', 'Total_Yards_Per_Game',
            # touchdown stats
            'TD', 'TD.1', 'Total_TDs', 'Total_TDs_Per_Game',
            # per game usage - usually the most predictive
            'Attempts_Per_Game', 'Targets_Per_Game', 'Receptions_Per_Game',
            'Rush_TD_Per_Game', 'Rec_TD_Per_Game',
            # efficiency ratios - separates good players from great ones
            'Yards_Per_Carry', 'Yards_Per_Target', 'Catch_Rate',
            'Rush_TD_Rate', 'Rec_TD_Rate',
            # negative plays and special stuff
            'Int', 'FL', 'Fmb', '2PM', '2PP',
            'Fantasy_Points_Consistency'
        ]
        
        # grab any position dummy variables we created
        pos_columns = [col for col in df.columns if col.startswith('Pos_')]
        potential_features.extend(pos_columns)
        
        # only use features that actually exist in our dataset
        self.features = [f for f in potential_features if f in df.columns]
        
        # make sure everything is numeric
        for feature in self.features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # filter out any weird FPPG values
        df = df[df['FPPG'].notna() & (df['FPPG'] >= 0)]
        
        # X is our features, y is what we're trying to predict (FPPG)
        X = df[self.features]
        y = df['FPPG']
        
        # clean up any infinite values that snuck through
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Training data prepared with {len(X)} samples and {len(self.features)} features")
        print(f"Features: {self.features[:10]}{'...' if len(self.features) > 10 else ''}")
        return X, y
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Let optuna find the best hyperparameters - this takes a while but worth it
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
            
            # test these params with cross-validation
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            return -scores.mean() # return positive MAE
        
        print("Optimizing hyperparameters... grab a coffee, this takes a few minutes")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"Best MAE found: {study.best_value:.3f}")
        return study.best_params
    
    def train_model(self, optimize_hyperparameters: bool = True) -> Optional[xgb.XGBRegressor]:
        """
        Train our XGBoost model - so much better than linear regression
        """
        X, y = self.prepare_training_data()
        if X is None:
            return None
        
        # scale features so they're all on similar ranges
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        if optimize_hyperparameters and len(X_train) > 100:  
            # run optuna optimization if we have enough data
            self.best_params = self._optimize_hyperparameters(X_train, y_train)
        else:
            # decent default params if we skip optimization
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
        
        # train the final model with our best params
        print("Training final XGBoost model...")
        self.model = xgb.XGBRegressor(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # see how well we did
        y_pred = self.model.predict(X_test)
        
        # bunch of different metrics to get the full picture
        mae = mean_absolute_error(y_test, y_pred) # main one we care about
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred) # how much variance we explain
        
        # cross-validation gives us a more robust estimate
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS")
        print("="*50)
        print(f"Test MAE: {mae:.3f} fantasy points")
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test R²: {r2:.3f} (higher is better)")
        print(f"CV MAE: {cv_mae:.3f} (±{cv_scores.std():.3f})")
        
        # see which features the model thinks are most important
        self.feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (what drives fantasy success):")
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
        Predict FPPG for a single player
        """
        if self.model is None:
            print("Model not trained yet. Please run train_model() first.")
            return None
        
        # build feature array in the same order as training
        stats_array = []
        for feature in self.features:
            stats_array.append(player_stats.get(feature, 0))
        
        # apply the same scaling we used in training
        stats_scaled = self.scaler.transform([stats_array])
        
        prediction = self.model.predict(stats_scaled)[0]
        return max(0, prediction)  # can't have negative fantasy points
    
    def generate_draft_recommendations(self, projections_df, top_n=20):
        """
        Generate draft recommendations based on projected fantasy points
        """
        if projections_df is None:
            print("No projections data available")
            return None
        
        # TODO: add VOR (value over replacement) and positional scarcity analysis
        # for now just sorting by projected points works pretty well 
        
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
        The full pipeline - load data, train model, scrape projections, generate rankings
        """
        print("Starting NFL Fantasy Football Analysis \n")
        
        # step 1: get historical data and train our model
        print("Loading Historical Data and Training Advanced XGBoost Model")
        print("-" * 60)
        self.load_historical_data(list(range(2015, 2025)))  # 10 years should be enough
        self.train_model(optimize_hyperparameters=True)
        
        # step 2: get current year projections
        print(f"\nScraping Current Projections")
        print("-" * 50)
        projections = self.scrape_all_positions()
        
        # step 3: generate our recommendations
        print(f"\nGenerating Draft Recommendations")
        print("-" * 50)
        recommendations = self.generate_draft_recommendations(projections)
        
        # step 4: show the results
        print(f"\nYour Draft Board")
        print("-" * 50)
        self.display_draft_board(recommendations)
        
        return recommendations

# run the whole thing
if __name__ == "__main__":
    predictor = NFLFantasyPredictor()
    
    # do the full analysis
    draft_recommendations = predictor.run_complete_analysis()
    
    # save results
    if draft_recommendations is not None:
        draft_recommendations.to_csv('fantasy_draft_recommendations.csv', index=False)
        print(f"\nDraft recommendations saved to 'fantasy_draft_recommendations.csv'")
        print(f"\nThese are PPR projections - adjust for your league scoring.")
    
    print(f"\nBooyah. Good luck drafting, friends. \n - Kevin Veeder")

# bonus functions for nerds who want to dig deeper
def compare_players(predictor, player1_stats, player2_stats, player1_name="Player 1", player2_name="Player 2"):
    """
    Head to head player comparison
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
    See how deep the talent goes at each position - helps with draft strategy
    """
    if projections_df is None:
        return
    
    pos_players = projections_df[projections_df['Position'] == position]
    
    if len(pos_players) == 0:
        print(f"No {position} players found in projections")
        return
    
    # how many guys are projected above the threshold?
    fpts_cols = [col for col in pos_players.columns if 'FPTS' in col or 'Fantasy' in col]
    if fpts_cols:
        high_value = pos_players[pos_players[fpts_cols[0]] >= threshold]
        print(f"{position} players projected for {threshold}+ points: {len(high_value)}")
        print(f"Position depth score: {len(high_value)/len(pos_players)*100:.1f}%")