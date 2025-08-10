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
from collections import defaultdict
import json
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
        self.qb_wr_chemistry_data = {} # stores QB-WR chemistry scores and historical data
        self.play_by_play_data = {} # stores detailed game-by-game connection data

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
    
    def scrape_qb_wr_connections(self, years=list(range(2020, 2025))):
        """
        This is where I scrape the data I need for QB-WR chemistry analysis
        PFR doesn't have direct QB-WR connection data, so I'm being clever here
        """
        print(f"Scraping QB-WR connection data for chemistry analysis...")
        
        all_connections = []
        
        for year in years:
            print(f"  Loading {year} QB-WR connection data...")
            
            try:
                # I'm using the same fantasy data but focusing on WRs/TEs this time
                receiving_url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                receiving_df = pd.read_html(receiving_url, header=1)[0]
                receiving_df = receiving_df[receiving_df['Rk'] != 'Rk']  # remove repeated headers
                receiving_df = receiving_df.fillna(0)
                
                # only want pass catchers for chemistry analysis
                if 'Pos' in receiving_df.columns:
                    wr_te_data = receiving_df[receiving_df['Pos'].isin(['WR', 'TE'])].copy()
                    
                    if 'Player' in wr_te_data.columns:
                        # extract team abbreviations - this is how I'll match QBs to WRs later
                        wr_te_data['Team'] = wr_te_data['Player'].str.extract(r'([A-Z]{2,3})$')
                        wr_te_data['CleanPlayer'] = wr_te_data['Player'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                        
                        wr_te_data['Year'] = year
                        
                        # convert the important stats to numbers - same as before
                        numeric_cols = ['Tgt', 'Rec', 'Yds.1', 'TD.1', 'G']
                        for col in numeric_cols:
                            if col in wr_te_data.columns:
                                wr_te_data[col] = pd.to_numeric(wr_te_data[col], errors='coerce').fillna(0)
                        
                        # calculate the metrics that matter for QB-WR chemistry
                        wr_te_data['Targets_Per_Game'] = wr_te_data['Tgt'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Receptions_Per_Game'] = wr_te_data['Rec'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Rec_Yards_Per_Game'] = wr_te_data['Yds.1'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Rec_TDs_Per_Game'] = wr_te_data['TD.1'] / wr_te_data['G'].replace(0, 1)
                        wr_te_data['Catch_Rate'] = wr_te_data['Rec'] / wr_te_data['Tgt'].replace(0, 1)  # this is key for chemistry
                        
                        all_connections.append(wr_te_data)
                        
                        print(f"    Found {len(wr_te_data)} WR/TE connections for {year}")
                
                time.sleep(1.5)  # don't hammer their servers
                
            except Exception as e:
                print(f"Error loading {year} connection data: {e}")
                continue
        
        if all_connections:
            self.play_by_play_data = pd.concat(all_connections, ignore_index=True)
            print(f"\nTotal QB-WR connections loaded: {len(self.play_by_play_data)}")
            return self.play_by_play_data
        else:
            print("No connection data loaded successfully")
            return None
    
    def scrape_team_qb_data(self, years=list(range(2020, 2025))):
        """
        Need to grab QB data so I can match them up with the WRs by team
        This is how I figure out which QB was throwing to which WR each year
        """
        print("Scraping team QB information...")
        
        qb_team_data = {}
        
        for year in years:
            print(f"  Loading {year} QB data...")
            
            try:
                # same fantasy data source, just filtering for QBs now
                url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
                df = pd.read_html(url, header=1)[0]
                df = df[df['Rk'] != 'Rk']  # get rid of header repeats
                df = df.fillna(0)
                
                # only want quarterbacks
                if 'Pos' in df.columns:
                    qb_data = df[df['Pos'] == 'QB'].copy()
                    
                    if 'Player' in qb_data.columns:
                        # same team extraction logic as WRs
                        qb_data['Team'] = qb_data['Player'].str.extract(r'([A-Z]{2,3})$')
                        qb_data['CleanPlayer'] = qb_data['Player'].str.replace(r'\s+[A-Z]{2,3}$', '', regex=True)
                        qb_data['Year'] = year
                        
                        # make sure key stats are numbers
                        for col in ['G', 'Cmp', 'Att.1', 'Yds.2', 'TD.2']:
                            if col in qb_data.columns:
                                qb_data[col] = pd.to_numeric(qb_data[col], errors='coerce').fillna(0)
                        
                        # store by team-year so I can easily match with WRs later
                        for _, qb in qb_data.iterrows():
                            team = qb['Team']
                            if team and not pd.isna(team):
                                key = f"{team}_{year}"
                                if key not in qb_team_data:
                                    qb_team_data[key] = []
                                qb_team_data[key].append(qb.to_dict())
                
                time.sleep(1)  # be nice to PFR
                
            except Exception as e:
                print(f"Error loading {year} QB data: {e}")
                continue
        
        print(f"QB data loaded for {len(qb_team_data)} team-year combinations")
        return qb_team_data
    
    def calculate_qb_wr_chemistry(self):
        """
        This is the magic - where I calculate how well QBs and WRs work together
        Using historical data to see which pairs have that special connection
        """
        if self.play_by_play_data is None or len(self.play_by_play_data) == 0:
            print("No connection data available. Please run scrape_qb_wr_connections() first.")
            return None
        
        print("Calculating QB-WR chemistry scores...")
        
        # need the QB data to match up with WRs
        qb_data = self.scrape_team_qb_data()
        
        chemistry_scores = {}
        
        # go through all the WR data and match with their QBs
        wr_data = self.play_by_play_data.copy()
        
        for _, wr in wr_data.iterrows():
            team = wr['Team']
            year = wr['Year']
            wr_name = wr['CleanPlayer']
            
            if not team or pd.isna(team):
                continue
            
            # find the main QB for this team in this year
            qb_key = f"{team}_{year}"
            if qb_key not in qb_data:
                continue
            
            # pick the QB who played the most games (usually the starter)
            team_qbs = qb_data[qb_key]
            primary_qb = max(team_qbs, key=lambda x: x.get('G', 0))
            qb_name = primary_qb['CleanPlayer']
            
            # create unique key for this QB-WR pair
            chemistry_key = f"{qb_name}_{wr_name}"
            
            if chemistry_key not in chemistry_scores:
                chemistry_scores[chemistry_key] = {
                    'qb_name': qb_name,
                    'wr_name': wr_name,
                    'years_together': [],
                    'total_games': 0,
                    'total_targets': 0,
                    'total_receptions': 0,
                    'total_yards': 0,
                    'total_tds': 0,
                    'avg_catch_rate': 0,
                    'chemistry_score': 0,
                    'consistency_score': 0
                }
            
            # add up all their stats together over the years
            chem = chemistry_scores[chemistry_key]
            chem['years_together'].append(year)
            chem['total_games'] += wr.get('G', 0)
            chem['total_targets'] += wr.get('Tgt', 0)
            chem['total_receptions'] += wr.get('Rec', 0)
            chem['total_yards'] += wr.get('Yds.1', 0)
            chem['total_tds'] += wr.get('TD.1', 0)
        
        # now calculate the actual chemistry scores - this is my secret sauce
        for key, chem in chemistry_scores.items():
            if chem['total_targets'] > 0 and chem['total_games'] > 0:
                # basic efficiency numbers - how well did they connect?
                chem['avg_catch_rate'] = chem['total_receptions'] / chem['total_targets']
                chem['yards_per_target'] = chem['total_yards'] / chem['total_targets']
                chem['tds_per_target'] = chem['total_tds'] / chem['total_targets']
                chem['targets_per_game'] = chem['total_targets'] / chem['total_games']
                
                # bonus points for playing together multiple years - chemistry builds over time
                years_together = len(set(chem['years_together']))
                longevity_bonus = min(years_together * 0.1, 0.3)  # cap at 30% bonus
                
                # high target share means QB trusts this WR
                volume_score = min(chem['targets_per_game'] / 8.0, 1.0)  # 8 targets/game is elite
                
                # catch rate is huge for chemistry - means they're on the same page
                catch_rate_score = chem['avg_catch_rate']  # already 0-1
                td_efficiency_score = min(chem['tds_per_target'] / 0.08, 1.0)  # 8% TD rate is solid
                
                # combine it all into final chemistry score
                base_score = (catch_rate_score * 0.4 +  # catch rate is most important
                             volume_score * 0.3 +       # volume shows trust
                             td_efficiency_score * 0.3)  # red zone chemistry matters
                
                chem['chemistry_score'] = base_score * (1 + longevity_bonus)
                
                # consistency bonus for guys who've been together multiple years
                if years_together >= 2:
                    chem['consistency_score'] = min(years_together / 3.0, 1.0)
                else:
                    chem['consistency_score'] = 0.5  # still decent for one year
        
        # filter out the noise - need at least 20 targets to be meaningful
        filtered_chemistry = {k: v for k, v in chemistry_scores.items() 
                            if v['total_targets'] >= 20}
        
        self.qb_wr_chemistry_data = filtered_chemistry
        
        print(f"Calculated chemistry scores for {len(filtered_chemistry)} QB-WR combinations")
        
        # show off the best chemistry pairs - this is the good stuff
        if filtered_chemistry:
            sorted_pairs = sorted(filtered_chemistry.items(), 
                                key=lambda x: x[1]['chemistry_score'], 
                                reverse=True)
            
            print("\nTop 10 QB-WR Chemistry Pairs:")
            print("-" * 50)
            for i, (key, chem) in enumerate(sorted_pairs[:10], 1):
                print(f"{i:2d}. {chem['qb_name']} → {chem['wr_name']}: "
                      f"{chem['chemistry_score']:.3f} "
                      f"({chem['total_targets']} targets, "
                      f"{chem['avg_catch_rate']:.1%} catch rate)")
        
        return self.qb_wr_chemistry_data
    
    def get_chemistry_multiplier(self, qb_name, wr_name):
        """
        Convert chemistry score to a projection multiplier
        Good chemistry = slight boost, bad chemistry = slight penalty
        """
        if not self.qb_wr_chemistry_data:
            return 1.0  # no data, no adjustment
        
        # try exact match first
        key = f"{qb_name}_{wr_name}"
        if key in self.qb_wr_chemistry_data:
            chem_score = self.qb_wr_chemistry_data[key]['chemistry_score']
            # convert chemistry score (0-2.0) to multiplier (0.8-1.3) - don't want huge swings
            multiplier = 0.8 + (chem_score * 0.25)  # scales nicely
            return min(max(multiplier, 0.8), 1.3)  # keep it reasonable
        
        # try fuzzy matching in case names don't match exactly
        for chem_key, chem_data in self.qb_wr_chemistry_data.items():
            stored_qb = chem_data['qb_name'].lower()
            stored_wr = chem_data['wr_name'].lower()
            
            # simple name matching - could be better but works for now
            if (qb_name.lower() in stored_qb or stored_qb in qb_name.lower()) and \
               (wr_name.lower() in stored_wr or stored_wr in wr_name.lower()):
                chem_score = chem_data['chemistry_score']
                multiplier = 0.8 + (chem_score * 0.25)
                return min(max(multiplier, 0.8), 1.3)
        
        # no match found, neutral multiplier
        return 1.0
    
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
    
    def generate_draft_recommendations(self, projections_df, top_n=20, use_chemistry=True):
        """
        Generate draft recommendations based on projected fantasy points with QB-WR chemistry adjustments
        """
        if projections_df is None:
            print("No projections data available")
            return None
        
        recommendations = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_players = projections_df[projections_df['Position'] == position].copy()
            
            if len(pos_players) > 0:
                # Apply chemistry adjustments for WRs and TEs
                if use_chemistry and position in ['WR', 'TE'] and self.qb_wr_chemistry_data:
                    pos_players = self._apply_chemistry_adjustments(pos_players)
                
                # Sort by a key metric
                if 'Chemistry_Adjusted_FPTS' in pos_players.columns:
                    pos_players = pos_players.sort_values('Chemistry_Adjusted_FPTS', ascending=False)
                elif 'FPTS' in pos_players.columns:
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
    
    def _apply_chemistry_adjustments(self, pos_players):
        """
        Apply QB-WR chemistry adjustments to WR/TE projections
        """
        pos_players_adjusted = pos_players.copy()
        
        # Find FPTS column
        fpts_col = None
        for col in ['FPTS', 'Fantasy Points', 'MISC FPTS']:
            if col in pos_players.columns:
                fpts_col = col
                break
        
        if not fpts_col:
            print("No fantasy points column found for chemistry adjustment")
            return pos_players
        
        # Convert to numeric
        pos_players_adjusted[fpts_col] = pd.to_numeric(pos_players_adjusted[fpts_col], errors='coerce').fillna(0)
        
        chemistry_adjustments = []
        
        for _, player in pos_players_adjusted.iterrows():
            player_name = player.get('Player', '').strip()
            
            # Extract clean player name (remove team designation)
            clean_player = player_name
            if ' ' in player_name:
                # Try to extract team abbreviation and remove it
                parts = player_name.split()
                if len(parts) >= 2 and len(parts[-1]) <= 3 and parts[-1].isupper():
                    clean_player = ' '.join(parts[:-1])
                    team_abbr = parts[-1]
                else:
                    team_abbr = None
            
            # Get current season QB projections to find likely QB-WR matchups
            base_fpts = player[fpts_col]
            chemistry_multiplier = 1.0
            best_qb_match = None
            
            # Try to find chemistry data for this receiver
            if self.qb_wr_chemistry_data:
                best_chemistry = 0
                for chem_key, chem_data in self.qb_wr_chemistry_data.items():
                    wr_name_stored = chem_data['wr_name'].lower()
                    clean_player_lower = clean_player.lower()
                    
                    # Simple name matching
                    if (clean_player_lower in wr_name_stored or 
                        wr_name_stored in clean_player_lower or
                        any(part in wr_name_stored for part in clean_player_lower.split() if len(part) > 2)):
                        
                        if chem_data['chemistry_score'] > best_chemistry:
                            best_chemistry = chem_data['chemistry_score']
                            chemistry_multiplier = self.get_chemistry_multiplier(chem_data['qb_name'], clean_player)
                            best_qb_match = chem_data['qb_name']
            
            # Apply chemistry adjustment
            chemistry_adjusted_fpts = base_fpts * chemistry_multiplier
            
            chemistry_adjustments.append({
                'player': player_name,
                'base_fpts': base_fpts,
                'chemistry_multiplier': chemistry_multiplier,
                'adjusted_fpts': chemistry_adjusted_fpts,
                'best_qb_match': best_qb_match
            })
        
        # Add adjusted column
        pos_players_adjusted['Chemistry_Adjusted_FPTS'] = [adj['adjusted_fpts'] for adj in chemistry_adjustments]
        pos_players_adjusted['Chemistry_Multiplier'] = [adj['chemistry_multiplier'] for adj in chemistry_adjustments]
        pos_players_adjusted['Best_QB_Match'] = [adj['best_qb_match'] for adj in chemistry_adjustments]
        
        # Show some adjustment examples
        significant_adjustments = [adj for adj in chemistry_adjustments 
                                 if abs(adj['chemistry_multiplier'] - 1.0) > 0.05]
        
        if significant_adjustments:
            print(f"\nChemistry Adjustments Applied to {len(significant_adjustments)} players:")
            print("-" * 60)
            for adj in significant_adjustments[:5]:  # Show top 5
                direction = "↑" if adj['chemistry_multiplier'] > 1.0 else "↓"
                print(f"{adj['player']:25} {direction} {adj['base_fpts']:5.1f} → {adj['adjusted_fpts']:5.1f} "
                      f"(x{adj['chemistry_multiplier']:.2f}) w/ {adj['best_qb_match']}")
        
        return pos_players_adjusted
    
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
    
    def run_complete_analysis(self, use_chemistry=True):
        """
        The full pipeline - load data, train model, scrape projections, generate rankings
        Now with QB-WR chemistry analysis for better WR projections!
        """
        print("Starting NFL Fantasy Football Analysis with QB-WR Chemistry \n")
        
        # step 1: get historical data and train our model
        print("Loading Historical Data and Training Advanced XGBoost Model")
        print("-" * 60)
        self.load_historical_data(list(range(2015, 2025)))  # 10 years should be enough
        self.train_model(optimize_hyperparameters=True)
        
        # step 2: QB-WR chemistry analysis - this is the new hotness
        if use_chemistry:
            print(f"\nAnalyzing QB-WR Chemistry (This Makes Us Better Than Everyone Else)")
            print("-" * 70)
            self.scrape_qb_wr_connections()  # get the connection data
            self.calculate_qb_wr_chemistry()  # crunch the chemistry numbers
        
        # step 3: get current year projections
        print(f"\nScraping Current Projections")
        print("-" * 50)
        projections = self.scrape_all_positions()
        
        # step 4: generate our recommendations with chemistry adjustments
        print(f"\nGenerating Chemistry-Enhanced Draft Recommendations")
        print("-" * 60)
        recommendations = self.generate_draft_recommendations(projections, use_chemistry=use_chemistry)
        
        # step 5: show the results
        print(f"\nYour Chemistry-Enhanced Draft Board")
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
        print(f"\nThese are PPR projections with QB-WR chemistry adjustments - adjust for your league scoring.")
        
        # save chemistry data too for reference
        if predictor.qb_wr_chemistry_data:
            chemistry_df = pd.DataFrame([
                {
                    'QB': data['qb_name'],
                    'WR': data['wr_name'],
                    'Chemistry_Score': data['chemistry_score'],
                    'Total_Targets': data['total_targets'],
                    'Catch_Rate': data['avg_catch_rate'],
                    'Years_Together': len(set(data['years_together']))
                }
                for data in predictor.qb_wr_chemistry_data.values()
            ])
            chemistry_df.to_csv('qb_wr_chemistry_scores.csv', index=False)
            print(f"QB-WR chemistry data saved to 'qb_wr_chemistry_scores.csv'")
    
    print(f"\nBooyah. Good luck drafting with your secret chemistry weapon, friends. \n - Kevin Veeder")

# bonus functions for nerds who want to dig deeper
def compare_players(predictor, player1_stats, player2_stats, player1_name="Player 1", player2_name="Player 2"):
    """
    Head to head player comparison - now with chemistry awareness for WRs
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

def analyze_qb_wr_chemistry(predictor, qb_name, wr_name):
    """
    Get detailed chemistry analysis for a specific QB-WR pair
    """
    if not predictor.qb_wr_chemistry_data:
        print("No chemistry data loaded. Run the analysis first.")
        return
    
    multiplier = predictor.get_chemistry_multiplier(qb_name, wr_name)
    
    # find the exact match if it exists
    key = f"{qb_name}_{wr_name}"
    if key in predictor.qb_wr_chemistry_data:
        chem = predictor.qb_wr_chemistry_data[key]
        print(f"\n{qb_name} → {wr_name} Chemistry Report:")
        print("-" * 50)
        print(f"Chemistry Score: {chem['chemistry_score']:.3f}")
        print(f"Fantasy Multiplier: {multiplier:.2f}x")
        print(f"Years Together: {len(set(chem['years_together']))}")
        print(f"Total Targets: {chem['total_targets']}")
        print(f"Catch Rate: {chem['avg_catch_rate']:.1%}")
        print(f"Targets/Game: {chem['targets_per_game']:.1f}")
        print(f"TDs/Target: {chem['tds_per_target']:.1%}")
    else:
        print(f"No specific chemistry data found for {qb_name} → {wr_name}")
        print(f"Using default multiplier: {multiplier:.2f}x")

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