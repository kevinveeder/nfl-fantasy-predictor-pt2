# NFL Fantasy Analytics - ML Model for Draft Strategy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning tool for NFL fantasy football draft strategy powered by XGBoost and advanced feature engineering.

## What's New in Version 2.3

### **NEW: Season-Ending Injury Analysis**
- **Injury History Tracking** - Identifies players with major injury history (ACL, Achilles, season-ending surgeries)
- **Risk Scoring System** - Creates injury risk scores (0-1 scale) based on frequency and severity
- **Projection Adjustments** - Applies injury multipliers (0.75x-1.0x) to account for injury-prone players
- **Historical Analysis** - Scrapes Pro Football Reference for injury patterns and games missed

### **Existing: QB Support System Analysis**
- **RB Support Quality** - Evaluates how RB quality affects QB performance
- **O-Line Protection Analysis** - Uses completion percentage and scrambling rate as protection indicators
- **Support Multipliers** - Adjusts QB projections (0.9x-1.2x) based on supporting cast quality

### **Existing: QB-WR Chemistry Analysis**
- **Connection Scoring** - Quantifies QB-WR chemistry using catch rate, target share, and touchdown efficiency
- **Multi-Year Bonuses** - Rewards established connections with chemistry boosts
- **Projection Adjustments** - Modifies WR/TE rankings (0.9x-1.2x) based on QB chemistry

### **Core Features**
- **XGBoost ML Model** with automated hyperparameter optimization
- **25+ Features** including efficiency metrics, usage patterns, and injury history
- **10 Years of Training Data** (2015-2024) for robust predictions
- **Realistic Draft Guides** following actual ADP patterns

## Features

### **Machine Learning**
- **XGBoost Regressor** with Optuna hyperparameter optimization
- **5-Fold Cross-Validation** for robust model evaluation
- **Feature Scaling** with StandardScaler for optimal performance
- **Comprehensive Metrics**: MAE, RMSE, R², Cross-validation scores

### **Advanced Feature Engineering**
- **Efficiency Metrics**: Yards per carry, yards per target, catch rate
- **Usage Patterns**: Attempts/targets/receptions per game
- **Production Metrics**: Total yards, total TDs, touchdown rates
- **Injury Features**: Risk scores, recent major injuries, career injury rates
- **Position Intelligence**: Position-specific dummy variables

### **Data Sources**
- **10 Years** of historical NFL data (2015-2024) from Pro Football Reference
- **Current Season Projections** from FantasyPros
- **Injury Data** scraped from Pro Football Reference injury reports
- **Multi-Position Support**: QB, RB, WR, TE with automatic data cleaning

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kevinveeder/nfl-fantasy-predictor-pt2
   cd nfl-fantasy-predictor-pt2
   ```
2. Create Virtual Environment (optional but recommended)
   ```bash
   python -m venv new_env
   new_env\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the predictor**
   ```bash
   python nfl_fantasy_predictor.py
   ```
***Please note.** Running these steps is everything you need to draft well. This script will also generate your Draft Guide (`fantasy_draft_guide.csv` in the project folder) that will give you round-by-round recommendations. Draft well, my friends.*

## Dependencies

### **Core ML Stack**
- `xgboost==2.1.3` - Advanced gradient boosting
- `optuna==4.1.0` - Hyperparameter optimization
- `scikit-learn==1.7.1` - ML utilities and preprocessing
- `pandas==2.3.1` - Data manipulation
- `numpy==2.3.2` - Numerical computing

### **Web Scraping**
- `requests==2.32.4` - HTTP requests
- `beautifulsoup4==4.13.4` - HTML parsing
- `lxml==6.0.0` - XML/HTML processing

## Model Pipeline

### **1. Data Collection & Engineering**
```
Historical NFL Data (2015-2024) + Injury Data (2020-2024)
        ↓
Advanced Feature Engineering
        ↓  
25+ Features including injury risk, chemistry, and support analysis
```

### **2. Multi-Factor Analysis**
```
QB Support Analysis → QB multipliers (0.9x-1.2x)
QB-WR Chemistry → WR/TE multipliers (0.9x-1.2x)  
Injury History → Risk multipliers (0.75x-1.0x)
        ↓
Combined adjustments applied to projections
```

### **3. Model Training & Predictions**
```
Feature Scaling → Train/Test Split → Hyperparameter Optimization
        ↓
XGBoost Training with Cross-Validation
        ↓
Current Season Projections + Multi-Factor Adjustments
        ↓
Final Draft Rankings
```

## Sample Output

```
==================================================
MODEL TRAINING RESULTS
==================================================
Test MAE: 0.158 fantasy points
Test RMSE: 0.263
Test R²: 0.997 (higher is better)
CV MAE: 0.170 (±0.033)

Top 10 Most Important Features:
                       Feature  Importance
13                   Total_TDs    0.530372
30  Fantasy_Points_Consistency    0.246101
9                  Total_Yards    0.080124
17         Receptions_Per_Game    0.074781
10        Total_Yards_Per_Game    0.019880

Analyzing player injury histories for season-ending injuries...
Processed injury history for 493 players

Top 10 Highest Injury Risk Players:
----------------------------------------------------------------------
 1. Aaron Rodgers             Risk: 1.000 (Major: 1, Recent: Yes)
 2. Christian McCaffrey       Risk: 0.800 (Major: 1, Recent: No)
 3. Saquon Barkley           Risk: 0.600 (Major: 0, Mod: 2, Recent: Yes)
...

Analyzing QB-WR Chemistry
----------------------------------------------------------------------
Calculated chemistry scores for 847 QB-WR combinations

Top 10 QB-WR Chemistry Pairs:
--------------------------------------------------
 1. Joe Burrow -> Ja'Marr Chase: 1.134 (chemistry score)
 2. Josh Allen -> Stefon Diggs: 1.087 (chemistry score)
 3. Dak Prescott -> CeeDee Lamb: 0.567 (chemistry score)
...

Multi-Factor Adjustments Applied:
------------------------------------------------------------
Ja'Marr Chase CIN         UP   12.8 -> 14.1 (x1.10) w/ Joe Burrow
Christian McCaffrey CAR   DOWN 15.2 -> 12.2 (x0.80) injury risk
Josh Allen BUF           UP   21.8 -> 24.9 (x1.14) w/ support
...
```

## Model Analysis

### **Feature Categories by Impact**

| Category | Examples | Impact Level |
|----------|----------|-------------|
| **Production** | Total TDs, Total Yards | Very High |
| **Consistency** | Fantasy Points Consistency | High |
| **Usage** | Targets/game, Attempts/game | High |
| **Efficiency** | Yards/target, Catch rate | Medium |
| **Injury Risk** | Risk scores, Recent injuries | Medium |
| **Position** | QB, RB, WR, TE indicators | Low |

### **Multi-Factor Analysis**
- **QB Support System**: Analyzes RB quality and O-line protection to adjust QB projections
- **QB-WR Chemistry**: Uses historical connection data to modify WR/TE rankings
- **Injury Risk Assessment**: Penalizes players with significant injury history
- **Combined Adjustments**: All factors work together for more accurate projections

## Multi-Factor Analysis Methodology

### **Injury Risk Assessment**
The system tracks season-ending injuries to identify injury-prone players:

- **Data Source**: Pro Football Reference injury patterns (2020-2024)
- **Risk Scoring**: 0-1 scale based on major injuries, frequency, and recency
- **Major Injuries**: ACL tears, Achilles injuries, season-ending surgeries
- **Projection Impact**: High-risk players receive 0.75x-1.0x multipliers
- **Recent Bias**: Recent major injuries weighted more heavily

### **QB Support System Analysis**
Evaluates supporting cast quality that affects QB performance:

- **RB Support**: Quality of running game (YPG, YPC, workload consistency)
- **O-Line Protection**: Uses completion percentage and scrambling rate as proxies
- **Combined Score**: RB support (40%) + O-line protection (60%)
- **Multiplier Range**: 0.9x-1.2x based on overall support quality

### **QB-WR Chemistry Analysis**
Quantifies historical QB-WR connections:

- **Base Formula**: Catch rate (40%) + Volume (30%) + TD efficiency (30%)
- **Longevity Bonus**: Multi-year connections get up to 30% bonus
- **Sample Size**: Minimum 20 targets required for meaningful analysis
- **Multiplier Range**: 0.9x-1.2x based on chemistry score

## Future Enhancements

Potential improvements being considered:

- **Weekly Prediction Models** for in-season management
- **Strength of Schedule** advanced modeling
- **Value Over Replacement** (VOR) calculations
- **Defense vs Position** matchup analysis
- **League-Specific Scoring** customization
- **Advanced Visualizations** and dashboards
## Author

**Kevin Veeder**

*"This isnt your dad's league anymore."*

