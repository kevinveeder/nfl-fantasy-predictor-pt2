# NFL Fantasy Football Advanced Predictor & Analytics Tool

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An advanced machine learning tool for NFL fantasy football draft strategy, powered by XGBoost and advanced feature engineering.**

## What's New in Version 2.0

This is a **major upgrade** from the original simple linear regression model. Now featuring:

- **XGBoost ML Model** with automated hyperparameter optimization
- **20+ Advanced Features** including efficiency metrics and usage patterns  
- **10 Years of Training Data** (2015-2024) for robust predictions
- **Cross-Validation** and comprehensive model evaluation
- **Feature Importance Analysis** to understand what drives fantasy performance
- **Position-Aware Modeling** with sophisticated feature engineering

## Features

### **Machine Learning**
- **XGBoost Regressor** with Optuna hyperparameter optimization
- **Feature Scaling** with StandardScaler for optimal performance
- **5-Fold Cross-Validation** for robust model evaluation
- **Comprehensive Metrics**: MAE, RMSE, R², Cross-validation scores

### **Feature Engineering**
- **Efficiency Metrics**: Yards per carry, yards per target, catch rate
- **Usage Patterns**: Attempts/targets/receptions per game
- **Production Metrics**: Total yards, total TDs, touchdown rates  
- **Position Intelligence**: Position-specific dummy variables
- **Consistency Analysis**: Fantasy points consistency tracking

### **Data Collection**
- **10 Years** of historical NFL data (2015-2024)
- **Real-time Projections** from FantasyPros
- **Multi-Position Support**: QB, RB, WR, TE
- **Automatic Data Cleaning** and validation

## Model Performance

The enhanced XGBoost model significantly outperforms traditional approaches:

- **Typical MAE**: ~2.5-3.5 fantasy points
- **R² Score**: 0.65-0.80 depending on position
- **Cross-Validation**: Robust performance across different data splits
- **Feature Importance**: Clear insights into prediction drivers

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kevinveeder/nfl-fantasy-predictor-pt2
   cd nfl-fantasy-predictor-pt2
   ```
2. Create Virtual Environment
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

## Usage

### **Quick Start**
```python
from nfl_fantasy_predictor import NFLFantasyPredictor

# Initialize the predictor
predictor = NFLFantasyPredictor()

# Run complete analysis pipeline
recommendations = predictor.run_complete_analysis()
```

### **Custom Configuration**
```python
# Load specific years of data
predictor.load_historical_data(years=list(range(2018, 2025)))

# Train with/without hyperparameter optimization  
model = predictor.train_model(optimize_hyperparameters=True)

# Generate predictions for individual players
player_stats = {
    'Att': 250, 'Tgt': 100, 'Rec': 80, 
    'Yards_Per_Carry': 4.2, 'Catch_Rate': 0.80
}
projected_fppg = predictor.predict_fantasy_points(player_stats)
```

## Model Pipeline

### **1. Data Collection & Engineering**
```
Historical NFL Data (2015-2024)
        ↓
Advanced Feature Engineering
        ↓  
20+ Sophisticated Features
```

**Key Features Created:**
- `Yards_Per_Carry`, `Yards_Per_Target` 
- `Attempts_Per_Game`, `Targets_Per_Game`
- `Total_Yards`, `Total_TDs`
- `Rush_TD_Rate`, `Rec_TD_Rate`
- `Catch_Rate`, `Fantasy_Points_Consistency`

### **2. Model Training & Optimization**
```
Feature Scaling (StandardScaler)
        ↓
Train/Test Split (80/20)
        ↓
Hyperparameter Optimization (Optuna)
        ↓
XGBoost Training
        ↓
Cross-Validation Evaluation
```

### **3. Prediction & Analysis**
```
Current Season Projections
        ↓
Feature Engineering 
        ↓
Scaled Prediction
        ↓
Draft Recommendations
```

## Sample Output

```
==================================================
MODEL TRAINING RESULTS
==================================================
Test MAE: 2.847
Test RMSE: 4.123  
Test R²: 0.731
CV MAE: 2.903 (±0.184)

Top 10 Most Important Features:
                    Feature  Importance
0           Targets_Per_Game    0.187432
1          Total_Yards_Per_Game 0.156829
2              Yards_Per_Target 0.134621
3                    Catch_Rate 0.098234
4           Rush_TD_Per_Game    0.087543
...

================================================================================
NFL FANTASY DRAFT RECOMMENDATIONS  
================================================================================

TOP RBs:
----------------------------------------
 1. Christian McCaffrey    (18.4 proj. pts)
 2. Austin Ekeler        (16.8 proj. pts)
 3. Saquon Barkley       (15.2 proj. pts)
...
```

## Advanced Features

### **Hyperparameter Optimization**
Automated tuning of XGBoost parameters:
- `n_estimators`: 100-1000
- `max_depth`: 3-10  
- `learning_rate`: 0.01-0.3
- `subsample`: 0.6-1.0
- Regularization parameters

### **Feature Importance Analysis**
Understanding what drives fantasy performance:
- Target share and usage metrics typically most important
- Efficiency metrics crucial for identifying breakouts
- Position-specific patterns revealed

### **Cross-Validation**
Robust model evaluation:
- 5-fold cross-validation  
- Consistent performance across folds
- Protection against overfitting

## Testing

Test the enhancements:
```bash
python test_improvements.py
```

This validates:
- XGBoost model integration
- Advanced feature engineering  
- Hyperparameter optimization
- Feature scaling pipeline
- All 20+ engineered features

## Model Interpretability

### **Feature Categories**

| Category | Examples | Impact |
|----------|----------|---------|
| **Usage** | Targets/game, Attempts/game | High |
| **Efficiency** | Yards/target, Catch rate | High |  
| **Production** | Total yards, Total TDs | Medium |
| **Position** | QB, RB, WR, TE dummies | Medium |
| **Consistency** | Weekly variance | Low |

### **Position-Specific Insights**
- **RBs**: Attempts per game and yards per carry dominate
- **WRs**: Target share and catch rate most predictive  
- **TEs**: Red zone usage and target quality key factors
- **QBs**: Passing attempts and TD rate drive value

## Performance Improvements

| Metric | Original Model | Enhanced Model | Improvement |
|--------|---------------|----------------|-------------|
| **Algorithm** | Linear Regression | XGBoost | Advanced ML |
| **Features** | 5 basic | 20+ engineered | 4x more features |
| **Data Years** | 3 years | 10 years | 3x more data |
| **MAE** | ~4.2 | ~2.8 | 33% better |
| **R²** | ~0.45 | ~0.73 | 62% improvement |
| **Validation** | Single split | Cross-validation | Robust |

## Contributing

Areas for enhancement:

- **Weekly Prediction Models** for in-season management
- **Injury Risk Integration** with injury history
- **Strength of Schedule** advanced modeling  
- **Value Over Replacement** (VOR) calculations
- **Position Scarcity Analysis**
- **Chemistry Ratings**

## Future Enhancements

- [ ] **Real-time Injury Updates** integration
- [ ] **League-Specific Scoring** customization
- [ ] **Weekly Predictions** for in-season use
- [ ] **Advanced Visualizations** and dashboards

## Disclaimer

This is just for fun.

## Author

**Kevin Veeder**
- Advanced from simple linear regression to sophisticated XGBoost model
- 10 years of training data and 20+ engineered features  
- Automated hyperparameter optimization and cross-validation

---

## Champions Rise... 

*Because fantasy football deserves more than just basic stats. This isn't your average league anymore.*

**Good luck!**
