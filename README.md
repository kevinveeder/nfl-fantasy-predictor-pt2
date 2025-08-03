# NFL Fantasy Football Draft Predictor & Analytics Tool

Please note, this is a *__simplified__* version. I'm not quite ready to share my private repo just yet... I mean, fantasy is a competition, right?

This project uses real 2022-2024 player statistics to train a linear regression model that predicts fantasy football performance for the upcoming 2025 season. It scrapes data from Pro Football Reference for historical stats, FantasyPros for current projections, and combines everything to generate full-season fantasy point predictions and highlight potential sleepers or overvalued players. This is done through custom web scraping functions and machine learning models that I had fun building and experimenting with.

## Quick Start:

```bash
# Clone the repository
git clone https://github.com/kevinveeder/nfl-fantasy-predictor-pt2.git
cd nfl-fantasy-predictor-pt2

# Install required packages
pip install pandas numpy scikit-learn requests beautifulsoup4 lxml

# Run the analysis
python nfl_fantasy_predictor.py
```

## Overview

• 2022-2024 Player Data scraped from Pro Football Reference and used to train a fantasy points per game (FPPG) prediction model

• Linear Regression Model trained using features like rushing attempts, pass targets, completions, and receptions

• 2025 Fantasy Projections scraped from FantasyPros for QBs, RBs, WRs, and TEs

• Model predictions combined with expert projections to identify value picks and draft targets

• Final Output generates ranked draft recommendations with projected fantasy points and model insights

## Features Used for Modeling:

• **Att** - Rushing attempts per game
• **Tgt** - Pass targets (for receivers and running backs)
• **Rec** - Receptions (catches)
• **Cmp** - Pass completions (for quarterbacks)  
• **Att.1** - Passing attempts (for quarterbacks)

These features were chosen because they represent player usage and opportunity, which typically correlates strongly with fantasy production regardless of efficiency.

## How It Works:

**Step 1: Historical Data Collection**
- Scrapes 2022-2024 NFL player stats from Pro Football Reference
- Cleans and processes the data, calculating FPPG (Fantasy Points Per Game)
- Combines multiple seasons for more robust model training
- Filters out players with minimal games played to improve model accuracy

**Step 2: Model Training**
- Uses scikit-learn's Linear Regression to find relationships between usage stats and fantasy points
- Splits data into training/testing sets to evaluate model performance
- Outputs model metrics like Mean Absolute Error and R² score

**Step 3: Current Projections Scraping**
- Scrapes expert projections from FantasyPros for all skill positions
- Handles complex HTML table structures and cleans player data
- Combines projections from multiple positions into unified dataset

**Step 4: Draft Analysis**
- Generates position-specific rankings based on projected fantasy points
- Creates a draft board showing top players at each position
- Exports results to CSV for easy reference during drafts

## Usage:

Once you run the last command in the quickstart, the script will automatically run through all steps and output a draft board plus save results to `fantasy_draft_recommendations.csv`.

## What You Get:

- **Draft Board** - Top players at each position with projected fantasy points
- **Model Insights** - Which stats matter most for fantasy success
- **Performance Metrics** - How accurate the model is at predicting fantasy points
- **CSV Export** - Portable draft rankings you can reference during your draft

## Customization:

You can easily modify the code to:
- Use different years of historical data
- Add or remove features from the model
- Scrape different positions or websites
- Adjust the ranking methodology

## Notes:

This is primarily a fun data science project to experiment with web scraping and machine learning in the context of fantasy football. The predictions should be used alongside other research and expert opinions, not as the sole basis for draft decisions. Past performance doesn't guarantee future results, and there are many variables (injuries, coaching changes, etc.) that aren't captured in basic usage statistics.

The web scraping functions include proper delays and headers to be respectful to the websites being scraped. Always make sure you're following a site's terms of service when scraping data.
