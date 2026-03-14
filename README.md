# Trader Behavior Insights

This project analyzes the relationship between Hyperliquid trader performance and Bitcoin Fear and Greed sentiment.

## Files in this project

- `app.py`: Streamlit dashboard for data exploration and insights
- `fear_greed_index.csv`: Sentiment dataset
- `historical_data.csv`: Hyperliquid historical trader data
- `requirements/requirements.txt`: Python dependencies list

## What the app does

1. Reads both CSV files directly from this folder.
2. Normalizes column names and data types.
3. Parses trade timestamps and aligns each trade to sentiment by date.
4. Computes key performance metrics such as:
   - trade count
   - unique accounts
   - total and average closed PnL
   - win rate
   - traded volume (USD)
5. Shows interactive analytics by:
   - sentiment regime (Fear, Greed, etc.)
   - date range
   - trade side
   - account filter
6. Displays charts and leaderboard-style account performance.

## Run locally

From this folder, run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- No dependencies were installed by this project setup step.
- All required packages are listed in `requirements.txt` for you to install.
