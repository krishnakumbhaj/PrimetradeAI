from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Trader Behavior Insights", layout="wide")


DATA_DIR = Path(__file__).resolve().parent
SENTIMENT_FILE = DATA_DIR / "fear_greed_index.csv"
TRADES_FILE = DATA_DIR / "historical_data.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def safe_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def infer_trade_datetime(trades: pd.DataFrame) -> pd.Series:
    trade_datetime = pd.Series(pd.NaT, index=trades.index, dtype="datetime64[ns]")

    if "timestamp_ist" in trades.columns:
        trade_datetime = pd.to_datetime(
            trades["timestamp_ist"], dayfirst=True, errors="coerce"
        )

    if trade_datetime.notna().sum() == 0 and "timestamp" in trades.columns:
        ts_num = pd.to_numeric(trades["timestamp"], errors="coerce")
        if ts_num.notna().sum() > 0:
            unit = "ms" if ts_num.dropna().median() > 1e11 else "s"
            trade_datetime = pd.to_datetime(ts_num, unit=unit, errors="coerce")

    return trade_datetime


@st.cache_data(show_spinner=False)
def load_and_prepare_data(
    sentiment_path: Path, trades_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sentiment_raw = pd.read_csv(sentiment_path)
    trades_raw = pd.read_csv(trades_path)

    sentiment = normalize_columns(sentiment_raw)
    trades = normalize_columns(trades_raw)

    required_sentiment_columns = {"date", "classification"}
    missing_sentiment = required_sentiment_columns - set(sentiment.columns)
    if missing_sentiment:
        raise ValueError(
            f"Missing expected sentiment columns: {sorted(missing_sentiment)}"
        )

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["classification"] = sentiment["classification"].astype(str).str.strip()
    if "value" in sentiment.columns:
        sentiment["value"] = pd.to_numeric(sentiment["value"], errors="coerce")

    sentiment = sentiment.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    sentiment["trade_date"] = sentiment["date"].dt.normalize()

    trades = safe_to_numeric(
        trades,
        [
            "execution_price",
            "size_tokens",
            "size_usd",
            "start_position",
            "closed_pnl",
            "fee",
            "leverage",
        ],
    )

    if "side" not in trades.columns and "direction" in trades.columns:
        trades["side"] = trades["direction"]

    if "side" in trades.columns:
        trades["side"] = trades["side"].astype(str).str.upper().str.strip()

    trades["trade_datetime"] = infer_trade_datetime(trades)
    trades = trades.dropna(subset=["trade_datetime"]).copy()
    trades["trade_date"] = trades["trade_datetime"].dt.normalize()

    sentiment_for_merge = sentiment[["trade_date", "classification"]].copy()
    if "value" in sentiment.columns:
        sentiment_for_merge["sentiment_value"] = sentiment["value"]

    merged = trades.merge(sentiment_for_merge, on="trade_date", how="left")
    merged["classification"] = merged["classification"].fillna("Unknown")

    return sentiment, trades, merged


def compute_win_rate(values: pd.Series) -> float:
    clean = values.dropna()
    wins = (clean > 0).sum()
    losses = (clean < 0).sum()
    decisions = wins + losses
    if decisions == 0:
        return np.nan
    return (wins / decisions) * 100


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}%"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 18%, #fff7db 0%, #f7f9ff 40%, #eef3fb 100%);
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Trader Behavior Insights")
st.caption("Hyperliquid historical trades merged with Bitcoin Fear and Greed sentiment")

if not SENTIMENT_FILE.exists() or not TRADES_FILE.exists():
    st.error(
        "Missing required files. Please keep both CSVs in the same folder as app.py: "
        "fear_greed_index.csv and historical_data.csv"
    )
    st.stop()

try:
    sentiment_df, trades_df, merged_df = load_and_prepare_data(SENTIMENT_FILE, TRADES_FILE)
except Exception as exc:
    st.exception(exc)
    st.stop()

if merged_df.empty:
    st.warning("Merged dataset is empty after parsing dates. Please verify CSV formats.")
    st.stop()

with st.sidebar:
    st.header("Filters")

    min_date = merged_df["trade_date"].min().date()
    max_date = merged_df["trade_date"].max().date()

    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
    else:
        start_date = pd.Timestamp(min_date)
        end_date = pd.Timestamp(max_date)

    sentiment_options = sorted(merged_df["classification"].dropna().unique().tolist())
    selected_sentiments = st.multiselect(
        "Sentiment class",
        options=sentiment_options,
        default=sentiment_options,
    )

    side_options = (
        sorted(merged_df["side"].dropna().unique().tolist()) if "side" in merged_df.columns else []
    )
    selected_sides = st.multiselect("Side", options=side_options, default=side_options)

    account_search = st.text_input("Account contains", value="")

filtered = merged_df[
    (merged_df["trade_date"] >= start_date)
    & (merged_df["trade_date"] <= end_date)
    & (merged_df["classification"].isin(selected_sentiments))
].copy()

if "side" in filtered.columns and selected_sides:
    filtered = filtered[filtered["side"].isin(selected_sides)]

if "account" in filtered.columns and account_search.strip():
    filtered = filtered[
        filtered["account"].astype(str).str.contains(account_search.strip(), case=False, na=False)
    ]

if filtered.empty:
    st.warning("No rows match current filters.")
    st.stop()

has_pnl = "closed_pnl" in filtered.columns
has_size = "size_usd" in filtered.columns

total_trades = len(filtered)
unique_accounts = filtered["account"].nunique() if "account" in filtered.columns else np.nan

if has_pnl:
    total_pnl = filtered["closed_pnl"].sum(min_count=1)
    avg_pnl = filtered["closed_pnl"].mean()
    overall_win_rate = compute_win_rate(filtered["closed_pnl"])
else:
    total_pnl = np.nan
    avg_pnl = np.nan
    overall_win_rate = np.nan

total_volume = filtered["size_usd"].sum(min_count=1) if has_size else np.nan

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Trades", f"{total_trades:,}")
m2.metric("Accounts", f"{int(unique_accounts):,}" if pd.notna(unique_accounts) else "N/A")
m3.metric("Total Closed PnL", money(total_pnl))
m4.metric("Avg Closed PnL", money(avg_pnl))
m5.metric("Win Rate", pct(overall_win_rate))
m6.metric("Volume (USD)", money(total_volume))

if has_pnl:
    sentiment_perf = (
        filtered.groupby("classification", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "trades": len(g),
                    "total_closed_pnl": g["closed_pnl"].sum(min_count=1),
                    "avg_closed_pnl": g["closed_pnl"].mean(),
                    "win_rate_pct": compute_win_rate(g["closed_pnl"]),
                    "avg_size_usd": g["size_usd"].mean() if "size_usd" in g.columns else np.nan,
                }
            )
        )
        .reset_index()
        .sort_values("total_closed_pnl", ascending=False, na_position="last")
    )

    top_regime = sentiment_perf.iloc[0]
    bottom_regime = sentiment_perf.iloc[-1]

    st.markdown(
        f"Most profitable regime by total PnL: **{top_regime['classification']}** "
        f"({money(top_regime['total_closed_pnl'])}). Least profitable: "
        f"**{bottom_regime['classification']}** ({money(bottom_regime['total_closed_pnl'])})."
    )

    c1, c2 = st.columns(2)

    with c1:
        fig_bar = px.bar(
            sentiment_perf,
            x="classification",
            y="avg_closed_pnl",
            color="classification",
            title="Average Closed PnL by Sentiment",
            color_discrete_sequence=["#0b7285", "#f08c00", "#2f9e44", "#c92a2a", "#495057"],
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        pnl_dist = filtered.dropna(subset=["closed_pnl"]).copy()
        fig_box = px.box(
            pnl_dist,
            x="classification",
            y="closed_pnl",
            color="classification",
            title="Closed PnL Distribution by Sentiment",
            points=False,
            color_discrete_sequence=["#0b7285", "#f08c00", "#2f9e44", "#c92a2a", "#495057"],
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Sentiment Performance Table")
    st.dataframe(
        sentiment_perf.style.format(
            {
                "total_closed_pnl": "${:,.2f}",
                "avg_closed_pnl": "${:,.2f}",
                "win_rate_pct": "{:.2f}%",
                "avg_size_usd": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )

    daily_perf = (
        filtered.groupby("trade_date", as_index=False)
        .agg(
            daily_trades=("trade_date", "size"),
            daily_total_pnl=("closed_pnl", "sum"),
            daily_avg_pnl=("closed_pnl", "mean"),
            daily_sentiment_value=("sentiment_value", "mean")
            if "sentiment_value" in filtered.columns
            else ("trade_date", "size"),
        )
        .sort_values("trade_date")
    )

    if "sentiment_value" not in filtered.columns:
        daily_perf["daily_sentiment_value"] = np.nan

    c3, c4 = st.columns(2)
    with c3:
        fig_daily_pnl = px.line(
            daily_perf,
            x="trade_date",
            y="daily_total_pnl",
            markers=True,
            title="Daily Total Closed PnL",
        )
        st.plotly_chart(fig_daily_pnl, use_container_width=True)

    with c4:
        if daily_perf["daily_sentiment_value"].notna().sum() > 0:
            fig_sentiment = px.line(
                daily_perf,
                x="trade_date",
                y="daily_sentiment_value",
                markers=True,
                title="Daily Fear and Greed Value",
                color_discrete_sequence=["#f08c00"],
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("Sentiment value column not available for daily trend chart.")

    if "account" in filtered.columns:
        st.subheader("Top Traders by Closed PnL")
        top_n = st.slider("Number of traders", min_value=5, max_value=50, value=15)
        account_perf = (
            filtered.groupby("account", as_index=False)
            .agg(
                trades=("account", "size"),
                total_closed_pnl=("closed_pnl", "sum"),
                avg_closed_pnl=("closed_pnl", "mean"),
                win_rate_pct=("closed_pnl", compute_win_rate),
            )
            .sort_values("total_closed_pnl", ascending=False)
            .head(top_n)
        )

        st.dataframe(
            account_perf.style.format(
                {
                    "total_closed_pnl": "${:,.2f}",
                    "avg_closed_pnl": "${:,.2f}",
                    "win_rate_pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )

st.subheader("Merged Dataset Preview")
st.dataframe(filtered.head(200), use_container_width=True)
