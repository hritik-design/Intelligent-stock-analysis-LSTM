"""
Stock Market Analysis & LSTM Prediction — Streamlit App
Converted from: stock-market-analysis-prediction-using-lstm.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Analysis & LSTM Prediction",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — subtle polish
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header  { font-size: 2.2rem; font-weight: 700; color: #1f77b4; }
    .section-header { font-size: 1.4rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; }
    .metric-card  { background:#f0f4f8; border-radius:10px; padding:1rem; }
    .stAlert      { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading stock data …")
def load_stock_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for each ticker and return as a dict."""
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False)
        if not df.empty:
            df["company_name"] = t
            data[t] = df
    return data


@st.cache_data(show_spinner="Downloading closing prices …")
def load_closing_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    # yfinance returns MultiIndex when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Adj Close"]
    return raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})


def add_moving_averages(df: pd.DataFrame, windows=(10, 20, 50)) -> pd.DataFrame:
    for w in windows:
        df[f"MA {w}d"] = df["Adj Close"].rolling(w).mean()
    return df


def plot_closing_prices(stock_data: dict) -> plt.Figure:
    tickers = list(stock_data.keys())
    n = len(tickers)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()
    for i, (ticker, df) in enumerate(stock_data.items()):
        axes[i].plot(df.index, df["Adj Close"], linewidth=1.5)
        axes[i].set_title(f"{ticker} — Closing Price", fontsize=12)
        axes[i].set_ylabel("Adj Close ($)")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def plot_volume(stock_data: dict) -> plt.Figure:
    tickers = list(stock_data.keys())
    n = len(tickers)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes = axes.flatten()
    for i, (ticker, df) in enumerate(stock_data.items()):
        axes[i].bar(df.index, df["Volume"], color="steelblue", alpha=0.7, width=1)
        axes[i].set_title(f"{ticker} — Volume", fontsize=12)
        axes[i].set_ylabel("Shares Traded")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def plot_moving_averages(stock_data: dict) -> plt.Figure:
    tickers = list(stock_data.keys())
    n = len(tickers)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes = axes.flatten()
    for i, (ticker, df) in enumerate(stock_data.items()):
        df_ma = add_moving_averages(df.copy())
        cols_to_plot = ["Adj Close"] + [c for c in df_ma.columns if c.startswith("MA")]
        df_ma[cols_to_plot].plot(ax=axes[i])
        axes[i].set_title(f"{ticker} — Moving Averages", fontsize=12)
        axes[i].set_ylabel("Price ($)")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


def plot_daily_returns(stock_data: dict) -> tuple[plt.Figure, plt.Figure]:
    tickers = list(stock_data.keys())
    n = len(tickers)

    fig1, axes1 = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes1 = axes1.flatten()
    for i, (ticker, df) in enumerate(stock_data.items()):
        df["Daily Return"] = df["Adj Close"].pct_change()
        axes1[i].plot(df.index, df["Daily Return"], linestyle="--", linewidth=0.8)
        axes1[i].set_title(f"{ticker} — Daily Return %", fontsize=12)
        axes1[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes1)):
        axes1[j].set_visible(False)
    plt.tight_layout()

    fig2, axes2 = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes2 = axes2.flatten()
    for i, (ticker, df) in enumerate(stock_data.items()):
        df["Daily Return"].hist(bins=50, ax=axes2[i], color="teal", edgecolor="white")
        axes2[i].set_title(f"{ticker} — Daily Return Distribution", fontsize=12)
        axes2[i].set_xlabel("Daily Return")
        axes2[i].set_ylabel("Count")
    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)
    plt.tight_layout()

    return fig1, fig2


def plot_correlation(closing_df: pd.DataFrame) -> plt.Figure:
    tech_rets = closing_df.pct_change().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(tech_rets.corr(), annot=True, cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Correlation — Daily Returns")
    sns.heatmap(closing_df.corr(), annot=True, cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Correlation — Closing Prices")
    plt.tight_layout()
    return fig


def plot_risk_return(closing_df: pd.DataFrame) -> plt.Figure:
    tech_rets = closing_df.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(tech_rets.mean(), tech_rets.std(), s=200, alpha=0.8, c="steelblue", edgecolors="k")
    for label, x, y in zip(tech_rets.columns, tech_rets.mean(), tech_rets.std()):
        ax.annotate(
            label, xy=(x, y), xytext=(40, 40),
            textcoords="offset points", ha="right",
            arrowprops=dict(arrowstyle="-", color="grey"),
        )
    ax.set_xlabel("Expected Daily Return")
    ax.set_ylabel("Risk (Std Dev)")
    ax.set_title("Risk vs. Return")
    plt.tight_layout()
    return fig


# ── LSTM helpers ──────────────────────────────

@st.cache_data(show_spinner="Downloading AAPL data for LSTM …")
def load_lstm_data() -> pd.DataFrame:
    df = yf.download("AAPL", start="2012-01-01",
                     end=datetime.now().strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=False, multi_level_index=False)
    return df


def run_lstm(df: pd.DataFrame, epochs: int = 1) -> dict:
    """Build, train, and evaluate the LSTM model. Returns result dict."""
    from sklearn.preprocessing import MinMaxScaler
    # Lazy import so app loads without TF if user skips this section
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    data = df.filter(["Close"])
    dataset = data.values
    training_len = int(np.ceil(len(dataset) * 0.95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)

    # Training set
    train = scaled[:training_len]
    x_tr, y_tr = [], []
    for i in range(60, len(train)):
        x_tr.append(train[i - 60:i, 0])
        y_tr.append(train[i, 0])
    x_tr, y_tr = np.array(x_tr), np.array(y_tr)
    x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], 1)

    # Model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_tr.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    progress = st.progress(0, text="Training LSTM …")
    for e in range(epochs):
        model.fit(x_tr, y_tr, batch_size=32, epochs=1, verbose=0)
        progress.progress((e + 1) / epochs, text=f"Epoch {e+1}/{epochs}")
    progress.empty()

    # Test set
    test = scaled[training_len - 60:]
    x_te = []
    y_te = dataset[training_len:]
    for i in range(60, len(test)):
        x_te.append(test[i - 60:i, 0])
    x_te = np.array(x_te).reshape(-1, 60, 1)

    preds = scaler.inverse_transform(model.predict(x_te))
    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))

    train_plot = data[:training_len]
    valid_plot = data[training_len:].copy()
    valid_plot["Predictions"] = preds

    return {
        "rmse": rmse,
        "train": train_plot,
        "valid": valid_plot,
    }


def plot_lstm_results(result: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(result["train"].index, result["train"]["Close"], label="Train", linewidth=1.5)
    ax.plot(result["valid"].index, result["valid"]["Close"], label="Actual", linewidth=1.5)
    ax.plot(result["valid"].index, result["valid"]["Predictions"],
            label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_title("AAPL — LSTM Price Prediction", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("📅 Date Range")
    default_end = datetime.today()
    default_start = default_end - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start)
    end_date   = st.date_input("End date",   value=default_end)

    st.markdown("---")
    st.subheader("🏢 Stocks to Analyse")
    default_tickers = ["AAPL", "GOOG", "MSFT", "AMZN"]
    selected_tickers = st.multiselect(
        "Select tickers",
        options=["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
        default=default_tickers,
    )
    custom_ticker = st.text_input("Or add a custom ticker (e.g. BABA)", "").upper()
    if custom_ticker:
        selected_tickers = list(set(selected_tickers + [custom_ticker]))

    st.markdown("---")
    st.subheader("🔮 LSTM Settings")
    lstm_epochs = st.slider("Training epochs", 1, 10, 1)

    st.markdown("---")
    st.caption("Data sourced from Yahoo Finance via yfinance.")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Stock Market Analysis & LSTM Prediction</p>',
            unsafe_allow_html=True)
st.markdown(
    "Analyse tech stocks with interactive charts, correlation heatmaps, "
    "risk/return scatter, and an LSTM-based price prediction for AAPL."
)

if not selected_tickers:
    st.warning("👈 Please select at least one ticker from the sidebar.")
    st.stop()

# ── Load data ────────────────────────────────
stock_data   = load_stock_data(selected_tickers, str(start_date), str(end_date))
closing_df   = load_closing_prices(selected_tickers, str(start_date), str(end_date))

if not stock_data:
    st.error("No data returned. Check the tickers or date range.")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📉 Moving Averages",
    "📈 Daily Returns",
    "🔗 Correlation",
    "⚠️ Risk vs Return",
    "🤖 LSTM Prediction",
])

# ── TAB 1 : Overview ─────────────────────────
with tab1:
    st.markdown('<p class="section-header">Descriptive Stats & Price / Volume Charts</p>',
                unsafe_allow_html=True)

    chosen = st.selectbox("Select stock for stats:", list(stock_data.keys()))
    df_chosen = stock_data[chosen]

    col1, col2, col3, col4 = st.columns(4)
    latest = df_chosen["Adj Close"].iloc[-1]
    earliest = df_chosen["Adj Close"].iloc[0]
    pct_chg = (latest - earliest) / earliest * 100
    col1.metric("Latest Adj Close", f"${latest:.2f}")
    col2.metric("Period Change", f"{pct_chg:+.2f}%")
    col3.metric("52-wk High", f"${df_chosen['High'].max():.2f}")
    col4.metric("Avg Daily Volume", f"{df_chosen['Volume'].mean():,.0f}")

    with st.expander("📋 Raw Data (last 10 rows)"):
        st.dataframe(df_chosen.tail(10))

    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df_chosen.describe().style.format("{:.4f}"))

    st.subheader("Closing Prices")
    st.pyplot(plot_closing_prices(stock_data))

    st.subheader("Trading Volume")
    st.pyplot(plot_volume(stock_data))

# ── TAB 2 : Moving Averages ───────────────────
with tab2:
    st.markdown('<p class="section-header">Moving Averages (10 / 20 / 50 days)</p>',
                unsafe_allow_html=True)
    st.info(
        "Moving averages smooth out short-term noise and help identify trend direction. "
        "The 10-day and 20-day MAs react faster; the 50-day MA shows the longer trend."
    )
    st.pyplot(plot_moving_averages(stock_data))

# ── TAB 3 : Daily Returns ─────────────────────
with tab3:
    st.markdown('<p class="section-header">Daily Return Analysis</p>',
                unsafe_allow_html=True)
    fig_line, fig_hist = plot_daily_returns(stock_data)
    st.subheader("Daily Return Over Time")
    st.pyplot(fig_line)
    st.subheader("Distribution of Daily Returns")
    st.pyplot(fig_hist)

# ── TAB 4 : Correlation ───────────────────────
with tab4:
    st.markdown('<p class="section-header">Correlation Between Stocks</p>',
                unsafe_allow_html=True)
    st.info(
        "A value close to **+1** means the stocks move together; "
        "close to **-1** means they move in opposite directions."
    )
    if len(selected_tickers) < 2:
        st.warning("Select at least 2 tickers to see a correlation heatmap.")
    else:
        st.pyplot(plot_correlation(closing_df))

        st.subheader("Pair Plot — Daily Returns")
        tech_rets = closing_df.pct_change().dropna()
        fig_pair = sns.pairplot(tech_rets, kind="reg", diag_kind="hist")
        st.pyplot(fig_pair.fig)

# ── TAB 5 : Risk vs Return ────────────────────
with tab5:
    st.markdown('<p class="section-header">Risk vs Expected Return</p>',
                unsafe_allow_html=True)
    st.info(
        "Each bubble represents a stock. **Higher on the Y-axis → more volatile (risky)**; "
        "**further right on the X-axis → higher average daily return**."
    )
    if len(selected_tickers) < 2:
        st.warning("Select at least 2 tickers to see the risk/return chart.")
    else:
        st.pyplot(plot_risk_return(closing_df))

# ── TAB 6 : LSTM Prediction ───────────────────
with tab6:
    st.markdown('<p class="section-header">LSTM Price Prediction for AAPL</p>',
                unsafe_allow_html=True)
    st.info(
        "This LSTM model uses **60 days of historical closing prices** to predict the next day's price. "
        "It trains on 95 % of AAPL data (from 2012) and validates on the remaining 5 %."
    )

    st.warning(
        "⏳ Training can take **1–3 minutes** even for 1 epoch because of the large AAPL dataset. "
        "Increase epochs for better accuracy (at the cost of more time)."
    )

    df_lstm = load_lstm_data()

    col_a, col_b = st.columns(2)
    col_a.metric("Total Trading Days", f"{len(df_lstm):,}")
    col_b.metric("Date Range", f"{df_lstm.index[0].date()} → {df_lstm.index[-1].date()}")

    st.subheader("AAPL Closing Price History")
    fig_hist_price, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_lstm.index, df_lstm["Close"], linewidth=1.2, color="steelblue")
    ax.set_title("AAPL Close Price (2012 – present)")
    ax.set_ylabel("Price ($)")
    plt.tight_layout()
    st.pyplot(fig_hist_price)

    if st.button("🚀 Train LSTM & Predict", type="primary"):
        try:
            result = run_lstm(df_lstm, epochs=lstm_epochs)
            st.success(f"✅ Training complete!  RMSE = **${result['rmse']:.2f}**")
            st.pyplot(plot_lstm_results(result))

            with st.expander("📋 Actual vs Predicted (last 20 rows)"):
                st.dataframe(result["valid"].tail(20).style.format("${:.2f}"))
        except ImportError:
            st.error(
                "Keras / TensorFlow not found. "
                "Install with:  `pip install tensorflow`"
            )
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · Data from Yahoo Finance · "
    "LSTM model via TensorFlow / Keras"
)