import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import ta
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AAPL Volatility Forecaster",
    page_icon   = "📈",
    layout      = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0a0e1a; }

    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }

    .metric-label {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 6px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        color: #f9fafb;
    }

    .risk-low    { color: #10b981; border-color: #10b981; }
    .risk-medium { color: #f59e0b; border-color: #f59e0b; }
    .risk-high   { color: #ef4444; border-color: #ef4444; }

    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        border: 1.5px solid;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .section-title {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 1rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .stButton > button {
        background: #1d4ed8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 1px;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #2563eb;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1f2937;
        font-size: 13px;
    }

    .info-key   { color: #6b7280; font-family: 'IBM Plex Mono', monospace; }
    .info-val   { color: #e5e7eb; font-weight: 600; }

    h1 { font-family: 'IBM Plex Mono', monospace !important; color: #f9fafb !important; }
    h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)


# ── Hyperparameters (must match your trained model) ───────────────────────────
W            = 90
HIDDEN_SIZE  = 128
NUM_LAYERS   = 1
DROPOUT      = 0.29255451180333386
N_FEATURES   = 17

FEATURES = [
    "return", "lag1", "lag5", "lag10",
    "ma10", "ma50", "price_vs_ma50", "high_low_range",
    "volatility5", "volatility10", "volatility20", "volatility60",
    "vol_ratio", "rsi", "macd", 'volume_change', "volume_ratio"
]

MODEL_PATH  = "models/model_weights5.pth"
SCALER_PATH = "models/scaler1.pkl"


# ── Model definition ─────────────────────────
class StockLSTM(nn.Module):
    """
    LSTM model for stock return prediction.
 
    Architecture:
        Input  → LSTM layers → Dropout → Linear → Output
 
    Input shape:  (batch_size, W, n_features)
    Output shape: (batch_size, 1)
    """
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super(StockLSTM, self).__init__()
 
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0,  # dropout only between layers
            batch_first  = True    # input shape: (batch, seq, features)
        )
 
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        # x shape: (batch_size, W, n_features)
 
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, W, hidden_size)
 
        # take only the last timestep's output
        # because we want to predict the next day after the sequence ends
        last_out = lstm_out[:, -1, :]
        # last_out shape: (batch_size, hidden_size)
 
        out = self.dropout(last_out)
        out = self.linear(out)
        out = torch.relu(out)
        # out shape: (batch_size, 1)
 
        return out.squeeze(1)   # shape: (batch_size,)


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV dataframe from yfinance and returns
    a clean dataframe with all 10 features + target.
    """
    out = pd.DataFrame(index=df.index)
 
    # 1. log return
    out["return"]   = df["Close"].pct_change()
 
    # 2. lag1 — return of yesterday
    out["lag1"]         = out["return"].shift(1)
 
    # 3. lag5 — return of exactly 5 days ago
    out["lag5"]         = out["return"].shift(5)

    out["lag10"]         = out["return"].shift(10)
    
    # 4. ma10 — 10-day moving average of price
    out["ma10"]         = df["Close"].rolling(10).mean()
 
    # 5. ma50 — 50-day moving average of price
    out["ma50"]         = df["Close"].rolling(50).mean()

    out['price_vs_ma50'] = (df['Close'] - out['ma50']) / out['ma50']

    out['high_low_range'] = (df['High'] - df['Low']) / df['Close']

    out["volatility5"] = out["return"].rolling(5).std()

    out["volatility10"] = out["return"].rolling(10).std()
 
    # 6. volatility20 — 20-day rolling std of log returns
    out["volatility20"] = out["return"].rolling(20).std()

    out["volatility60"] = out["return"].rolling(60).std()

    out['vol_ratio'] = out['volatility5'] / out['volatility20']
 
    # 8. rsi — relative strength index (14 days)
    out["rsi"]          = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
 
    # 9. macd — EMA(12) - EMA(26)
    out["macd"] = ta.trend.MACD(df["Close"]).macd()

    out['volume_change'] = df['Volume'].pct_change()
 
    # 10. volume_ratio — today's volume / 20-day average volume
    out["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
 
    # drop any rows with NaN (warmup period)
    out = out.dropna()
 
    return out


# ── Load model and scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model = StockLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ── Fetch and predict ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_and_predict(history_days):
    # download enough data for features + window
    
    success = True

    try:
        raw = yf.download("AAPL", period="3y", auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("Download returned empty data (no internet?)")
        raw.columns = raw.columns.get_level_values(0)
    except:
        success = False

    if success:
        raw.to_csv('cached.csv', encoding='utf-8')
    else:
        raw = pd.read_csv('cached.csv', index_col=0, parse_dates=True)
        print(f"Download failed, using cached data.")

    raw = raw.ffill()

    features_df = compute_features(raw)

    if len(features_df) < W:
        return None, None, None, None, None

    model, scaler = load_model_and_scaler()

    # get last W days as input
    window     = features_df[FEATURES].values[-W:]
    window_scaled = scaler.transform(window)
    X = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_vol = model(X).item()

    # historical volatility for chart
    hist_df    = features_df[["volatility20"]].tail(history_days)
    hist_mean  = features_df["volatility20"].mean()
    last_close = raw["Close"].iloc[-1]
    last_date  = features_df.index[-1]

    return predicted_vol, hist_df, hist_mean, last_close, last_date


# ── Risk level ────────────────────────────────────────────────────────────────
def get_risk(predicted_vol, hist_mean):
    ratio = predicted_vol / hist_mean
    if ratio < 0.85:
        return "LOW", "risk-low", "#10b981"
    elif ratio < 1.25:
        return "MEDIUM", "risk-medium", "#f59e0b"
    else:
        return "HIGH", "risk-high", "#ef4444"


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_volatility(hist_df, predicted_vol, hist_mean, risk_color):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    # historical volatility
    ax.plot(hist_df.index, hist_df["volatility20"],
            color="#3b82f6", linewidth=1.5, label="Historical volatility", zorder=3)

    # fill under curve
    ax.fill_between(hist_df.index, hist_df["volatility20"],
                    alpha=0.1, color="#3b82f6")

    # historical mean line
    ax.axhline(hist_mean, color="#4b5563", linewidth=1,
               linestyle="--", label=f"Historical mean ({hist_mean:.4f})")

    # predicted point
    next_date = hist_df.index[-1] + timedelta(days=1)
    ax.scatter([next_date], [predicted_vol],
               color=risk_color, s=120, zorder=5, label=f"Tomorrow's prediction ({predicted_vol:.4f})")
    ax.plot([hist_df.index[-1], next_date],
            [hist_df["volatility20"].iloc[-1], predicted_vol],
            color=risk_color, linewidth=1.5, linestyle="--", zorder=4)

    # styling
    ax.set_xlabel("Date", color="#6b7280", fontsize=10)
    ax.set_ylabel("Volatility", color="#6b7280", fontsize=10)
    ax.tick_params(colors="#6b7280")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")
    ax.legend(facecolor="#1f2937", edgecolor="#374151",
              labelcolor="#9ca3af", fontsize=9)
    ax.grid(axis="y", color="#1f2937", linewidth=0.8)
    plt.tight_layout()
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown("# 📈 AAPL Volatility Forecaster")
st.markdown(
    "<p style='color:#6b7280; font-size:14px; margin-top:-10px;'>"
    "LSTM-based next-day volatility prediction for Apple Inc. (AAPL)"
    "</p>", unsafe_allow_html=True
)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    history_days = st.slider(
        "Historical chart window",
        min_value = 30,
        max_value = 365,
        value     = 180,
        step      = 30,
        format    = "%d days"
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Tomorrow's Volatility")

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown(f"""
    <div class='info-row'><span class='info-key'>Architecture</span><span class='info-val'>LSTM</span></div>
    <div class='info-row'><span class='info-key'>Hidden size</span><span class='info-val'>{HIDDEN_SIZE}</span></div>
    <div class='info-row'><span class='info-key'>Layers</span><span class='info-val'>{NUM_LAYERS}</span></div>
    <div class='info-row'><span class='info-key'>Window (W)</span><span class='info-val'>{W} days</span></div>
    <div class='info-row'><span class='info-key'>Features</span><span class='info-val'>{N_FEATURES}</span></div>
    <div class='info-row'><span class='info-key'>Relative error</span><span class='info-val'>9.6%</span></div>
    <div class='info-row'><span class='info-key'>vs GARCH</span><span class='info-val'>✅ Better</span></div>
    <div class='info-row'><span class='info-key'>vs Persistence</span><span class='info-val'>✅ Better</span></div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Fetching latest AAPL data and running prediction..."):
        predicted_vol, hist_df, hist_mean, last_close, last_date = fetch_and_predict(history_days)

    if predicted_vol is None:
        st.error("Not enough data to make a prediction. Please try again.")
    else:
        risk_label, risk_class, risk_color = get_risk(predicted_vol, hist_mean)
        ratio = predicted_vol / hist_mean
        vs_mean = ((predicted_vol - hist_mean) / hist_mean) * 100

        # ── metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Predicted Volatility</div>
                <div class='metric-value'>{predicted_vol:.4f}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Risk Level</div>
                <div class='metric-value'>
                    <span class='risk-badge {risk_class}'>{risk_label}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col3:
            sign = "+" if vs_mean >= 0 else ""
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>vs Historical Mean</div>
                <div class='metric-value' style='color: {"#ef4444" if vs_mean > 0 else "#10b981"};'>
                    {sign}{vs_mean:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Last Close Price</div>
                <div class='metric-value'>${last_close:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── chart
        st.markdown("<div class='section-title'>Volatility History + Tomorrow's Forecast</div>",
                    unsafe_allow_html=True)
        fig = plot_volatility(hist_df, predicted_vol, hist_mean, risk_color)
        st.pyplot(fig)

        # ── interpretation
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Interpretation</div>", unsafe_allow_html=True)

        if risk_label == "LOW":
            msg = f"Tomorrow's predicted volatility ({predicted_vol:.4f}) is **{abs(vs_mean):.1f}% below** the historical average. The market is expected to be relatively calm. This suggests lower risk for AAPL positions."
        elif risk_label == "MEDIUM":
            msg = f"Tomorrow's predicted volatility ({predicted_vol:.4f}) is **near the historical average** ({hist_mean:.4f}). Normal market conditions are expected."
        else:
            msg = f"Tomorrow's predicted volatility ({predicted_vol:.4f}) is **{vs_mean:.1f}% above** the historical average. Elevated risk is expected. Consider reducing exposure or hedging AAPL positions."

        st.info(msg)

        st.markdown(f"<p style='color:#4b5563; font-size:11px;'>Last data point: {last_date.date()} · Model trained on AAPL 20010-2026 · For educational purposes only</p>",
                    unsafe_allow_html=True)

else:
    # placeholder before prediction
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #374151;'>
        <div style='font-size: 48px; margin-bottom: 1rem;'>📊</div>
        <div style='font-size: 16px; color: #6b7280;'>
            Click <b style='color:#3b82f6;'>Predict Tomorrow's Volatility</b> in the sidebar to get started.
        </div>
        <div style='font-size: 13px; color: #4b5563; margin-top: 0.5rem;'>
            The app will automatically fetch the latest AAPL data from Yahoo Finance.
        </div>
    </div>
    """, unsafe_allow_html=True)