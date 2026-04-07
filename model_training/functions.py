import pandas as pd
import ta
import yfinance as yf
import numpy as np
from sklearn.preprocessing import RobustScaler

# feature enginering
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


# download and process all ticker
def load_ticker(ticker: str, period: str) -> pd.DataFrame:
    """
    Downloads raw data and computes features for each ticker.
    Returns a dict of {ticker: feature_dataframe}.
    """
    print(f"Downloading {ticker}...", end=" ")
 
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=True)
    raw.columns = raw.columns.get_level_values(0)   # flatten MultiIndex
 
    # forward fill any missing days (trading halts etc.)
    raw = raw.ffill()
 
    df = compute_features(raw)
 
    return df


# split / test data
def split_data(data: pd.DataFrame, train_pct: float) -> tuple:
    """
    Splits each ticker's dataframe chronologically into train and test.
    Returns two dicts: train_data and test_data.
    """
    split = int(len(data) * train_pct)
    train_data = data.iloc[:split]
    test_data = data.iloc[split:]
 
    print(f"train={len(train_data)} rows, "
        f"test={len(test_data)} rows, "
        f"split date={data.index[split].date()}")
        
 
    return train_data, test_data



# sliding window
# it's making the instances
def make_windows(df: pd.DataFrame, W: int, features:list, target:str) -> tuple:
    """
    Converts a feature dataframe into sliding window instances.
 
    For each position i, we take:
        X = features of days [i → i+W-1]   shape: (W, n_features)
        y = target of day    [i+W]          shape: (1,)
 
    Returns:
        X: np.array of shape (n_instances, W, n_features)
        y: np.array of shape (n_instances,)
    """
    feature_vals = df[features].values   # shape: (n_days, n_features)
    target_vals  = df[target].values     # shape: (n_days,)
 
    X, y = [], []
 
    for i in range(len(df) - W):
        X.append(feature_vals[i : i + W])   # W days of features
        y.append(target_vals[i + W])         # next day's target

    print(f'number of instances : {len(y)}')
 
    return np.array(X), np.array(y)



# fitting the scaler
def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    """
    Fits a RobustScaler on the training data only.
 
    X_train shape: (n_instances, W, n_features)
    We reshape to (n_instances * W, n_features) to fit the scaler,
    then reshape back.
    """

    _, _, n_features = X_train.shape
 
    # flatten instances and timesteps into one dimension
    X_flat = X_train.reshape(-1, n_features)
 
    scaler = RobustScaler()
    scaler.fit(X_flat)
 
    return scaler


#scaling
def apply_scaler(X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    """
    Applies a fitted scaler to X.
    Handles the reshape automatically.
    """
    n_instances, w, n_features = X.shape
    X_flat   = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n_instances, w, n_features)