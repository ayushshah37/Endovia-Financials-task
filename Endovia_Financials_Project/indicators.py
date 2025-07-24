# indicators.py
import pandas as pd
import numpy as np

def compute_indicators(df):           
    # ESA (Exponential Smoothed Average)
    df['esa'] = df['close'].ewm(span=20, adjust=False).mean()       

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # CCI (Commodity Channel Index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['ci'] = cci

    # TCI (Trend Confirmation Index - derived from CCI)
    df['tci'] = df['ci'].ewm(span=5, adjust=False).mean()

    # WaveTrend Oscillator (wt1 and wt2)
    ap = (df['high'] + df['low']) / 2
    esa = ap.ewm(span=10, adjust=False).mean()
    d = abs(ap - esa).ewm(span=10, adjust=False).mean()
    ci_wave = (ap - esa) / (0.015 * d)
    wt1 = ci_wave.ewm(span=21, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    df['wt1'] = wt1
    df['wt2'] = wt2

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # EMA Fast and Slow
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    print(df[['rsi', 'macd', 'macd_signal', 'ema_fast', 'ema_slow']].head(10)) 
    return df

