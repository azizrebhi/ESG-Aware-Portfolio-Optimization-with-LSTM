import yfinance as yf
import pandas as pd
import numpy as np

def load_price_data(tickers, start="2018-01-01", end="2024-12-31"):
    data = yf.download(tickers, start=start, end=end)['Close']
    data = data.dropna(how="all").fillna(method="ffill").fillna(method="bfill")
    return data

def get_esg_scores(tickers):
    esg_scores = {}
    for ticker in tickers:
        info = yf.Ticker(ticker).sustainability
        if info is not None:
            if isinstance(info, pd.DataFrame) and "Value" in info.columns and "totalEsg" in info.index:
                esg_value = info.loc["totalEsg", "Value"]
            elif isinstance(info, pd.Series) and "totalEsg" in info.index:
                esg_value = info["totalEsg"]
            else:
                esg_value = np.nan
        else:
            esg_value = np.nan

        if pd.isna(esg_value):
            esg_value = np.random.randint(60, 90)
        esg_scores[ticker] = float(esg_value)
    
    return pd.Series(esg_scores)
