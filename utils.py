import pandas as pd


RF_PRICES = [
    "STK_SBER", "STK_GAZP", "STK_LKOH", "STK_GMKN", "STK_ROSN",
    "STK_NVTK", "STK_TATN", "STK_MGNT", "STK_MTSS", "STK_CHMF",
    "FX_USD", "FX_EUR", "BRENT",
]

RF_PRICES_TO_RETURNS = {
    "STK_SBER": "r_SBER",  "STK_GAZP": "r_GAZP", 
    "STK_LKOH": "r_LKOH",  "STK_GMKN": "r_GMKN", 
    "STK_ROSN": "r_ROSN",  "STK_NVTK": "r_NVTK",
    "STK_TATN": "r_TATN",  "STK_MGNT": "r_MGNT",
    "STK_MTSS": "r_MTSS",  "STK_CHMF": "r_CHMF",
    "FX_USD": "r_USD",    "FX_EUR": "r_EUR",
    "BRENT": "r_BRENT",
}

RF_RETURNS_TO_PRICES = {
    "r_SBER": "STK_SBER",  "r_GAZP": "STK_GAZP", 
    "r_LKOH": "STK_LKOH",  "r_GMKN": "STK_GMKN", 
    "r_ROSN": "STK_ROSN",  "r_NVTK": "STK_NVTK",
    "r_TATN": "STK_TATN",  "r_MGNT": "STK_MGNT",
    "r_MTSS": "STK_MTSS",  "r_CHMF": "STK_CHMF",
    "r_USD": "FX_USD",    "r_EUR": "FX_EUR",
    "r_BRENT": "BRENT",
}


def prices_to_returns(df_prices, columns, dropna=True):
    df = df_prices.copy()
    df[columns] = df[columns].pct_change()

    if dropna:
        df = df.dropna(axis=0)
    
    return df


def returns_to_prices(df_returns, base_prices, columns=None):
    returns = df_returns.copy()
    returns = returns[columns] if columns is not None else returns

    base_prices = base_prices.copy()
    base_prices = base_prices[columns] if columns is not None else base_prices

    missing = returns.columns.difference(base_prices.index)

    if len(missing):
        raise ValueError(f"Нет базовых цен для: {list(missing)}")

    base_prices = base_prices.reindex(returns.columns)
    growth = (1 + returns).cumprod()
    prices = growth.mul(base_prices, axis=1)

    return prices
