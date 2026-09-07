"""
Pull real historical data for the 5-asset portfolio universe.
---------------------------------------------------------------------------
Run this on your OWN machine (not inside Claude's sandbox — Yahoo Finance
isn't on its allowed network list). Requires:

    pip install yfinance pandas

Usage:
    python pull_portfolio_data.py

Output (flat files, easy to download individually from Colab's file browser
on the left sidebar — right-click each -> Download):
    tech_training_data.csv        /  tech_testing_data.csv
    utility_training_data.csv     /  utility_testing_data.csv
    energy_training_data.csv      /  energy_testing_data.csv
    healthcare_training_data.csv  /  healthcare_testing_data.csv
    bond_like_training_data.csv   /  bond_like_testing_data.csv

Each CSV matches the format your existing pipeline already reads:
    Date, Close, Volume, Vix Price, Pct Change

Swap TICKERS below for whichever real names you actually want to represent
each category — these are just reasonable stand-ins.
"""
import pandas as pd
import yfinance as yf

# One real ticker per asset in the portfolio universe.
# Swap these for whatever you actually want the 5 assets to represent.
TICKERS = {
    "Tech": "NVDA",
    "Utility": "XLU",       # Utilities Select Sector SPDR ETF
    "Energy": "XLE",        # Energy Select Sector SPDR ETF
    "Healthcare": "XLV",    # Health Care Select Sector SPDR ETF
    "Bond-like": "BND",     # Vanguard Total Bond Market ETF
}

YEARS_OF_HISTORY = 5
TRAIN_FRACTION = 0.8

def pull_vix(period_years: int) -> pd.DataFrame:
    vix = yf.download("^VIX", period=f"{period_years}y", auto_adjust=True)
    vix = vix[["Close"]].rename(columns={"Close": "Vix Price"})
    vix.index = vix.index.date
    return vix

def pull_asset(ticker: str, vix_df: pd.DataFrame, period_years: int) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{period_years}y", auto_adjust=True)
    df = raw[["Close", "Volume"]].copy()
    df.index = df.index.date
    df["Pct Change"] = df["Close"].pct_change() * 100
    df = df.join(vix_df, how="inner")
    df = df.dropna().reset_index().rename(columns={"index": "Date"})
    df.columns = ["Date", "Close", "Volume", "Pct Change", "Vix Price"]
    df = df[["Date", "Close", "Volume", "Vix Price", "Pct Change"]]
    return df

def main():
    print("Pulling VIX...")
    vix_df = pull_vix(YEARS_OF_HISTORY)

    for asset, ticker in TICKERS.items():
        print(f"\nPulling {asset} ({ticker})...")
        df = pull_asset(ticker, vix_df, YEARS_OF_HISTORY)

        split = int(len(df) * TRAIN_FRACTION)
        train, test = df.iloc[:split], df.iloc[split:]

        name = asset.lower().replace(" ", "_").replace("-", "_")
        train.to_csv(f"{name}_training_data.csv", index=False)
        test.to_csv(f"{name}_testing_data.csv", index=False)

        print(f"  {len(train)} training days, {len(test)} testing days")
        print(f"  Latest date pulled: {df['Date'].iloc[-1]}")
        print(f"  Saved: {name}_training_data.csv, {name}_testing_data.csv")

    print("\nDone. In Colab: click the folder icon on the left sidebar,")
    print("right-click each CSV -> Download, then upload all 10 files here.")

if __name__ == "__main__":
    main()
