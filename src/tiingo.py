import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import (
    RAW_DATA_DIR,
    START_DATE,
    TICKERS,
    TIINGO_BASE_URL,
)


load_dotenv()


def get_api_key():
    """
    Read the Tiingo API key from the .env file.
    """

    api_key = os.getenv("TIINGO_API_KEY")

    if not api_key:
        raise ValueError(
            "TIINGO_API_KEY was not found. "
            "Add it to your .env file."
        )

    return api_key


def download_ticker(ticker, force=False):
    """
    Download daily historical prices for one ticker.

    Data is cached locally so we do not repeatedly
    request the same history from Tiingo.
    """

    output_file = RAW_DATA_DIR / f"{ticker}.csv"

    if output_file.exists() and not force:
        print(f"{ticker}: using cached data")
        return pd.read_csv(
            output_file,
            parse_dates=["date"]
        )

    print(f"{ticker}: downloading from Tiingo...")

    api_key = get_api_key()

    url = f"{TIINGO_BASE_URL}/{ticker}/prices"

    params = {
        "startDate": START_DATE,
        "token": api_key,
        "format": "json",
        "resampleFreq": "daily",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            f"No data returned for {ticker}"
        )

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True
    ).dt.tz_localize(None)

    df = df.sort_values("date")

    df.to_csv(output_file, index=False)

    # Be polite to the API
    time.sleep(0.15)

    return df


def download_all(force=False):
    """
    Download all requested stocks.
    """

    histories = {}

    for ticker in TICKERS:
        try:
            histories[ticker] = download_ticker(
                ticker,
                force=force
            )

        except Exception as exc:
            print(
                f"WARNING: could not download "
                f"{ticker}: {exc}"
            )

    return histories