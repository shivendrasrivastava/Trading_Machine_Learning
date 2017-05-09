""" nzou3@gatech.edu """
"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def normalize(A):
    return (A - np.mean(A)) / np.std(A)


def indicator_df(sdate='2008-01-01', edate='2009-12-31', sym='AAPL', window=21, momentum_day=15):
    symbols = [sym]
    indi_df_all = pd.read_csv(symbol_to_path(symbols[0]), index_col='Date', parse_dates=True, na_values=['nan'])
    date_range = pd.date_range(sdate, edate)
    indi_df = pd.DataFrame(index=date_range)
    indi_df = indi_df.join(indi_df_all).dropna()

    indi_df['SMA'] = pd.Series.rolling(indi_df['Adj Close'], window=window, center=False).mean().fillna(method='bfill')
    indi_df['stdev'] = pd.Series.rolling(indi_df['Adj Close'], window=window, center=False).std().fillna(method='bfill')
    indi_df['Price/SMA'] = indi_df['Adj Close'] / indi_df['SMA']
    indi_df['bbp'] = (indi_df['Adj Close'] - indi_df['SMA'] + 2 * indi_df['stdev']) / (4 * indi_df['stdev'])
    indi_df['momentum'] = (indi_df['Adj Close'] / indi_df['Adj Close'].shift(momentum_day) - 1).fillna(method='bfill')
    indi_df['CLV'] = ((indi_df['Close'] - indi_df['Low']) - (indi_df['High'] - indi_df['Close'])) / (
        indi_df['High'] - indi_df['Low'])
    indi_df['Accdist'] = np.nan
    indi_df['Accdist'].ix[0] = indi_df['CLV'].ix[0] * indi_df['Volume'].ix[0]
    for i in range(1, indi_df.shape[0]):
        indi_df['Accdist'].ix[i] = indi_df['Accdist'].ix[i - 1] + indi_df['CLV'].ix[i] * indi_df['Volume'].ix[i]

    indi_df['Norm_price'] = normalize(indi_df['Adj Close'])
    indi_df['Norm_SMA'] = normalize(indi_df['SMA'])
    indi_df['Norm_Price/SMA_ratio'] = normalize(indi_df['Price/SMA'])
    indi_df['Norm_bbp'] = normalize(indi_df['bbp'])
    indi_df['Norm_momentum'] = normalize(indi_df['momentum'])
    indi_df['Norm_Accdist'] = normalize(indi_df['Accdist'])

    #print(indi_df)

    return indi_df


if __name__ == '__main__':
    sdate_train = '2008-01-01'
    edate_train = '2009-12-31'
    sdate_test = '2010-01-01'
    edate_test = '2011-12-31'

    indi_df = indicator_df(sdate=sdate_train, edate=edate_train, sym='AAPL', window=21, momentum_day=15)
    ax1 = indi_df.plot(y='Norm_Price/SMA_ratio', fontsize=12, title='Normalized Price/SMA ratio')
    ax2 = indi_df.plot(y='Norm_bbp', fontsize=12, title='Bollinger Bands (21 days) %')
    ax3 = indi_df.plot(y='Norm_momentum', fontsize=12, title='Normalized momentum (15 days)')
    ax4 = indi_df.plot(y='Norm_Accdist', fontsize=12, title='Normalized accumulation/distribution index ')

    plt.show()
