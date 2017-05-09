""" nzou3@gatech.edu """
"""MLT: Utility code."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None
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


if __name__ == '__main__':
    sdate_train = '2008-01-01'
    edate_train = '2009-12-31'
    sdate_test = '2010-01-01'
    edate_test = '2011-12-31'

    symbols = ['AAPL']
    stock_df_all = pd.read_csv(symbol_to_path(symbols[0]), index_col='Date', parse_dates=True, na_values=['nan'])
    initial_total = 100000

    date_range = pd.date_range(sdate_train, edate_train)
    stock_df = pd.DataFrame(index=date_range)
    stock_df = stock_df.join(stock_df_all).dropna()

    stock_df['Benchmark_reg'] = np.nan
    stock_df['Benchmark_reg'].ix[0] = 100000
    stock_df['Position'] = np.nan
    stock_df['BUY'] = 0
    stock_df['SELL'] = 0

    stock_df['Portfolio'] = np.nan
    for i in range(0, stock_df.shape[0]):
        stock_df['Benchmark_reg'].ix[i] = 100000 - 200 * stock_df['Adj Close'].ix[0] + 200 * stock_df['Adj Close'].ix[i]

    if stock_df['Adj Close'].ix[1] > stock_df['Adj Close'].ix[0]:
        stock_df['Position'].ix[0] = 200
        stock_df['BUY'].ix[0] = 200
    elif stock_df['Adj Close'].ix[1] == stock_df['Adj Close'].ix[0]:
        stock_df['Position'].ix[0] = 0
    elif stock_df['Adj Close'].ix[1] < stock_df['Adj Close'].ix[0]:
        stock_df['Position'].ix[0] = -200
        stock_df['SELL'].ix[0] = 200

    for i in range(0, stock_df.shape[0] - 1):
        if stock_df['Adj Close'].ix[i + 1] > stock_df['Adj Close'].ix[i]:
            if stock_df['Position'].ix[i - 1] == 200:
                stock_df['Position'].ix[i] = 200
            elif stock_df['Position'].ix[i - 1] == -200:
                stock_df['Position'].ix[i] = 200
                stock_df['BUY'].ix[i] = 400
            elif stock_df['Position'].ix[i - 1] == 0:
                stock_df['Position'].ix[i] = 200
                stock_df['BUY'].ix[i] = 200

        elif stock_df['Adj Close'].ix[i + 1] < stock_df['Adj Close'].ix[i]:
            if stock_df['Position'].ix[i - 1] == 200:
                stock_df['Position'].ix[i] = -200
                stock_df['SELL'].ix[i] = 400

            elif stock_df['Position'].ix[i - 1] == -200:
                stock_df['Position'].ix[i] = -200

            elif stock_df['Position'].ix[i - 1] == 0:
                stock_df['Position'].ix[i] = -200
                stock_df['SELL'].ix[i] = 200

        elif stock_df['Adj Close'].ix[i + 1] == stock_df['Adj Close'].ix[i]:
            if stock_df['Position'].ix[i - 1] == 0:
                stock_df['Position'].ix[i] = 0

            elif stock_df['Position'].ix[i - 1] == 200:
                stock_df['Position'].ix[i] = 200

            elif stock_df['Position'].ix[i - 1] == -200:
                stock_df['Position'].ix[i] = -200

    stock_df['Portfolio'].ix[0] = 100000
    stock_df['Position'].ix[-1] = stock_df['Position'].ix[-2]

    stock_df['Position_price'] = abs(stock_df['Position'] * stock_df['Adj Close'])
    for i in range(1, stock_df.shape[0]):
        stock_df['Portfolio'].ix[i] = stock_df['Portfolio'].ix[i - 1] + abs(
            stock_df['Position_price'].ix[i] - stock_df['Position_price'].ix[i - 1])

    stock_df['Best_portfolio'] = stock_df['Portfolio'] / stock_df['Portfolio'].ix[0]
    stock_df['Benchmark'] = stock_df['Benchmark_reg'] / stock_df['Benchmark_reg'].ix[0]

    cum_bench = stock_df['Benchmark'].ix[-1] / stock_df['Benchmark'].ix[0] - 1
    cum_best = stock_df['Best_portfolio'].ix[-1] / stock_df['Best_portfolio'].ix[0] - 1

    daily_return_bench = np.zeros((stock_df.shape[0]))
    daily_return_best = np.zeros((stock_df.shape[0]))

    for i in range(0, stock_df.shape[0] - 1):
        daily_return_bench[i] = stock_df['Benchmark_reg'].ix[i + 1] / stock_df['Benchmark_reg'].ix[i] - 1
        daily_return_best[i] = stock_df['Portfolio'].ix[i + 1] / stock_df['Portfolio'].ix[i] - 1

    orderdf = pd.DataFrame(index=stock_df.index)
    orderdf.index.names = ['Date']
    orderdf['Symbol'] = 'AAPL'
    orderdf['Order'] = ''
    orderdf['Shares'] = 0

    for i in range(0, orderdf.shape[0]):
        if stock_df['BUY'].ix[i] > 0:
            orderdf['Order'].ix[i] = 'BUY'
            orderdf['Shares'].ix[i] = stock_df['BUY'].ix[i]
        elif stock_df['SELL'].ix[i] > 0:
            orderdf['Order'].ix[i] = 'SELL'
            orderdf['Shares'].ix[i] = stock_df['SELL'].ix[i]

    orderdf = orderdf.loc[(orderdf['Shares'] != 0)]
    print(orderdf)

    orderdf.to_csv("order_best.csv")

    #print(stock_df)

    print 'Cumulative return of the benchmark is ', cum_bench
    print 'Cumulative return of the best possible portfolio is ', cum_best
    print 'Standard deviation of daily returns of benchmark is ', np.std(daily_return_bench)
    print 'Standard deviation of daily returns of best portfolio is ', np.std(daily_return_best)
    print 'Mean of daily returns of benchmark is ', np.mean(daily_return_bench)
    print 'Mean of daily returns of portfolio is ', np.mean(daily_return_best)

    ax = stock_df.plot(y=['Benchmark', 'Best_portfolio'], color=['k', 'b'], fontsize=12)
    plt.title('Benchmark vs. Best possible portfolio')
    plt.show()
