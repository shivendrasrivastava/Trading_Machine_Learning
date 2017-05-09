"""MC2-P1: Market simulator.
Ningmu Zou
nzou3@gatech.edu
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math

def compute_portvals(orders_file = "order_best.csv", start_val = 100000,start_date = '2008-01-01',end_date ='2009-12-31'):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    order_df = pd.read_csv(orders_file, parse_dates=True)
    #start_date = order_df.ix[0, 'Date']
    number_orders=order_df.shape[0]
    #end_date = order_df.ix[number_orders - 1, 'Date']

    syms = []
    for k, row in order_df.iterrows():
        if row['Symbol'] not in syms:
            syms.append(row['Symbol'])

    price_df = get_data(syms, pd.date_range(start_date, end_date))

    for sym in syms:
        price_df[sym + ' Shares'] = pd.Series(0, index=price_df.index)
    price_df['Cash'] = pd.Series(start_val, index=price_df.index)
    price_df['Port_val'] = pd.Series(start_val, index=price_df.index)

    for date in range(0, price_df.shape[0]):
        if date >= 1:
            price_df.ix[date, len(syms) + 1:] = price_df.ix[date - 1, len(syms) + 1:]

        for order in range(0, number_orders):
            if order_df.ix[order, 'Date'] == price_df.index[date].strftime('%Y-%m-%d'):

                if order_df.ix[order, 'Order'] == 'BUY':
                    b_s = 1
                elif order_df.ix[order, 'Order'] == 'SELL':
                    b_s = -1

                no_share = order_df.ix[order, 'Shares']
                sym = order_df.ix[order, 'Symbol']
                price_df.ix[date, sym + ' Shares'] += no_share * b_s
                sym_price = price_df.ix[date, sym]
                price_df.ix[date, 'Cash'] += -no_share * sym_price * b_s

                # leverage = (sum(abs(all stock positions))) / (sum(all stock positions) + cash)
                abs_stock_positions = np.dot(price_df.ix[date, 1:len(syms) + 1],
                                             np.absolute(price_df.ix[date, len(syms) + 1:2 * len(syms) + 1]))
                stock_positions = np.dot(price_df.ix[date, 1:len(syms) + 1],
                                         price_df.ix[date, len(syms) + 1:2 * len(syms) + 1])
                leverage_now = abs_stock_positions / (stock_positions + price_df.ix[date, 'Cash'])

                if leverage_now > 1.5:
                    price_df.ix[date, sym + ' Shares'] -= no_share * b_s
                    price_df.ix[date, 'Cash'] -= -no_share * sym_price * b_s

            stock_sum = np.dot(price_df.ix[date, 1:len(syms) + 1], price_df.ix[date, len(syms) + 1:2 * len(syms) + 1])

        price_df.ix[date, 'Port_val'] = stock_sum + price_df.ix[date, 'Cash']

    portvals = (price_df.ix[:, 'Port_val']).to_frame()

    return portvals


def author():
    return 'nzou3' # replace tb34 with your Georgia Tech username.


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "order_best.csv"

    sv = 100000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    #print(portvals)
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.

    start_date = portvals.index[0].strftime('%Y-%m-%d')
    end_date = portvals.index[-1].strftime('%Y-%m-%d')

    prices_SPX = (get_data(['$SPX'], pd.date_range(start_date, end_date)))[['$SPX']]
    portvals_SPX = get_portfolio_value(prices_SPX)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = compute_portfolio_stats(portvals_SPX)


    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])




def get_portfolio_value(prices,start_val=1):
    return (prices / prices.ix[0, :] * start_val).sum(axis=1)


def compute_portfolio_stats(port_val,rfr = 0.0, sf = 252.0):

    cr = (port_val[-1] / port_val[0]) - 1
    dr = (port_val / port_val.shift(1)) - 1
    adr = dr.mean()
    sddr = dr.std()
    k = np.sqrt(sf)
    sr = k * np.mean(adr - rfr)/sddr
    return cr, adr, sddr, sr


if __name__ == "__main__":
    test_code()
