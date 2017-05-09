"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case

    allocs = opt_alloc(prices)

    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs)
    # Get daily portfolio value
    port_val = ((prices/prices.ix[0,:])*allocs).sum(axis=1)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df = df_temp / df_temp.ix[0, :]
        ax = df.plot(title="Daily portfolio value and SPY", fontsize=12)
        ax.set_xlabel("Date",fontsize=12)
        ax.set_ylabel("Normalized price",fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr


def compute_portfolio_stats(prices, allocs=[0.1,0.2,0.3,0.4], rfr = 0.0, sf = 252.0):
    norm_value = prices/prices.ix[0,:]
    dev = norm_value*allocs
    port_val = dev.sum(axis=1)
    cr = (port_val[-1] / port_val[0]) - 1
    dr = (port_val / port_val.shift(1)) - 1
    adr = dr.mean()
    sddr = dr.std()
    k = np.sqrt(sf)
    sr = k * np.mean(adr - rfr)/sddr
    return cr, adr, sddr, sr

def minsr(allocs,prices):
    cr, adr, sddr, sr = compute_portfolio_stats(prices, allocs)
    return (1-sr)

def opt_alloc(prices):

    syms_number=len(prices.columns)
    initial_alloc=np.ones(syms_number)/syms_number
    constraints=({'type':'eq', 'fun': lambda allocs: 1-sum(allocs)})
    bounds=[(0,1) for i in range(syms_number)]
    opt_solution = scipy.optimize.minimize(minsr,initial_alloc,args=(prices),bounds=bounds,constraints=constraints)
    #print(opt_solution)
    allocs = opt_solution['x']

    for i in range(syms_number):
        if allocs[i]<1e-10:
            allocs[i]=0

    if (sum(allocs)-1>1e-4):
        allocs=allocs/sum(allocs)

    return allocs


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    #symbols = ['IBM', 'X', 'HNZ', 'XOM', 'GLD']

    symbols =  ['IBM', 'X', 'GLD']
    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,syms = symbols,gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
