""" nzou3@gatech.edu """

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import indicators as ind
import marketsim as mks
import RTLearner as rt
import math
import BagLearner as bl
import util

sdate_train = '2008-01-01'
edate_train = '2009-12-31'
sdate_test = '2010-01-01'
edate_test = '2011-12-31'
pd.options.mode.chained_assignment = None
indi_df_aapl = ind.indicator_df(sdate=sdate_train, edate=edate_train, sym='AAPL', window=21, momentum_day=15)
indi_df_spy = ind.indicator_df(sdate=sdate_train, edate=edate_train, sym='SPY', window=21, momentum_day=15)
indi_df_aapl_out = ind.indicator_df(sdate=sdate_test, edate=edate_test, sym='AAPL', window=21, momentum_day=15)
indi_df_spy_out = ind.indicator_df(sdate=sdate_test, edate=edate_test, sym='SPY', window=21, momentum_day=15)

## Following is the same manual rule for out of sample data

M_orderdf = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
i = 0
k = 0
while i < indi_df_aapl_out.shape[0] - 21:
    if ((indi_df_aapl_out['Norm_Price/SMA_ratio'].ix[i] > 1.3 and indi_df_spy_out['Norm_Price/SMA_ratio'].ix[i] <= 0.65) \
                and (indi_df_spy_out['Norm_bbp'].ix[i] < 0.8)):

        M_orderdf.loc[k] = [indi_df_aapl_out.index[i], 'AAPL', 'SELL', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl_out.index[i + 21], 'AAPL', 'BUY', 200]

        k += 2
        i += 21

    elif ((indi_df_aapl_out['Norm_Price/SMA_ratio'].ix[i] < 0.95 and indi_df_spy_out['Norm_Price/SMA_ratio'].ix[
        i] >= 0.95) \
                  and (indi_df_spy_out['Norm_bbp'].ix[i] >= 0)):

        M_orderdf.loc[k] = [indi_df_aapl_out.index[i], 'AAPL', 'BUY', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl_out.index[i + 21], 'AAPL', 'SELL', 200]
        k += 2
        i += 21
    else:
        i = i + 1

# print(M_orderdf.to_string())

M_orderdf.to_csv("order_manual_outsample.csv")
################

## Use the sample learner from the in sample data
total_date = indi_df_aapl.shape[0]
total_date_out = indi_df_aapl_out.shape[0]
indi_df_aapl_ml = indi_df_aapl[['Norm_Price/SMA_ratio', 'Norm_bbp', 'Norm_momentum', 'Norm_Accdist']]
indi_df_aapl_ml_out = indi_df_aapl_out[['Norm_Price/SMA_ratio', 'Norm_bbp', 'Norm_momentum', 'Norm_Accdist']]
indi_df_aapl_ml['kreturn'] = np.nan
indi_df_aapl_ml['kreturn'].ix[0:-21] = indi_df_aapl['Adj Close'].ix[21:total_date].values \
                                       / indi_df_aapl['Adj Close'].ix[0:-21].values - 1
indi_df_aapl_ml['kreturn'].ix[-21:] = 0

YBUY = 0.06
YSELL = -0.06
for i in range(0, total_date):
    if indi_df_aapl_ml['kreturn'].ix[i] > YBUY:
        indi_df_aapl_ml['kreturn'].ix[i] = 1
    elif indi_df_aapl_ml['kreturn'].ix[i] < YSELL:
        indi_df_aapl_ml['kreturn'].ix[i] = -1
    else:
        indi_df_aapl_ml['kreturn'].ix[i] = 0

data = indi_df_aapl_ml.as_matrix()
train_rows = total_date
trainX = data[:train_rows, 0:-1]
trainY = data[:train_rows, -1]

"""IMPORTANT: GET TEST DATA"""
data_test = indi_df_aapl_ml_out.as_matrix()
testX = data_test[:total_date_out, :]

# learner = rt.RTLearner(leaf_size=5, verbose=False)
# learner.addEvidence(trainX, trainY)
# predY = learner.query(trainX)
# predtestY = learner.query(testX)

learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=10)
learner.addEvidence(trainX, trainY)
predY = learner.query(trainX)
"""IMPORTANT: USE THE SAME LEARNER"""
predtestY = learner.query(testX)

rmsein = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
c = np.corrcoef(predY, y=trainY)

# print "In sample results"
# print "RMSE: ", rmsein
# print "corr: ", c[0, 1]

indi_df_aapl_ml['mlresult'] = predY
indi_df_aapl_ml_out['mlresult'] = predtestY

M_orderdf = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

i = 0
k = 0
while i < indi_df_aapl_ml_out.shape[0] - 21:
    if (indi_df_aapl_ml_out['mlresult'].ix[i] == -1):
        M_orderdf.loc[k] = [indi_df_aapl_ml_out.index[i], 'AAPL', 'SELL', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl_ml_out.index[i + 21], 'AAPL', 'BUY', 200]
        k += 2
        i += 21

    elif (indi_df_aapl_ml_out['mlresult'].ix[i] == 1):
        M_orderdf.loc[k] = [indi_df_aapl_ml_out.index[i], 'AAPL', 'BUY', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl_ml_out.index[i + 21], 'AAPL', 'SELL', 200]
        k += 2
        i += 21
    else:
        i = i + 1

# print(M_orderdf.to_string())

indi_df_aapl_out['Benchmark_reg'] = np.nan
indi_df_aapl_out['Benchmark_reg'].ix[0] = 100000
for i in range(0, indi_df_aapl_out.shape[0]):
    indi_df_aapl_out['Benchmark_reg'].ix[i] = 100000 - 200 * indi_df_aapl_out['Adj Close'].ix[0] + \
                                              200 * indi_df_aapl_out['Adj Close'].ix[i]

indi_df_aapl_out['Benchmark_reg'] = indi_df_aapl_out['Benchmark_reg'] / indi_df_aapl_out['Benchmark_reg'].ix[0]

M_orderdf.to_csv("order_ml_outsample.csv")
portvals_ml = mks.compute_portvals(orders_file='order_ml_outsample.csv', start_val=100000,
                                   start_date=sdate_test, end_date=edate_test)
portvals_ml = portvals_ml / portvals_ml.ix[0]

portval_manual = mks.compute_portvals(orders_file='order_manual_outsample.csv', start_val=100000,
                                      start_date=sdate_test, end_date=edate_test)
portval_manual = portval_manual / portval_manual.ix[0]

indi_df_aapl_ml_out = indi_df_aapl_ml_out.merge(indi_df_aapl_out, how='outer', left_index=True, right_index=True)
indi_df_aapl_ml_out = indi_df_aapl_ml_out.merge(portval_manual, how='outer', left_index=True, right_index=True)
indi_df_aapl_ml_out = indi_df_aapl_ml_out.rename(columns={'Port_val': 'Rule-based'})
indi_df_aapl_ml_out = indi_df_aapl_ml_out.merge(portvals_ml, how='outer', left_index=True, right_index=True)
indi_df_aapl_ml_out = indi_df_aapl_ml_out.rename(columns={'Port_val': 'ML-based'})


outsample_result = []
outsample_result.append([])
outsample_result[0].append('Benchmark')
outsample_result.append([])
outsample_result[1].append('Rule-based')
outsample_result.append([])
outsample_result[2].append('ML-based')

k=0
for i in ['Benchmark_reg', 'Rule-based', 'ML-based']:
    portvals = indi_df_aapl_ml_out[i]
    start_date = portvals.index[0].strftime('%Y-%m-%d')
    end_date = portvals.index[-1].strftime('%Y-%m-%d')

    prices_SPX = (util.get_data(['$SPX'], pd.date_range(start_date, end_date)))[['$SPX']]
    portvals_SPX = mks.get_portfolio_value(prices_SPX)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = mks.compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = mks.compute_portfolio_stats(portvals_SPX)

    outsample_result[k].append(start_date)
    outsample_result[k].append(end_date)
    outsample_result[k].append(float('%.3g' % sharpe_ratio))
    outsample_result[k].append(float('%.3g' % cum_ret))
    outsample_result[k].append(float('%.3g' % std_daily_ret))
    outsample_result[k].append(float('%.3g' % avg_daily_ret))

    print outsample_result[k]
    k+=1


# print "Date Range: {} to {}".format(start_date, end_date)
# print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
# print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
# print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
# print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)

indi_df_aapl_ml_out.plot(y=['Benchmark_reg', 'Rule-based', 'ML-based'], fontsize=12, color=['k', 'b', 'g'],
                         title='Out of sample comparison')
i = 0
while i < M_orderdf.shape[0]:
    if M_orderdf['Order'].ix[i] == 'BUY':
        plt.axvline(M_orderdf['Date'].ix[i], color='g')
        i += 2
    elif M_orderdf['Order'].ix[i] == 'SELL':
        plt.axvline(M_orderdf['Date'].ix[i], color='r')
        i += 2
plt.figure(1)
plt.show()
