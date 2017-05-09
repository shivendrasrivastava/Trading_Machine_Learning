""" nzou3@gatech.edu """

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import indicators as ind
import marketsim as mks
import RTLearner as rt
import numpy as np
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

total_date = indi_df_aapl.shape[0]
indi_df_aapl_ml = indi_df_aapl[['Norm_Price/SMA_ratio', 'Norm_bbp', 'Norm_momentum', 'Norm_Accdist']]
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


# learner = rt.RTLearner(leaf_size=5, verbose=False)
# learner.addEvidence(trainX, trainY)
# predY = learner.query(trainX)

learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=10)
learner.addEvidence(trainX, trainY)
predY = learner.query(trainX)

rmsein = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
c = np.corrcoef(predY, y=trainY)

# print "In sample results"
# print "RMSE: ", rmsein
# print "corr: ", c[0, 1]

indi_df_aapl_ml['mlresult'] = predY


M_orderdf = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

i = 0
k = 0
while i < indi_df_aapl.shape[0] - 21:
    if (indi_df_aapl_ml['mlresult'].ix[i] == -1):
        M_orderdf.loc[k] = [indi_df_aapl.index[i], 'AAPL', 'SELL', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl.index[i + 21], 'AAPL', 'BUY', 200]
        k += 2
        i += 21

    elif (indi_df_aapl_ml['mlresult'].ix[i] == 1):
        M_orderdf.loc[k] = [indi_df_aapl.index[i], 'AAPL', 'BUY', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl.index[i + 21], 'AAPL', 'SELL', 200]
        k += 2
        i += 21
    else:
        i = i + 1

#print(M_orderdf.to_string())

indi_df_aapl['Benchmark_reg'] = np.nan
indi_df_aapl['Benchmark_reg'].ix[0] = 100000
for i in range(0, indi_df_aapl.shape[0]):
    indi_df_aapl['Benchmark_reg'].ix[i] = 100000 - 200 * indi_df_aapl['Adj Close'].ix[0] + \
                                          200 * indi_df_aapl['Adj Close'].ix[i]

indi_df_aapl['Benchmark_reg'] = indi_df_aapl['Benchmark_reg'] / indi_df_aapl['Benchmark_reg'].ix[0]

M_orderdf.to_csv("order_ml.csv")
portvals_ml = mks.compute_portvals(orders_file='order_ml.csv', start_val=100000,
                                   start_date='2008-01-01', end_date='2009-12-31')
portvals_ml = portvals_ml / portvals_ml.ix[0]

portval_manual = mks.compute_portvals(orders_file='order_manual.csv', start_val=100000,
                                      start_date='2008-01-01', end_date='2009-12-31')
portval_manual = portval_manual / portval_manual.ix[0]

indi_df_aapl = indi_df_aapl.merge(portval_manual, how='outer', left_index=True, right_index=True)
indi_df_aapl = indi_df_aapl.rename(columns={'Port_val': 'Rule-based'})
indi_df_aapl = indi_df_aapl.merge(portvals_ml, how='outer', left_index=True, right_index=True)
indi_df_aapl = indi_df_aapl.rename(columns={'Port_val': 'ML-based'})

indi_df_aapl.plot(y=['Benchmark_reg', 'Rule-based', 'ML-based'], fontsize=12, color=['k', 'b', 'g'],
                  title='Benchmark vs. Rule-based trader vs. ML-based trader')



insample_result = []
insample_result.append([])
insample_result[0].append('Benchmark')
insample_result.append([])
insample_result[1].append('Rule-based')
insample_result.append([])
insample_result[2].append('ML-based')

k=0
for i in ['Benchmark_reg', 'Rule-based', 'ML-based']:
    portvals=indi_df_aapl[i]
    start_date = portvals.index[0].strftime('%Y-%m-%d')
    end_date = portvals.index[-1].strftime('%Y-%m-%d')

    prices_SPX = (util.get_data(['$SPX'], pd.date_range(start_date, end_date)))[['$SPX']]
    portvals_SPX = mks.get_portfolio_value(prices_SPX)

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = mks.compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = mks.compute_portfolio_stats(portvals_SPX)

    insample_result[k].append(start_date)
    insample_result[k].append(end_date)
    insample_result[k].append(float('%.3g' % sharpe_ratio))
    insample_result[k].append(float('%.3g' % cum_ret))
    insample_result[k].append(float('%.3g' % std_daily_ret))
    insample_result[k].append(float('%.3g' % avg_daily_ret))

    print insample_result[k]
    k+=1

print "Date Range: {} to {}".format(start_date, end_date)
print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)



"""Following are all about PLOT"""
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

indi_df_aapl['indicator_result'] = np.nan
i = 0
for i in range(0, indi_df_aapl.shape[0]):
    if ((indi_df_aapl['Norm_Price/SMA_ratio'].ix[i] > 1.3 and indi_df_spy['Norm_Price/SMA_ratio'].ix[i] <= 0.65) \
                and (indi_df_spy['Norm_bbp'].ix[i] < 0.8)):
        indi_df_aapl['indicator_result'].ix[i] = -1
    elif ((indi_df_aapl['Norm_Price/SMA_ratio'].ix[i] < 0.95 and indi_df_spy['Norm_Price/SMA_ratio'].ix[i] >= 0.95) \
                  and (indi_df_spy['Norm_bbp'].ix[i] >= 0)):
        indi_df_aapl['indicator_result'].ix[i] = 1
    else:
        indi_df_aapl['indicator_result'].ix[i] = 0

dv_1_x1 = indi_df_aapl['Norm_Price/SMA_ratio'].as_matrix()
dv_1_x2 = indi_df_aapl['Norm_bbp'].as_matrix()
color1 = indi_df_aapl['indicator_result'].as_matrix()
color2 = indi_df_aapl_ml['kreturn'].as_matrix()
color3 = indi_df_aapl_ml['mlresult'].as_matrix()

plt.figure(2)
LONG1 = color1 == 1
SHORT1 = color1 == -1
CLOSE1 = color1 == 0
plt.hold(True)
plt.scatter(dv_1_x1[CLOSE1], dv_1_x2[CLOSE1], label='CLOSE', c='k', s=100, lw=0)
plt.scatter(dv_1_x1[LONG1], dv_1_x2[LONG1], label='LONG', c='g', s=100, lw=0)
plt.scatter(dv_1_x1[SHORT1], dv_1_x2[SHORT1], label='SHORT', c='r', s=100, lw=0)
plt.title('Rule-based strategy', fontsize=20)
plt.xlabel('Norm_Price/SMA_ratio', fontsize=20)
plt.ylabel('Norm_bbp', fontsize=20)
plt.legend(loc=2)
plt.axis((-1.5, 1.5, -1.5, 1.5))

plt.figure(3)
LONG2 = color2 == 1
SHORT2 = color2 == -1
CLOSE2 = color2 == 0
plt.hold(True)
plt.scatter(dv_1_x1[CLOSE2], dv_1_x2[CLOSE2], label='CLOSE', c='k', s=100, lw=0)
plt.scatter(dv_1_x1[LONG2], dv_1_x2[LONG2], label='LONG', c='g', s=100, lw=0)
plt.scatter(dv_1_x1[SHORT2], dv_1_x2[SHORT2], label='SHORT', c='r', s=100, lw=0)
plt.title('Training data for ML strategy', fontsize=20)
plt.xlabel('Norm_Price/SMA_ratio', fontsize=20)
plt.ylabel('Norm_bbp', fontsize=20)
plt.legend(loc=2)
plt.axis((-1.5, 1.5, -1.5, 1.5))

plt.figure(4)
LONG3 = color3 == 1
SHORT3 = color3 == -1
CLOSE3 = color3 == 0
plt.hold(True)
plt.scatter(dv_1_x1[CLOSE3], dv_1_x2[CLOSE3], label='CLOSE', c='k', s=100, lw=0)
plt.scatter(dv_1_x1[LONG3], dv_1_x2[LONG3], label='LONG', c='g', s=100, lw=0)
plt.scatter(dv_1_x1[SHORT3], dv_1_x2[SHORT3], label='SHORT', c='r', s=100, lw=0)
plt.title('Response of ML strategy', fontsize=20)
plt.xlabel('Norm_Price/SMA_ratio', fontsize=20)
plt.ylabel('Norm_bbp', fontsize=20)
plt.legend(loc=2)
plt.axis((-1.5, 1.5, -1.5, 1.5))

plt.show()
