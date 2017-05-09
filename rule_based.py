""" nzou3@gatech.edu """

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import indicators as ind
import marketsim as mks

sdate_train = '2008-01-01'
edate_train = '2009-12-31'
sdate_test = '2010-01-01'
edate_test = '2011-12-31'

indi_df_aapl = ind.indicator_df(sdate=sdate_train, edate=edate_train, sym='AAPL', window=21, momentum_day=15)
indi_df_spy = ind.indicator_df(sdate=sdate_train, edate=edate_train, sym='SPY', window=21, momentum_day=15)

M_orderdf = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

i = 0
k = 0
while i < indi_df_aapl.shape[0] - 21:
    if ((indi_df_aapl['Norm_Price/SMA_ratio'].ix[i] > 1.3 and indi_df_spy['Norm_Price/SMA_ratio'].ix[i] <= 0.65) \
                and (indi_df_spy['Norm_bbp'].ix[i] < 0.8)):

        M_orderdf.loc[k] = [indi_df_aapl.index[i], 'AAPL', 'SELL', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl.index[i + 21], 'AAPL', 'BUY', 200]

        k += 2
        i += 21

    elif ((indi_df_aapl['Norm_Price/SMA_ratio'].ix[i] < 0.95 and indi_df_spy['Norm_Price/SMA_ratio'].ix[i] >= 0.95) \
                  and (indi_df_spy['Norm_bbp'].ix[i] >= 0)):

        M_orderdf.loc[k] = [indi_df_aapl.index[i], 'AAPL', 'BUY', 200]
        M_orderdf.loc[k + 1] = [indi_df_aapl.index[i + 21], 'AAPL', 'SELL', 200]
        k += 2
        i += 21
    else:
        i = i + 1

print(M_orderdf.to_string())

M_orderdf.to_csv("order_manual.csv")
portvals = mks.compute_portvals(orders_file='order_manual.csv', start_val=100000, start_date='2008-01-01',
                                end_date='2009-12-31')

indi_df_aapl['Benchmark_reg'] = np.nan
indi_df_aapl['Benchmark_reg'].ix[0] = 100000
for i in range(0, indi_df_aapl.shape[0]):
    indi_df_aapl['Benchmark_reg'].ix[i] = 100000 - 200 * indi_df_aapl['Adj Close'].ix[0] + \
                                          200 * indi_df_aapl['Adj Close'].ix[i]

indi_df_aapl['Benchmark_reg']=indi_df_aapl['Benchmark_reg']/indi_df_aapl['Benchmark_reg'].ix[0]
portvals=portvals/portvals.ix[0]

indi_df_aapl = indi_df_aapl.merge(portvals, how='outer', left_index=True, right_index=True)
indi_df_aapl = indi_df_aapl.rename(columns={'Port_val': 'Manual rule-based'})
indi_df_aapl.plot(y=['Benchmark_reg', 'Manual rule-based'], fontsize=12, color=['k', 'b'],
                  title='Benchmark vs. Manual rule-based trader')

i = 0
while i < M_orderdf.shape[0]:
    if M_orderdf['Order'].ix[i] == 'BUY':
        plt.axvline(M_orderdf['Date'].ix[i], color='g')
        i += 2
    elif M_orderdf['Order'].ix[i] == 'SELL':
        plt.axvline(M_orderdf['Date'].ix[i], color='r')
        i += 2

plt.show()



