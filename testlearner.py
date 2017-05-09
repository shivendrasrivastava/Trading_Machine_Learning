"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import sys
import RTLearner as rt
import BagLearner as bl
import matplotlib.pyplot as plt

#config "Data\winequality-red.csv"

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])



    # learner = lrl.LinRegLearner(verbose = False) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it



    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 20}, bags=20, boost=False, verbose=False)
    # learner.addEvidence(trainX, trainY)
    # predY = learner.query(trainX)



    np.random.shuffle(data)
    train_rows = int(math.floor(0.6 * data.shape[0]))
    test_rows = data.shape[0] - train_rows
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]



    leaf_s=1
    runtime=1
    RMSEin=np.zeros((leaf_s, runtime));
    RMSEout = np.zeros((leaf_s, runtime));
    for k in range(leaf_s):


        for i in range(runtime):
            #print(i)

            learner = rt.RTLearner(leaf_size=k, verbose=False)
            learner.addEvidence(trainX, trainY)
            predY = learner.query(trainX)

            rmsein = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            RMSEin[k,i]=rmsein
            c = np.corrcoef(predY, y=trainY)
            # print
            # print "In sample results"
            # print "RMSE: ", rmsein
            # print "corr: ", c[0,1]

            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            rmseout = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            RMSEout[k, i] = rmseout
            c = np.corrcoef(predY, y=testY)
            # print
            # print "Out of sample results"
            # print "RMSE: ", rmseout
            # print "corr: ", c[0,1]

    xaxis=np.arange(1,leaf_s+1)
    insampleplot, = plt.plot(xaxis,np.mean(RMSEin, axis=1), label="In sample")
    outsampleplot, = plt.plot(xaxis,np.mean(RMSEout, axis=1), label="Out of sample")
    first_legend = plt.legend(handles=[insampleplot], loc=1)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[outsampleplot], loc=4)
    plt.ylabel('RMSE')
    plt.xlabel('Leaf size')
    plt.title('RTlearner on various leaf size')
    plt.show()



    leaf_s=1
    runtime=1
    RMSEin=np.zeros((leaf_s, runtime));
    RMSEout = np.zeros((leaf_s, runtime));
    for k in range(leaf_s):


        for i in range(runtime):
            #print(i)
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": k}, bags=20, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY)
            print trainX.size
            print trainY.size
            predY = learner.query(trainX)

            rmsein = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            RMSEin[k,i]=rmsein
            c = np.corrcoef(predY, y=trainY)
            print
            print "In sample results"
            print "RMSE: ", rmsein
            print "corr: ", c[0,1]

            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            rmseout = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            RMSEout[k, i] = rmseout
            c = np.corrcoef(predY, y=testY)
            print
            print "Out of sample results"
            print "RMSE: ", rmseout
            print "corr: ", c[0,1]
            print(i)
        print(k)
    xaxis=np.arange(1,leaf_s+1)
    insampleplot, = plt.plot(xaxis,np.mean(RMSEin, axis=1), label="In sample")
    outsampleplot, = plt.plot(xaxis,np.mean(RMSEout, axis=1), label="Out of sample")
    first_legend = plt.legend(handles=[insampleplot], loc=1)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[outsampleplot], loc=4)
    plt.ylabel('RMSE')
    plt.xlabel('Leaf size')
    plt.title('BagLearner on various leaf size')
    plt.show()


    number_bags=10
    runtime=1
    RMSEin=np.zeros((number_bags));
    RMSEout = np.zeros((number_bags));
    for k in range(number_bags):

        RMSEini=0;
        RMSEouti = 0;
        for i in range(runtime):
            #print(i)
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=k+1, boost=False, verbose=False)
            learner.addEvidence(trainX, trainY)
            predY = learner.query(trainX)

            RMSEini+= math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])

            c = np.corrcoef(predY, y=trainY)
            # print
            # print "In sample results"
            # print "RMSE: ", rmsein
            # print "corr: ", c[0,1]

            # evaluate out of sample
            predY = learner.query(testX) # get the predictions
            RMSEouti += math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            c = np.corrcoef(predY, y=testY)
            # print
            # print "Out of sample results"
            # print "RMSE: ", rmseout
            # print "corr: ", c[0,1]
            print(k,i)

        RMSEin[k]=RMSEini
        RMSEout[k] = RMSEouti


    xaxis=np.arange(1,number_bags+1)
    insampleplot, = plt.plot(xaxis,RMSEin/number_bags, label="In sample")
    outsampleplot, = plt.plot(xaxis,RMSEout/number_bags, label="Out of sample")
    first_legend = plt.legend(handles=[insampleplot], loc=1)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[outsampleplot], loc=4)
    plt.ylabel('RMSE')
    plt.xlabel('Leaf size')
    plt.title('BagLearner on various leaf size')
    plt.show()