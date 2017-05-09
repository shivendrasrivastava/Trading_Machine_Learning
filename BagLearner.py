'''
2017 Spring CS7646 ML4T mc3p1
Ningmu Zou
nzou3@gatech.edu
'''

import numpy as np
import RTLearner as rt
from collections import Counter

np.set_printoptions(threshold=np.nan)

class BagLearner(object):
    def __init__(self, learner = rt.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False):

        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        self.kwargs = {"k": 10}
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'nzou3' # replace tb34 with your Georgia Tech username.

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.dataX = dataX
        self.dataY = dataY


    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        Y = np.zeros((points.shape[0], self.bags))
        for lr in self.learners:
            add_x = np.zeros(self.dataX.shape)
            randind = np.random.randint(0, self.dataX.shape[0], self.dataX.shape[0])

            for c_ind in range(self.dataX.shape[1]):
                add_x[:, c_ind] = np.array([self.dataX[r_ind, c_ind] for r_ind in randind])
                add_y = np.array([self.dataY[r_ind] for r_ind in randind])

            for i in range(0, self.bags):
                lr.addEvidence(add_x, add_y)
                Y[:, i] = lr.query(points)

        Y_return= np.zeros(points.shape[0])
        for i in range(0,points.shape[0]):
            Y_return[i] = Counter(Y[i, :]).most_common(1)[0][0]

        return Y_return


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
