'''
2017 Spring CS7646 ML4T mc3p1
Ningmu Zou
nzou3@gatech.edu
'''

import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'nzou3' # replace tb34 with your Georgia Tech username.

#funtion of building a decision tree
    def build_tree(self, dataX, dataY):
        # In case of small training data or repeated training data

        leaf = np.array([-1, max(dataY.tolist(),key=dataY.tolist().count), np.nan, np.nan])
        #leaf = np.array([-1, np.mean(dataY), np.nan, np.nan])


        # Record the number of columns of X and number of rows of X
        number_X = dataX.shape[1]
        number_Row = dataX.shape[0]

        if ( all(x == dataY[0] for x in dataY) or dataX.shape[0] <= self.leaf_size):
            return leaf;

        # This while-loop is to get avoid of selecting two different rows with same Y but different X.
        p = 1;
        while (p < 15):
            xi = np.random.randint(0, number_X);
            r_ind = [np.random.randint(0, number_Row), np.random.randint(0, number_Row)];
            p += 1;
            if dataX[r_ind[0], xi] != dataX[r_ind[1], xi]:
                break

        # If the data set is too small, it has to return to a leaf
        if dataX[r_ind[0], xi] == dataX[r_ind[1], xi]:
            return leaf;

        # Setting split value of random tree
        spv = (dataX[r_ind[0], xi] + dataX[r_ind[1], xi]) / 2;

        # Separating tree into two branches
        lefttree = self.build_tree(dataX[(dataX[:, xi] <= spv), :], dataY[(dataX[:, xi] <= spv)]);
        righttree = self.build_tree(dataX[(dataX[:, xi] > spv), :], dataY[(dataX[:, xi] > spv)]);

        lefttree_size = lefttree.ndim;

        # Setting the root according to the tree size
        if lefttree_size > 1:
            root = np.array([xi, spv, 1, lefttree.shape[0] + 1]);
        elif lefttree_size == 1:
            root = np.array([xi, spv, 1, 2]);

        # Return to a tree
        return np.vstack((root, lefttree, righttree));


    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)

        if self.verbose == True:
            print(self.tree)


    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        trainY = []

        for point in points:
            i = 0;
            while (i < self.tree.shape[0]):
                node_ind = int(self.tree[i, 0]);

                # In case the training dataset is too small
                if node_ind == -1:
                    break;

                #Judge the data should go to left tree or right tree
                elif point[node_ind] > self.tree[i, 1]:
                    i += int(self.tree[i, 3]);
                elif point[node_ind] <= self.tree[i, 1]:
                    i += 1;

            if node_ind >= 0:
                trainY.append(np.nan);
            else:
                trainY.append(self.tree[i, 1]);

        return trainY


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
