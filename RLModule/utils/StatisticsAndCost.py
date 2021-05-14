'''
    Python class for computing element statistics
'''

from collections import namedtuple
import numpy as np
from math import *
from scipy.stats import describe

class StatisticsAndCost:

    def __init__(self, penalty):
        self.penalty = penalty
        self.Statistics = namedtuple('Statistics',
                            ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                             'kurtosis', 'cost'))

    def __call__(self, eta):
        # description = describe(eta,bias=False)
        e = len(eta)*np.abs(eta)**2
        # description = describe(e,bias=False)
        description = describe(np.log(e),bias=False)
        # description = describe(np.log(eta) + np.log(len(eta)),bias=False)
        self.nobs = description.nobs
        self.minmax = description.minmax
        self.mean = description.mean
        self.variance = description.variance
        self.skewness = description.skewness
        self.kurtosis = description.kurtosis
        self.cost = self.mean
        # if self.nobs > 100: 
            # self.cost = 0.
        # self.cost = self.mean + self.penalty * self.variance
        # self.cost = self.mean + self.penalty * np.sqrt(self.variance) / self.mean
        # self.cost = self.variance / self.mean
        # self.cost = 1.0/(1.0/self.variance + 1.0/self.mean)
        # self.cost = self.mean + self.penalty * self.variance
        # self.cost = self.mean + self.penalty * np.sqrt(self.variance)
        return self.Statistics(self.nobs, self.minmax, self.mean, self.variance, self.skewness, self.kurtosis, self.cost)

############################################################################

if __name__ == "__main__":

    penalty = 1.0
    eta = np.arange(10)

    stats = StatisticsAndCost(penalty)
    print(stats(eta))