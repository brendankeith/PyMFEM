'''
    Python class for computing element statistics
'''

from collections import namedtuple
import numpy as np
from math import *
from scipy.stats import describe

class StatisticsAndCost:

    def __init__(self):
        self.Statistics = namedtuple('Statistics',
                            ('nels', 'min', 'max', 'mean', 'variance', 'skewness',
                             'kurtosis', 'cost'))

    def __call__(self, eta):
        e = len(eta)*np.abs(eta)**2
        description = describe(np.log(e),bias=False)
        self.nels = description.nobs
        self.min = description.minmax[0]
        self.max = description.minmax[1]
        self.mean = description.mean
        self.variance = description.variance
        self.skewness = description.skewness
        self.kurtosis = description.kurtosis
        self.cost = np.log(np.sum(np.abs(eta)**2))/2 # log of total estimated error
        return self.Statistics(self.nels, self.min, self.max, self.mean, self.variance, self.skewness, self.kurtosis, self.cost)

############################################################################

if __name__ == "__main__":

    eta = np.arange(1,10)
    stats = StatisticsAndCost()
    print(stats(eta))