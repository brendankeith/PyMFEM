'''
    Python class for computing element statistics
'''

from collections import namedtuple
import numpy as np
from math import *
from scipy.stats import describe

stats = namedtuple('Statistics',
                ('nels', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis'))

########################
# local error statistics
########################
def Statistics(eta, num_dofs=np.e):
    # zeta = np.log(len(eta)*np.abs(eta)**2)/np.log(len(eta))
    zeta = np.log(len(eta)*np.abs(eta)**2)/np.log(num_dofs)
    # zeta = np.log(len(eta)*np.abs(eta)**2)
    description = describe(zeta,bias=False)
    nels = description.nobs
    min = description.minmax[0]
    max = description.minmax[1]
    mean = description.mean
    variance = description.variance
    skewness = description.skewness
    kurtosis = description.kurtosis
    return stats(nels, min, max, mean, variance, skewness, kurtosis)

########################
# total estimated error
########################
def GlobalError(eta):
    return np.sqrt(np.sum(np.abs(eta)**2))
    
############################################################################

if __name__ == "__main__":

    eta = np.arange(1,10)
    print(Statistics(eta))