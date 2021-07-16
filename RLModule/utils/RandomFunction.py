from math import sin, cos, atan2
import numpy as np
from pylab import *
from scipy.fft import *

class RandomFunction():

    def __init__(self, C=1, sigma=1e-3, N=1000):

        x = np.linspace(1/N,1,N)
        cov = C/np.sqrt(2*np.pi)/sigma * np.exp(-(x/sigma)**2/2)
        self.N = N
        self.cov = cov

    def sample(self):
        w = np.random.normal(0,1,size=self.N)
        g = idst(w*self.cov)
        self.g = [0] + g
        return self.g
    
    def __call__(self, s):
        assert s >= 0, "s must be normalized to [0,1]"
        assert s <= 1, "s must be normalized to [0,1]"
        if not hasattr(self, 'g'):
            self.sample()
        if s == 1:
            return self.g[-1]
        else:
            ind = math.floor(s*self.N)
            ds = (s*self.N - ind)/self.N
            return self.g[ind]*(1-ds) + self.g[ind+1]*ds

if __name__ == '__main__':
    
    load = RandomFunction()
    xx = np.linspace(0,1,5)
    g = np.array([load(x) for x in xx])
    plot(xx,g)

    xx = np.linspace(0,1,100)
    g = np.array([load(x) for x in xx])
    plot(xx,g)

    # for i in range(100):
        # plot(load.sample())
    plt.show()