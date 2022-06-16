import numpy as np
from scipy.stats import multinomial, poisson
import abc


class Distribution(abc.ABC):
    @abc.abstractmethod
    def make_distribution(self):
        pass

    @abc.abstractmethod
    def sample(self, random_state=None):
        pass


class Multinomial(Distribution):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.kvals = np.arange(vmin, vmax)
        self.k = vmax - vmin

    def make_distribution(self, p=None):
        if p is None:
            p = [1/self.k] * self.k
        dist = multinomial(n=1, p=p)
        return dist

    def sample(self, n=1, random_state=None):
        dist = self.make_distribution()
        smpls = dist.rvs(n, random_state)
        smpls = self.kvals[np.argmax(smpls, axis=1)]
        return smpls

class Poisson(Distribution):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def make_distribution(self):
        dist = poisson(self.mu)
        return dist

    def sample(self, n=1, random_state=None):
        dist = self.make_distribution()
        smpls = dist.rvs(n, random_state)
        smpls += 2
        return smpls


if __name__ == "__main__":
    d = Poisson(2)
    d.sample(30)