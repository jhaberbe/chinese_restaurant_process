import numpy as np
from scipy.special import gammaln

class ChineseRestaurantTable:
    """Base for Chinese Restaurant Tables"""

    def __init__(self, data):
        self.data = data
        self.members = set()
    
    def add_member(self, index: int):
        if index not in self.members:
            self.members.add(index)

    def remove_member(self, index: int):
        if index in self.members:
            self.members.remove(index)

    def log_likelihood(self, index: int, posterior: bool = False):
        pass

    def predict(self, count: np.ndarray):
        pass

class DirichletTable(ChineseRestaurantTable):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.members = set()
        self.concentration = np.ones((1, self.data.shape[1]))  # shape: (D,)

    def add_member(self, index: int):
        if index not in self.members:
            self.members.add(index)
            self.concentration += self.data[index]

    def remove_member(self, index: int):
        if index in self.members:
            self.members.remove(index)
            self.concentration -= self.data[index]

    def _dirichlet_multinomial_log_likelihood(self, count: np.ndarray, concentration: np.ndarray) -> float:
        N = np.sum(count)
        return (
            gammaln(N + 1)
            - np.sum(gammaln(count + 1))
            + gammaln(np.sum(concentration))
            - gammaln(np.sum(concentration) + N)
            + np.sum(gammaln(count + concentration) - gammaln(concentration))
        )

    def log_likelihood(self, index: int, posterior: bool = False):
        x = self.data[index]
        concentration = self.concentration + x if posterior else self.concentration
        return self._dirichlet_multinomial_log_likelihood(x, concentration)

    def predict(self, count: np.ndarray):
        return self._dirichlet_multinomial_log_likelihood(count, self.concentration)

class NegativeBinomialTable(ChineseRestaurantTable):
    def __init__(self, data: np.ndarray):
        self.data = data
        self.members = set()

        D = self.data.shape[1]
        self.alpha = np.ones(D)  # prior shape
        self.beta = np.ones(D)   # prior rate

        self.reference_total = np.mean(np.sum(data, axis=1))

    def add_member(self, index: int):
        if index not in self.members:
            self.members.add(index)
            self.alpha += self.data[index]
            self.beta += 1  # One new data point

    def remove_member(self, index: int):
        if index in self.members:
            self.members.remove(index)
            self.alpha -= self.data[index]
            self.beta -= 1

    def _gamma_poisson_log_likelihood(self, count: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> float:
        count = np.asarray(count).reshape(-1)
        alpha = np.asarray(alpha).reshape(-1)
        beta = np.asarray(beta).reshape(-1)

        # Compute size factor from total count vs. mean total count of self.data
        total = np.sum(count)
        reference_total = self.reference_total
        size_factor = total / reference_total if reference_total > 0 else 1.0
        log_sf = np.log(size_factor)

        # Gamma-Poisson log-likelihood with offset
        term1 = gammaln(count + alpha)
        term2 = -gammaln(count + 1)
        term3 = -gammaln(alpha)
        term4 = alpha * np.log(beta / (beta + np.exp(log_sf)))
        term5 = count * np.log(np.exp(log_sf) / (beta + np.exp(log_sf)))

        return np.sum(term1 + term2 + term3 + term4 + term5)

    def log_likelihood(self, index: int, posterior: bool = False):
        x = self.data[index]
        if posterior:
            alpha = self.alpha + x
            beta = self.beta + 1
        else:
            alpha = self.alpha
            beta = self.beta

        return self._gamma_poisson_log_likelihood(x, alpha, beta)

    def predict(self, count: np.ndarray):
        return self._gamma_poisson_log_likelihood(count, self.alpha, self.beta)