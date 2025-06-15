import numpy as np
from typing import Union
from tqdm import tqdm, trange
from scipy.special import gammaln

class ChineseRestaurantTable:
    """A class representing a table in the Chinese Restaurant Process using NumPy."""

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float64)
        self.members = set()

        self.concentration = np.ones((1, data.shape[1]), dtype=np.float64)
        self.last_updated_epoch = 0

    def add_member(self, index: int, epoch: Union[int, None] = None):
        self.members.add(index)
        self.concentration += self.data[index]

        if epoch is not None:
            self.last_updated_epoch = epoch

    def remove_member(self, index: int):
        if index in self.members:
            self.members.remove(index)
            self.concentration -= self.data[index]

    def posterior_concentration(self, index: Union[int, np.ndarray]) -> np.ndarray:
        if isinstance(index, np.ndarray):
            return self.concentration + index
        else:
            return self.concentration + self.data[int(index)]

    def _log_dirichlet_multinomial(self, count: np.ndarray, concentration: np.ndarray) -> float:
        total_count = np.sum(count)
        return (
            gammaln(np.sum(concentration))
            - gammaln(np.sum(concentration) + total_count)
            + np.sum(gammaln(concentration + count) - gammaln(concentration))
        )

    def log_likelihood(self, index: Union[int, np.ndarray]) -> float:
        if isinstance(index, np.ndarray):
            return self._log_dirichlet_multinomial(index, self.concentration)
        else:
            count = self.data[int(index)]
            return self._log_dirichlet_multinomial(count, self.concentration)

    def posterior_log_likelihood(self, index: Union[int, np.ndarray]) -> float:
        if isinstance(index, np.ndarray):
            conc = self.posterior_concentration(index)
            return self._log_dirichlet_multinomial(index, conc)
        else:
            count = self.data[int(index)]
            conc = self.posterior_concentration(index)
            return self._log_dirichlet_multinomial(count, conc)

class ChineseRestaurantProcess:

    def __init__(self, data: np.ndarray, expected_classes: int = 10):
        self.data = data.astype(np.float64)
        self.classes = {}
        self.assignments = [-1] * data.shape[0]

        self.expected_classes = expected_classes
        self._alpha = expected_classes / np.log(data.shape[0])

    def generate_new_table(self):
        return ChineseRestaurantTable(self.data)

    def add_table(self, table, index=None):
        new_class_id = 0
        while new_class_id in self.classes:
            new_class_id += 1
        self.classes[new_class_id] = table
        if index is not None:
            table.add_member(index)
        return new_class_id

    def remove_table(self, class_id):
        if class_id in self.classes:
            for member in self.classes[class_id].members:
                self.assignments[member] = -1
            del self.classes[class_id]
        else:
            raise ValueError(f"Class ID {class_id} does not exist.")

    def run(self, epochs=1, max_classes=100, min_membership=0.01):
        for epoch in range(epochs):
            for i in tqdm(np.random.permutation(self.data.shape[0])):
                x_i = self.data[i]

                # Generate new table for this round
                crp_new = self.generate_new_table()

                # Existing class log-likelihoods
                cluster_keys = list(self.classes.keys()) + ["new"]
                nlls = []
                for k in self.classes:
                    table = self.classes[k]
                    log_like = table.posterior_log_likelihood(i)
                    log_prior = np.log1p(len(table.members))
                    nlls.append(log_like + log_prior)

                # New table likelihood
                log_new = crp_new.posterior_log_likelihood(i) + np.log(self._alpha)
                nlls.append(log_new)

                # Softmax sampling
                probs = np.exp(nlls - np.max(nlls))  # stability
                probs /= probs.sum()
                sampled_idx = np.random.choice(len(probs), p=probs)
                sampled_class = cluster_keys[sampled_idx]

                # Assignment
                if sampled_class == "new":
                    new_table = self.generate_new_table()
                    new_table.add_member(i, epoch)
                    self.add_table(new_table, i)
                else:
                    self.classes[sampled_class].add_member(i, epoch)
                    self.assignments[i] = int(sampled_class)

    def predict(self, X_new: np.ndarray, min_membership: float = 0.01) -> np.ndarray:
        """
        Predict the class for each sample in X_new using posterior log-likelihood.

        Parameters:
        - X_new: New data points to assign. Shape: (n_samples, n_features)
        - min_membership: Minimum proportion of total data a class must have to be used for prediction.

        Returns:
        - assignments: np.ndarray of predicted class labels
        """
        if not self.classes:
            raise ValueError("No classes have been trained. Run `run()` before predicting.")

        valid_classes = {
            k: v for k, v in self.classes.items()
            if len(v.members) >= min_membership * self.data.shape[0]
        }

        if not valid_classes:
            raise ValueError("No classes meet the minimum membership threshold.")

        assignments = []

        for x in tqdm(X_new):
            nlls = []
            keys = list(valid_classes.keys())
            for k in keys:
                table = valid_classes[k]
                log_like = table.posterior_log_likelihood(x)
                log_prior = np.log1p(len(table.members))
                nlls.append(log_like + log_prior)

            # Choose class with highest posterior log-likelihood + prior
            best_class = keys[np.argmax(nlls)]
            assignments.append(best_class)

        return np.array(assignments)