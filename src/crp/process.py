from tqdm import tqdm
from crp.process import DirichletMultinomialTable

class ChineseRestaurantProcess:
    _table_type = DirichletMultinomialTable

    def __init__(self, data: np.array, expected_number_of_classes: int = 1):
        # Data
        self.data = data

        # Classes
        self.classes = {}
        self.assignments = [-1] * data.shape[0]

        # Expected number of classes, setting alpha prior
        self.expected_number_of_classes = expected_number_of_classes
        self._alpha = self.expected_number_of_classes / np.log(self.data.shape[0])
    
    def set_table_type(self, cls):
        self._table_type = cls
    
    def generate_new_table(self):
        return self._table_type(self.data)

    def add_table(self, table: ChineseRestaurantTable, index: int):
        # Find smallest unused lot.
        new_class_id = 0
        while new_class_id in self.classes:
            new_class_id += 1
        self.classes[new_class_id] = table
        self.classes[new_class_id].add_member(index)
        self.assignments[index] = new_class_id

    def remove_table(self, class_id):
        if class_id in self.classes:
            for member in self.classes[class_id].members:
                self.assignments[member] = -1
            del self.classes[class_id]
        else:
            raise ValueError(f"Class ID {class_id} does not exist.")

    def run(self, epochs=1, min_membership=0.01):
        # For each epoch
        for epoch in range(epochs):
            # For each item (shuffled selection for robustness).
            for index in tqdm(np.random.permutation(self.data.shape[0])):

                # Generate new table for this round
                crp_new = self.generate_new_table()

                # Existing class log-likelihoods
                cluster_keys = list(self.classes.keys()) + ["new"]
                nlls = []
                for k in self.classes:
                    table = self.classes[k]
                    log_like = table.log_likelihood(index, posterior = True)
                    log_prior = np.log1p(len(table.members))
                    nlls.append(log_like + log_prior)

                # New table likelihood
                log_new = crp_new.log_likelihood(index, posterior = True) + np.log(self._alpha)
                nlls.append(log_new)

                # Softmax sampling
                probs = np.exp(nlls - np.max(nlls))
                probs /= probs.sum()
                sampled_idx = np.random.choice(len(probs), p=probs)
                sampled_class = cluster_keys[sampled_idx]

                # Assignment
                if sampled_class == "new":
                    self.add_table(crp_new, index)
                else:
                    # This is sort of bad.
                    self.classes[sampled_class].add_member(index)
                    self.assignments[index] = int(sampled_class)