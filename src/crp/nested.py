import numpy as np
from .table import ChineseRestaurantTable, DirichletMultinomialTable, NegativeBinomialTable

class CRPNode:
    def __init__(self, data, depth=0, parent=None, table_class=None):
        self.depth = depth
        self.parent = parent
        self.children = []
        self.members = set()
        self.table = table_class(data)
        self.table_class = table_class

    def add_child(self, data):
        child = CRPNode(data, depth=self.depth + 1, parent=self, table_class=self.table_class)
        self.children.append(child)
        return child

    def add_member(self, index):
        self.members.add(index)
        self.table.add_member(index)

    def remove_member(self, index):
        self.members.discard(index)
        self.table.remove_member(index)

class NestedCRP:
    def __init__(self, data, table_class = DirichletMultinomialTable, alpha=1.0, max_depth=3):
        self.data = data
        self.table_class = table_class
        self.alpha = alpha
        self.max_depth = max_depth
        self.root = CRPNode(data, table_class=table_class)
        self.assignments = [None] * data.shape[0]

    def _sample_path(self, index):
        path = []
        node = self.root

        for depth in range(self.max_depth):
            path.append(node)

            # Score existing children
            log_scores = [child.table.log_likelihood(index, posterior=True) + np.log1p(len(child.members))
                          for child in node.children]

            # Score potential new child
            new_child = CRPNode(self.data, depth=node.depth + 1, parent=node, table_class=self.table_class)
            log_new = new_child.table.log_likelihood(index, posterior=True) + np.log(self.alpha)

            log_scores.append(log_new)
            scores = np.exp(log_scores - np.max(log_scores))
            probs = scores / scores.sum()
            choice = np.random.choice(len(probs), p=probs)

            if choice == len(node.children):
                # Create new node
                node = node.add_child(self.data)
                node.add_member(index)
            else:
                node = node.children[choice]
                node.add_member(index)

            if not node.children or node.depth + 1 == self.max_depth:
                break

        return path

    def run_epoch(self):
        for index in tqdm(np.random.permutation(self.data.shape[0]), desc="Sampling"):
            # Remove from current assignment
            current_path = self.assignments[index]
            if current_path:
                for node in current_path:
                    node.remove_member(index)

            # Sample new path
            path = self._sample_path(index)

            for node in path:
                node.add_member(index)

            self.assignments[index] = path

    def run(self, epochs=1):
        for _ in range(epochs):
            self.run_epoch()

    def get_node_assignments(self, as_dataframe=True):
        """Return path assignments for each sample as strings like '0.1.2'."""
        assignments = []

        for path in self.assignments:
            if path is None:
                assignments.append(None)
                continue

            node = path[-1]
            path_ids = []
            while node.parent is not None:
                parent = node.parent
                for idx, child in enumerate(parent.children):
                    if child is node:
                        path_ids.append(str(idx))
                        break
                node = parent
            path_str = ".".join(reversed(path_ids)) if path_ids else "0"
            assignments.append(path_str)

        if as_dataframe:
            return pd.DataFrame({"sample_index": range(len(assignments)), "node_path": assignments})
        return assignments