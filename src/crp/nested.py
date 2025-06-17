import numpy as np
from tqdm import trange, tqdm
from .table import ChineseRestaurantTable, DirichletMultinomialTable, NegativeBinomialTable

class ChineseRestaurantProcessNode:
    def __init__(
            self, 
            data, 
            table_class: ChineseRestaurantTable, 
            parent = None, 
            depth: int = 0,
            expected_number_of_classes: int = 1
        ):
        # Setup data.
        self.data = data
        self.table_class = table_class
        self.table = self.table_class(data)

        # Setup tree structure.
        self.depth = depth
        self.parent = parent
        self.children = {}
        self.members = set()

        # Setup infernce machinery.
        self.expected_number_of_classes = expected_number_of_classes
        self.alpha = expected_number_of_classes / np.log(self.data.shape[0])

    def add_child(self, data):
        # Create a new child node with the given data.
        child = ChineseRestaurantProcessNode(
            data,
            depth=self.depth + 1,
            parent=self,
            table_class=self.table_class,
            expected_number_of_classes = self.expected_number_of_classes
        )

        # Find the next available slot for the child.
        i = 0
        while i in self.children:
            i+=1

        # Add the child to the children dictionary.
        self.children[i] = child

        # Return the newly created child node.
        return child

    def add_member(self, index):
        self.members.add(index)
        self.table.add_member(index)

    def remove_member(self, index):
        self.members.discard(index)
        self.table.remove_member(index)
    
    def has_member(self, index):
        return index in self.members
    
    @staticmethod
    def sample_path(node, index, depth=0, max_depth=4):
        node.add_member(index)
        existing_children = list(node.children.items())
        log_posteriors = []

        # Score existing children
        for child_key, child_node in existing_children:
            ll = child_node.table.log_likelihood(index, posterior=True)
            prior = np.log1p(len(child_node.members))  # prior favors larger children
            log_posteriors.append(ll + prior)

        # Score new child
        new_child = ChineseRestaurantProcessNode(
            node.table.data,
            depth = node.depth + 1,
            parent = node,
            table_class = node.table_class,
            expected_number_of_classes = node.expected_number_of_classes
        )

        # Log likelihood of the new child
        ll_new = new_child.table.log_likelihood(index, posterior=True)
        prior_new = np.log(node.alpha if hasattr(node, 'alpha') else 1.0)  # Use alpha if set, else 1.0
        log_posteriors.append(ll_new + prior_new)

        # Normalize and sample
        log_posteriors = np.array(log_posteriors)
        max_log = np.max(log_posteriors)
        probs = np.exp(log_posteriors - max_log)
        probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)

        if choice == len(existing_children):
            # Create and add new child
            new_key = 0
            while new_key in node.children:
                new_key += 1
            node.children[new_key] = new_child
            return [new_key]

        else:
            child_key = existing_children[choice][0]
            if depth + 1 < max_depth:
                # Recurse down the chosen child node
                path = ChineseRestaurantProcessNode.sample_path(
                    node.children[child_key],
                    index,
                    depth = depth + 1,
                    max_depth = max_depth
                )
                return [child_key] + path
            else:
                # At max depth, add member to this node and return path
                node.children[child_key].add_member(index)
                return [child_key]

    def predict_paths(root, count_matrix, max_depth=4):
        """
        For each row in the count_matrix, traverse the tree from root,
        selecting the most likely child at each level, returning the path.

        Returns:
            paths: list of lists of keys, one per sample.
        """
        paths = []
        for index in trange(count_matrix.shape[0]):
            node = root
            path = []
            depth = 0

            while depth < max_depth and node.children:
                best_key = None
                best_score = -np.inf

                for key, child in node.children.items():
                    ll = child.table.log_likelihood(index, posterior=True)
                    prior = np.log1p(len(child.members))  # favor larger clusters
                    score = ll + prior
                    if score > best_score:
                        best_score = score
                        best_key = key

                if best_key is None:
                    break  # no children

                path.append(best_key)
                node = node.children[best_key]
                depth += 1

            paths.append(path)

        return paths

    def run(self, epochs = 1, max_depth = 4):
        for _ in range(epochs):
            for index in trange(self.data.shape[0]):
                path = ChineseRestaurantProcessNode.sample_path(self, index, max_depth = max_depth)