# chinese_restaurant_process
================

chinese_restaurant_process is a Python package that provides an easy to use interface for simulating the Chinese Restaurant Process (CRP), a popular model in Bayesian nonparametrics.

## Installation
---------------

### Installation Methods

You can install Chinese Restaurant Process using pip:

```bash
pip install git+https://www.github.com/jhaberbe/chinese_restaurant_process
```

Or, if you want to install the package from source, you can use:

```bash
git clone https://github.com/jhabere/chinese_restaurant_process.git
pip install .
```

## Usage
--------

### Example Use Cases

To perform the initial inference of classes:

```python
import numpy as np
from crp.process import ChineseRestaurantProcess

# Your data, (n_samples, n_features)
X = np.random.randint(1, 100, size=(1000, 10))

# Run inference on train data.
crp = ChineseRestaurantProcess(X, expected_number_of_classes=1)
crp.run(epochs=1)
```

After training, you can predict the class of new data points:

```python
# Your data, (n_samples, n_features)
X_new = np.random.randint(1, 100, size=(1000, 10))

# Setting min_membership = 0.01 is recommended usually.
# Since this is random data, we set it to 0
labels = crp.predict(X_new, min_membership=0.0)
```

### Documentation

For more information on the package's functionality, please refer to the [documentation](https://[package_name]-docs.readthedocs.io/).

## Contributing
------------

We welcome contributions. If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your changes.
3. Make the changes to your branch.
4. Commit your changes with a meaningful commit message.
5. Create a pull request against the main branch.

## License
--------

Chinese Restaurant Process is released under the GPL v3 license. See the LICENSE file for more information.