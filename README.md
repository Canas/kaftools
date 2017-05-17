**Warning: this is a side-project in progress so many bugs could arise. Please raise an issue if this happens.**

# Kernel Adaptive Filtering for Python
This package implements several Kernel Adaptive Filtering algorithms for research purposes. It aims to be easily extendable.

# Requirements
- Python 3
- NumPy
- SciPy
- (Optional) Matplotlib

# Features
## Adaptive Kernel Filters
- Kernel Least Mean Squares (KLMS) - `KlmsFilter`
- Exogenous Kernel Least Mean Squares (KLMS-X) - `KlmsxFilter`
- Kernel Recursive Least Squares (KRLS) - `KrlsFilter`

## Sparsification Criteria
- Novelty (KLMS)
- Approximate Linear Dependency (KLRS)

## Additional Features
- Delayed input support (KLMS)
- Adaptive kernel parameter learning (KLMS)

For a more visual comparison, check the [latest features sheet](https://docs.google.com/spreadsheets/d/1kvBNAqDSgNGBTcXqMDN7j_dpp949peH_-F1GYVP29y8/edit?usp=sharing).

# Quickstart
Let's do a simple example using a KLMS Filter over given input and target arrays:
```
from kaftools.filters import KlmsFilter
from kaftools.kernels import GaussianKernel

klms = KlmsFilter(input, target)
klms.fit(learning_rate=0.1, kernel=GaussianKernel(sigma=0.1))
```

And that's it!
