# Adaptive Kernel Filtering for Python
This library implements several Kernel Adaptive Filtering algorithms for research purposes. 

Currently supports Python 3.3+ only.

# Features
## Adaptive Kernel Filters
- Kernel Least Mean Squares (KLMS): `KlmsFilter`
- Exogenous Kernel Least Mean Squares (KLMS-X): `KlmsxFilter`
- Kernel Recursive Least Squares: `KrlsFilter`

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
from kernel_filtering.filters import KlmsFilter
from kernel_filtering.kernels import GaussianKernel

klms = KlmsFilter(input, target)
klms.fit(learning_rate=0.1, kernel=GaussianKernel(sigma=0.1))
```

And that's it! 

# Requirements
- NumPy
- SciPy
- (Optional) Matplotlib