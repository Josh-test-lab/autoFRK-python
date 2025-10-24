# autoFRK-python

[![PyPI Version](https://img.shields.io/pypi/v/autoFRK.svg)](https://pypi.org/project/autoFRK/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-darkgreen.svg)](https://github.com/Josh-test-lab/autoFRK-python/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/autofrk?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/autofrk)
[![GitHub stars](https://img.shields.io/github/stars/Josh-test-lab/autoFRK-python.svg?style=social)](https://github.com/Josh-test-lab/autoFRK-python/stargazers)

`autoFRK-python` is a Python implementation of the R package `autoFRK` v1.4.3 (Tzeng S et al., 2021). `autoFRK` provides a **Resolution Adaptive Fixed Rank Kriging (FRK)** approach for handling regular and irregular spatial data, reducing computational cost through multi-resolution basis functions.


## Features

- Spatial modeling based on multi-resolution basis functions
- Supports single or multiple time points
- Offers approximate or EM-based model estimation
- Suitable for global latitude-longitude data
- Implemented in PyTorch, supporting CPU and GPU (requires PyTorch with CUDA support for GPU)


## Main Functions
- `AutoFRK`
  
  Automatic Fixed Rank Kriging.

- `MRTS`

  Multi-Resolution Thin-Plate Spline basis function.

## Installation

Install via pip:

```bash
pip install autoFRK
```

Install directly from GitHub:

```bash
pip install git+https://github.com/Josh-test-lab/autoFRK-python.git
```

Or clone and install manually:

```bash
git clone https://github.com/Josh-test-lab/autoFRK-python.git
cd autoFRK-python
pip install .
```



## Usage

### 1. Import and Initialize

```python
import torch
from autoFRK import AutoFRK

# Initialize the autoFRK model
model = AutoFRK(dtype=torch.float64, device="cpu")
```

### 2. Model Fitting

```python
# Assume `data` is (n, T) observations (NA allowed) and `loc` is (n, d) spatial coordinates  corresponding to n locations
data = torch.randn(100, 1)  # Example data
loc = torch.rand(100, 2)    # Example 2D coordinates

model_object = model.forward(
    data=data,
    loc=loc,
    maxit=50,
    tolerance=1e-6,
    method="fast",          # "fast" or "EM"
    n_neighbor=3
)

print(result.keys())
# ['M', 's', 'negloglik', 'w', 'V', 'G', 'LKobj']
```

`forward()` returns a dictionary including:

- **M**: ML estimate of *M*.
- **s**: Estimate for the scale parameter of measurement errors.
- **negloglik**: Negative log-likelihood.
- **w**: *K* by *T* matrix with *w[t]* as the *t*-th column.
- **V**: *K* by *K* matrix of the prediction error covariance matrix of *w[t]*.
- **G**: User specified basis function matrix or an automatically generated object from `MRTS`.
- **LKobj**: Not used yet.

### 3. Predicting New Data

```python
# Assume `newloc` contains new spatial coordinates
newloc = torch.rand(20, 2)

pred = model.predict(
    obj=result,
    newloc=newloc,
    se_report=True
)

print(pred['pred.value'])  # Predicted values
print(pred.get('se'))            # Standard errors
```

`predict()` can optionally return standard errors (`se_report=True`). If `obj` is not provided, the most recent `forward()` result is used.

## Arguments

- `AutoFRK`

`AutoFRK.forward()` supports various parameters:
| Parameter                  | Description                                                                                                                                                                          | Type                    | Default         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- | --------------- |
| `data`                     | *n* by *T* data matrix (NA allowed) with *z[t]* as the *t*-th column.                                                                                                                | `torch.Tensor`          | (Required)      |
| `loc`                      | *n* by *d* matrix of coordinates corresponding to *n* locations.                                                                                                                     | `torch.Tensor`          | (Required)      |
| `mu`                       | *n*-vector or scalar for µ.                                                                                                                                                          | `float \| torch.Tensor` | 0               |
| `D`                        | *n* by *n* matrix (preferably sparse) for the covariance matrix of the measurement errors up to a constant scale.                                                                    | `torch.Tensor`          | Identity matrix |
| `G`                        | *n* by *K* matrix of basis function values with each column being a basis function taken values at `loc`. Automatically determined if `None`.                                        | `torch.Tensor`          | `None`          |
| `maxK`                     | Maximum number of basis functions considered. Default is `None`, which means 10 · √*n* (for *n* > 100) or *n* (for *n* ≤ 100).                                                       | `int`                   | `None`          |
| `Kseq`                     | User-specified vector of numbers of basis functions considered. Default is `None`, which is determined from `maxK`.                                                                  |
| `maxknot`                  | Maximum number of knots used in generating basis functions.                                                                                                                          | `torch.Tensor`          | `None`          |
| `method`                   | `"fast"` or `"EM"`; `"fast"` fills missing data using *k*-nearest-neighbor imputation, while `"EM"` handles missing data via the EM algorithm.                                       | `str`                   | `"fast"`        |
| `n_neighbor`               | Number of neighbors used in the `"fast"` imputation method.                                                                                                                          | `int`                   | 3               |  | `int` | 5000 |
| `maxit`                    | Maximum number of iterations used in the `"EM"` imputation method.                                                                                                                   | `int`                   | 50              |
| `tolerance`                | Precision tolerance for convergence check used in the `"EM"` imputation method.                                                                                                      | `float`                 | 1e-6            |
| `requires_grad`            | If `True`, enables gradient computation for `data` tensor.                                                                                                                           | `bool`                  | `False`         |
| `calculate_with_spherical` | If `True`, calculates thin-plate spline distances using spherical coordinates.                                                                                                       | `bool`                  | `False`         |
| `finescale`                | Logical; if `True`, an (approximate) stationary finer-scale process *η[t]* will be included based on the **LatticeKrig** package. Only the diagonals of `D` are used. (Not used yet) | `bool`                  | `FALSE`         |
| `dtype`                    | Data type used in computations (e.g.,`torch.float64`).                                                                                                                               | `torch.dtype`           | `torch.float64` |
| `device`                   | Target computation device ("cpu", "cuda", "mps", etc.). If `None`, automatically selected.                                                                                           | `torch.device \| str`   | `None`          |

`AutoFRK.predict()` supports various parameters:
| Parameter   | Description                                                                                                                                                                                             | Type                      | Default         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | --------------- |
| `obj`       | A model object obtained from `AutoFRK`. If `None`, the model object produced by the `forward` method will be used.                                                                                      | `dict \| None`            | `None`          |
| `obsData`   | A vector with observed data used for prediction. Default is `None`, which uses the `data` input from `obj`.                                                                                             | `torch.Tensor \| None`    | `None`          |
| `obsloc`    | A matrix with rows being coordinates of observation locations for `obsData`. Only objects using `mrts` basis functions can have `obsloc` different from the `loc` input of `object`. Default is `None`. | `torch.Tensor \| None`    | `None`          |
| `mu_obs`    | A vector or scalar for the deterministic mean values at `obsloc`.                                                                                                                                       | `float` \| `torch.Tensor` | 0               |
| `newloc`    | A matrix with rows being coordinates of new locations for prediction. Default is `None`, which gives prediction at the locations of the observed data.                                                  | `torch.Tensor \| None`    | `None`          |
| `basis`     | A matrix with each column being a basis function taken values at `newloc`. Can be omitted if `object` was fitted using default `MRTS` basis functions.                                                  | `torch.Tensor \| None`    | `None`          |
| `mu_new`    | A vector or scalar for the deterministic mean values at `newloc`.                                                                                                                                       | `float` \| `torch.Tensor` | 0               |
| `se_report` | Logical; if `True`, the standard error of prediction is reported.                                                                                                                                       | `bool`                    | `False`         |
| `dtype`     | Data type used in computations (e.g., `torch.float64`). Defaults to the dtype of the model `obj` if available                                                                                           | `torch.dtype`             | `torch.float64` |
| `device`    | Target device for computations (e.g., 'cpu', 'cuda', 'mps'). If `None`, it will be selected automatically, with the device of the model `obj` used first if available.                                  | `torch.device \| str`     | `None`          |

- `MRTS`

`MRTS.forward()` supports various parameters:
| Parameter                  | Description                                                                                                                                                                              | Type                   | Default         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------- |
| `knot`                     | *m* by *d* matrix (*d* ≤ 3) for *m* locations of *d*-dimensional knots as in ordinary splines. Missing values are not allowed.                                                           | `torch.Tensor`         | (Required)      |
| `k`                        | The number (≤*m*) of basis functions.                                                                                                                                                    | `int`                  | `None`          |
| `x`                        | *n* by *d* matrix of coordinates corresponding to *n* locations where the values of basis functions are to be evaluated. Default is `None`, which uses the *m* by *d* matrix in  `knot`. | `torch.Tensor \| None` | `None`          |
| `maxknot`                  | Maximum number of knots to be used in generating basis functions. If `maxknot` <*m*, a deterministic subset selection of `knot`s will be used. To use all `knot`s, set `maxknot` ≥ *m*.  | `int`                  | 5000            |
| `calculate_with_spherical` | If `True`, calculates thin-plate spline distances using spherical coordinates.                                                                                                           | `bool`                 | `False`         |
| `dtype`                    | Data type used in computations (e.g.,`torch.float64`).                                                                                                                                   | `torch.dtype`          | `torch.float64` |
| `device`                   | Target computation device ("cpu", "cuda", "mps", etc.). If `None`, automatically selected.                                                                                               | `torch.device \| str`  | `None`          |

`MRTS.predict()` supports various parameters:
| Parameter                  | Description                                                                                                                                                            | Type                   | Default         |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------- |
| `obj`                      | A model object obtained from `MRTS`. If `None`, the model object produced by the `forward` method will be used.                                                        | `dict \| None`         | `None`          |
| `newx`                     | *n* by *d* matrix of coordinates corresponding to *n* locations where prediction is desired.                                                                           | `torch.Tensor \| None` | `None`          |
| `calculate_with_spherical` | If `True`, calculates thin-plate spline distances using spherical coordinates. The default `None` uses the method specified in `forward`.                              | `bool \| None`         | `None`          |
| `dtype`                    | Data type used in computations (e.g., `torch.float64`). Defaults to the dtype of the model `obj` if available                                                          | `torch.dtype`          | `torch.float64` |
| `device`                   | Target device for computations (e.g., 'cpu', 'cuda', 'mps'). If `None`, it will be selected automatically, with the device of the model `obj` used first if available. | `torch.device \| str`  | `None`          |

## Example Code

- `AutoFRK`
```python
import torch
from autoFRK import AutoFRK

# Generate fake data
n, T = 200, 1
data = torch.randn(n, T)
loc = torch.rand(n, 2)

# Initialize model
model = AutoFRK(device="cpu")

# Fit model
res = model.forward(
    data=data,
    loc=loc
)

# Predict new data
newloc = torch.rand(10, 2)
pred = model.predict(
    newloc=newloc
)

print("Predicted values:", pred['pred.value'])
```

- `MRTS`
```python
import torch
from autoFRK import MRTS

```

## Authors

- [ShengLi Tzeng](https://math.nsysu.edu.tw/p/405-1183-189657,c959.php?Lang=en) — *Original Paper Author*
- [Hsin-Cheng Huang](http://www.stat.sinica.edu.tw/hchuang/ "Hsin-Cheng Huang") — *Original Paper Author*
- [Wen-Ting Wang](https://www.linkedin.com/in/wen-ting-wang-6083a17b "Wen-Ting Wang") — *R Package Author*
- [Yao-Chih Hsu](https://github.com/Josh-test-lab/) — *Python Package Author*

## Contributors

- [Hao-Yun Huang](https://scholar.google.com/citations?user=AaydI0gAAAAJ&hl=zh-TW) — *Spherical Coordinate Thin Plate Spline Provider*
- [Yi-Xuan Xie](https://github.com/yixuan-dev) — *Python Package Tester*
- [Xuan-Chun Wang](https://github.com/wangxc1117) — *Python Package Tester*

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-darkgreen.svg)](https://github.com/Josh-test-lab/autoFRK-python/blob/main/LICENSE)
- GPL (>= 3)


## Development and Contribution

- Report bugs or request features on [GitHub issues](https://github.com/Josh-test-lab/autoFRK-python/issues)


## References

- Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). *autoFRK: Automatic Fixed Rank Kriging*. R package version 1.4.3, [https://CRAN.R-project.org/package=autoFRK](https://CRAN.R-project.org/package=autoFRK)
- Wang, J. W.-T. (n.d.). *autoFRK*. GitHub. Retrieved January 7, 2023, from [https://egpivo.github.io/autoFRK/](https://egpivo.github.io/autoFRK/)
- Tzeng, S. & Huang, H.-C. (2018). *Resolution Adaptive Fixed Rank Kriging*. Technometrics. [https://doi.org/10.1080/00401706.2017.1345701](https://doi.org/10.1080/00401706.2017.1345701)
- Nychka, D., Hammerling, D., Sain, S., & Lenssen, N. (2016). *LatticeKrig: Multiresolution Kriging Based on Markov Random Fields*


## Citation

- To cite the Python package `autoFRK-python` in publications use:

```
  Tzeng S, Huang H, Wang W, Hsu Y (2025). _autoFRK-python: Automatic Fixed Rank Kriging. The Python version with PyTorch_. Python package version 1.1.2, 
  <https://pypi.org/project/autoFRK/>.
```

- A BibTeX entry for LaTeX users to cite the Python package is:

```
  @Manual{,
    title = {autoFRK-python: Automatic Fixed Rank Kriging. The Python version with PyTorch},
    author = {ShengLi Tzeng and Hsin-Cheng Huang and Wen-Ting Wang and Yao-Chih Hsu},
    year = {2025},
    note = {Python package version 1.1.2},
    url = {https://pypi.org/project/autoFRK/},
  }
```

- To cite the R package `autoFRK` in publications use:

```
  Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). _autoFRK: Automatic Fixed Rank Kriging_. R package version 1.4.3,
  <https://CRAN.R-project.org/package=autoFRK>.
```

- A BibTeX entry for LaTeX users to cite the R package is:

```
  @Manual{,
    title = {autoFRK: Automatic Fixed Rank Kriging},
    author = {ShengLi Tzeng and Hsin-Cheng Huang and Wen-Ting Wang and Douglas Nychka and Colin Gillespie},
    year = {2021},
    note = {R package version 1.4.3},
    url = {https://CRAN.R-project.org/package=autoFRK},
  }
```

## Experimental Features

- Spherical coordinate basis function computation
- Gradient tracking (using torch's `requires_grad_()`)

## Release Notes

### v1.1.2
2025-10-23
- Removed dependencies on `faiss` and `scikit-learn`.
- Added a validation check to ensure that `data` and `loc` have the same number of rows.
- Other bug fixed.

### v1.1.1
2025-10-23
- Fixed a `ValueError` caused by a missing `v` in the model object when using the "EM" method.
- Fixed an issue with absent indices in the `EM0miss` function when using the "EM" method with missing data.
- Fixed a bug in the `EM0miss` function where some variables could not be found when handling missing data with the "EM" method.
- Improved the handling of `device` selection to reduce redundant checks and repeated triggers.
- Added input validation for the `mu` and `mu_new` variable.
- Updated additional functions to fully support `requires_grad`.
- Update README.

### v1.1.0
2025-10-21
- Added `dtype` and `device` parameters to `AutoFRK.predict()` and `MRTS.predict()`.
- Added `logger_level` parameter to `AutoFRK.__init__()` and `MRTS.__init__()` (default: 20). Options include `NOTSET`(0), `DEBUG`(10), `INFO`(20), `WARNING`(30), `ERROR`(40), `CRITICAL`(50).
- Enhanced automatic device selection, including MPS support.
- Fixed device assignment issue when `device` is not specified, preventing redundant parameter transfers.

### v1.0.0
2025-10-19
- Ported R package `autoFRK` to Python.

## Repositories
- Python Repository: 
  [https://github.com/Josh-test-lab/autoFRK-python](https://github.com/Josh-test-lab/autoFRK-python)
- R Repository: 
  [https://github.com/egpivo/autoFRK](https://github.com/egpivo/autoFRK)


## To Do
- [ ] Update `MRTS` examples in README
- [ ] Check all examples in README
- [ ] Check all Arguments in README
- [ ] Rewrite all discriptions in functions
- [ ] Rewrite `calculate_with_spherical: bool` function to `tps_method: str`
- [ ] Move some `README` chapters to files

---
If you like this project, don't forget to give it a star [here](https://github.com/Josh-test-lab/autoFRK-python).
