# autoFRK-python

[![PyPI Version](https://img.shields.io/pypi/v/autoFRK.svg)](https://pypi.org/project/autoFRK/)
[![Downloads](https://img.shields.io/pypi/dm/autoFRK.svg)](https://pypi.org/project/autoFRK/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Josh-test-lab/autoFRK-python/LICENSE)

- **Repository:** [https://github.com/Josh-test-lab/autoFRK-python](https://github.com/Josh-test-lab/autoFRK-python)

`autoFRK-python` is a Python implementation of the R package `autoFRK` v1.4.3 (Tzeng S et al., 2021). `autoFRK` provides a **Resolution Adaptive Fixed Rank Kriging (FRK)** approach for handling regular and irregular spatial data, reducing computational cost through multi-resolution basis functions.


## Features

- Spatial modeling based on multi-resolution basis functions
- Supports single or multiple time points
- Offers approximate or EM-based model estimation
- Suitable for global latitude-longitude data
- Implemented in PyTorch, supporting CPU and GPU



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
# Assume `data` is (n, T) observations and `loc` is (n, d) spatial coordinates
data = torch.randn(100, 1)  # Example data
loc = torch.rand(100, 2)    # Example 2D coordinates

result = model.forward(
    data=data,
    loc=loc,
    maxit=50,
    tolerance=1e-6,
    method="fast",          # "fast" or "EM"
    n_neighbor=3,
    maxK=50,
    calculate_with_spherical=False
)

print(result.keys())
# ['M', 's', 'negloglik', 'w', 'V', 'G', 'LKobj', 'calculate_with_spherical']
```

`forward()` returns a dictionary including:

- **M**: Covariance matrix of random effects
- **s**: Measurement error variance
- **negloglik**: Final negative log-likelihood
- **w**: Estimated random effects for each time point
- **V**: Prediction error covariance of `w`
- **G**: Basis function matrix used

### 3. Predicting New Data

```python
# Assume `newloc` contains new spatial coordinates
newloc = torch.rand(20, 2)

pred = model.predict(
    obj=result,
    newloc=newloc,
    se_report=True
)

print(pred['pred.value'].shape)  # Predicted values
print(pred.get('se'))            # Standard errors
```

`predict()` can optionally return standard errors (`se_report=True`). If `obj` is not provided, the most recent `forward()` result is used.

## Advanced Settings

`forward()` supports various parameters:

| Parameter                  | Description                         | Default                         |
| -------------------------- | ----------------------------------- | ------------------------------- |
| `mu`                       | Mean value (scalar or tensor)       | 0.0                             |
| `D`                        | Measurement error covariance        | None (identity matrix used)     |
| `G`                        | Basis function matrix (optional)    | None (TPS basis auto-generated) |
| `finescale`                | Include fine-scale process η[t]     | False                           |
| `maxit`                    | Maximum iterations                  | 50                              |
| `tolerance`                | Convergence tolerance               | 1e-6                            |
| `maxK`                     | Maximum number of basis functions   | Auto-set based on n             |
| `method`                   | Model estimation method             | "fast"                          |
| `n_neighbor`               | Number of neighbors for fast method | 3                               |
| `calculate_with_spherical` | Use spherical distance calculation  | False                           |

## Example Code

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
    loc=loc,
    maxit=100,
    method="fast"
)

# Predict new data
newloc = torch.rand(10, 2)
pred = model.predict(obj=res, newloc=newloc, se_report=True)

print("Predicted values:", pred['pred.value'])
print("Prediction standard errors:", pred.get('se'))
```

## Experimental Features

- Spherical coordinate basis function computation


## Authors

- [ShengLi Tzeng](https://math.nsysu.edu.tw/p/405-1183-189657,c959.php?Lang=en)
- [Hsin-Cheng Huang](http://www.stat.sinica.edu.tw/hchuang/ "Hsin-Cheng Huang")
- [Wen-Ting Wang](https://www.linkedin.com/in/wen-ting-wang-6083a17b "Wen-Ting Wang")
- [Yao-Chih Hsu](https://github.com/Josh-test-lab/)
- [Yi-Xuan Xie](https://github.com/yixuan-dev)

## License

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Josh-test-lab/autoFRK-python/LICENSE)
- GPL (>= 3)


## Development and Contribution

- Built with PyTorch, supporting GPU acceleration
- Report bugs or request features on [GitHub issues](https://github.com/Josh-test-lab/autoFRK-python/issues)


## References

- Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). *autoFRK: Automatic Fixed Rank Kriging*. R package version 1.4.3, [https://CRAN.R-project.org/package=autoFRK](https://CRAN.R-project.org/package=autoFRK)
- Wang, J. W.-T. (n.d.). *autoFRK*. GitHub. Retrieved January 7, 2023, from [https://egpivo.github.io/autoFRK/](https://egpivo.github.io/autoFRK/)
- Tzeng, S. & Huang, H.-C. (2018). *Resolution Adaptive Fixed Rank Kriging*. Technometrics. [https://doi.org/10.1080/00401706.2017.1345701](https://doi.org/10.1080/00401706.2017.1345701)
- Nychka, D., Hammerling, D., Sain, S., & Lenssen, N. (2016). *LatticeKrig: Multiresolution Kriging Based on Markov Random Fields*


### Citation

- To cite the Python package `autoFRK-python` in publications use:

```
  Yao-Chih Hsu (2025). _autoFRK-python: Automatic Fixed Rank Kriging. The Python version with PyTorch_. Python package version 1.0.0, 
  <https://github.com/Josh-test-lab/autoFRK-python>.
```

- A BibTeX entry for LaTeX users is:

```
  @Manual{,
    title = {autoFRK-python: Automatic Fixed Rank Kriging. The Python version with PyTorch},
    author = {Yao-Chih Hsu},
    year = {2025},
    note = {Python package version 1.0.0},
    url = {https://github.com/Josh-test-lab/autoFRK-python},
  }
```

- To cite the original R package `autoFRK`:

```
  Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). _autoFRK: Automatic Fixed Rank Kriging_. R package version 1.4.3,
  <https://CRAN.R-project.org/package=autoFRK>.
```

- A BibTeX entry for the original R package is:

```
  @Manual{,
    title = {autoFRK: Automatic Fixed Rank Kriging},
    author = {ShengLi Tzeng and Hsin-Cheng Huang and Wen-Ting Wang and Douglas Nychka and Colin Gillespie},
    year = {2021},
    note = {R package version 1.4.3},
    url = {https://CRAN.R-project.org/package=autoFRK},
  }
```
