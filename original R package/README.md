## autoFRK Package
  [![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/autoFRK?color=green)](https://cran.r-project.org/package=autoFRK)
  [![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/autoFRK?color=brightgreen)](https://www.r-pkg.org/pkg/autoFRK)
  [![R build status](https://github.com/egpivo/autoFRK/workflows/R-CMD-check/badge.svg)](https://github.com/egpivo/autoFRK/actions)
  [![Coverage Status](https://img.shields.io/codecov/c/github/egpivo/autoFRK/master.svg)](https://codecov.io/github/egpivo/autoFRK?branch=master)
  

### Introduction
`autoFRK` (***Auto***matic ***F***ixed ***R***ank ***K***riging) is an R package to mitigate the intensive computation for modeling regularly/irregularly located spatial data using a class of basis functions with multi-resolution features and ordered in terms of their resolutions. 


### Installation
To get the current development version from GitHub:

```r
devtools::install_github("egpivo/autoFRK")
```

### Usage
- Main function `autoFRL`: see [demo](https://egpivo.github.io/autoFRK/reference/autoFRK.html#examples)


### Authors
- [ShengLi Tzeng](https://math.nsysu.edu.tw/p/405-1183-189657,c959.php?Lang=en)
- [Hsin-Cheng Huang](http://www.stat.sinica.edu.tw/hchuang/ "Hsin-Cheng Huang")
- [Wen-Ting Wang](https://www.linkedin.com/in/wen-ting-wang-6083a17b "Wen-Ting Wang")

### Reference
- Tzeng and Huang (2018). [Resolution Adaptive Fixed Rank Kriging, Technometrics Technometrics 
Volume 60, 2018 - Issue 2](https://www.tandfonline.com/doi/abs/10.1080/00401706.2017.1345701?journalCode=utch20). 

### License
  GPL (>= 2)

### Citation
- To cite package ‘autoFRK’ in publications use:
```
  Tzeng S, Huang H, Wang W, Nychka D, Gillespie C (2021). _autoFRK: Automatic Fixed
  Rank Kriging_. R package version 1.4.3,
  <https://CRAN.R-project.org/package=autoFRK>.
```

- A BibTeX entry for LaTeX users is
```
  @Manual{,
    title = {autoFRK: Automatic Fixed Rank Kriging},
    author = {ShengLi Tzeng and Hsin-Cheng Huang and Wen-Ting Wang and Douglas Nychka and Colin Gillespie},
    year = {2021},
    note = {R package version 1.4.3},
    url = {https://CRAN.R-project.org/package=autoFRK},
  }
```
