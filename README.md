# BYOST (Build Your Own Spectral Template)

Using Principal Component analysis (PCA) and Gaussian Process Regression (GPR) to build a spectral template based on two conditions (such as time and light-curve-shape parameter). 
This work is initially developped to construct the NIR spectral template of type Ia supernovae using the data obtained by CSP-II, see our [publication](https://arxiv.org/abs/2211.05998) for details. 

But it does NOT need to be limited to spectroscopic data, you can use this for other modeling purposes as long as the dataset and attached conditions show correlations (all value need to be finite).  The major process is:
- Apply PCA on the dataset to reduce the dimension to a small subspace
- Moeling the subspace (PC projection values) dependence on the given conditons using GPR 
- Inverse PCA transformation using the predicted PC from GPR given disired condition


## Installation
```
pip install BYOST
```

## Quick guide
A quick guide is provided in the notebook [BYOST_quick_guide.ipynb](https://github.com/DeerWhale/BYOST/blob/main/BYOST_quick_guide.ipynb).

## Cites
If you used this package for your research work, please kindly cite [our paper](https://arxiv.org/abs/2211.05998), much appriciated! :)
