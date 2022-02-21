# Graph Weather
Implementation of the Graph Weather paper (https://arxiv.org/pdf/2202.07575.pdf) in PyTorch.


## Installation

This library can be installed through

```bash
pip install graph-weather
```

## Pretrained Weights
Coming soon! We plan to train a model on GFS 0.25 degree operational forecasts, as well as MetOffice NWP forecasts.
We also plan trying out adaptive meshes, and predicting future satellite imagery as well.

## Training Data
Training data will be available through HuggingFace Datasets for the GFS forecasts. MetOffice NWP forecasts we cannot 
redistribute, but can be accessed through [CEDA](https://data.ceda.ac.uk/).