# Graph Weather
Implementation of the Graph Weather paper (https://arxiv.org/pdf/2202.07575.pdf) in PyTorch. Additionally, an implementation
of a modified model that assimilates raw or processed observations into analysis files. 


## Installation

This library can be installed through

```bash
pip install graph-weather
```

## Example Usage

The models generate the graphs internally, so the only thing that needs to be passed to the model is the node features
in the same order as the ```lat_lons```.

```python
import torch
from graph_weather import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))
model = GraphWeatherForecaster(lat_lons)

features = torch.randn((2, len(lat_lons), 78))

out = model(features)
criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
loss = criterion(out, features)
loss.backward()
```

And for the assimilation model, which assumes each lat/lon point also has a height above ground, and each observation 
is a single value + the relative time. The assimlation model also assumes the desired output grid is given to it as 
well.

```python
import torch
from graph_weather import GraphWeatherAssimilator
from graph_weather.models.losses import NormalizedMSELoss

obs_lat_lons = []
for lat in range(-90, 90, 7):
    for lon in range(0, 180, 6):
        obs_lat_lons.append((lat, lon, np.random.random(1)))
    for lon in 360 * np.random.random(100):
        obs_lat_lons.append((lat, lon, np.random.random(1)))

output_lat_lons = []
for lat in range(-90, 90, 5):
    for lon in range(0, 360, 5):
        output_lat_lons.append((lat, lon))
model = GraphWeatherAssimilator(output_lat_lons=output_lat_lons, analysis_dim=24)

features = torch.randn((1, len(obs_lat_lons), 2))
lat_lon_heights = torch.tensor(obs_lat_lons)
out = model(features, lat_lon_heights)
assert not torch.isnan(out).all()
assert out.size() == (1, len(output_lat_lons), 24)

criterion = torch.nn.MSELoss()
loss = criterion(out, torch.randn((1, len(output_lat_lons), 24)))
loss.backward()
```

## Pretrained Weights
Coming soon! We plan to train a model on GFS 0.25 degree operational forecasts, as well as MetOffice NWP forecasts.
We also plan trying out adaptive meshes, and predicting future satellite imagery as well.

## Training Data
Training data will be available through HuggingFace Datasets for the GFS forecasts. MetOffice NWP forecasts we cannot
redistribute, but can be accessed through [CEDA](https://data.ceda.ac.uk/).
