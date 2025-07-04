# Graph Weather
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-11-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repo implements graph neural networks for weather forecasting, originally an implementation of
the Graph Weather paper (https://arxiv.org/pdf/2202.07575.pdf) in PyTorch. Additionally, multiple other
models have now been added, as well as general models for assimilation and forecasting.

The models implemented include:

DeepMind's [Functional Generative Network (FGN)](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/how-we-re-supporting-better-tropical-cyclone-prediction-with-ai/skillful-joint-probabilistic-weather-forecasting-from-marginals.pdf) for probablistic ensemble forecasting

DeepMind's [GenCast](https://www.nature.com/articles/s41586-024-08252-9) for graph diffusion-based forecasting

WindBorne's [WeatherMesh-3](https://arxiv.org/abs/2503.22235) for highly efficient forecasting with Neighborhood Attention

Microsoft's [Aurora](https://arxiv.org/abs/2405.13063) forecasting model

And [FengWu-GHR](https://arxiv.org/abs/2402.00059) forecasting, using LoRA to correct for per-forecast step errors

The components of these models should be fairly modular and be able to be swapped around to experiment with graph-based weather forecasting.

## Installation

This library can be installed through

```bash
pip install graph-weather
```

Alternatively, you can install the latest version from the repository easily with `pixi`:

```bash
pixi install # `-e cuda` for GPU support, `-e cpu` for CPU-only
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

# Generate 78 random features + 24 non-NWP features (i.e. landsea mask)
features = torch.randn((2, len(lat_lons), 102))

target = torch.randn((2, len(lat_lons), 78))
out = model(features)

criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
loss = criterion(out, target)
loss.backward()
```

And for the assimilation model, which assumes each lat/lon point also has a height above ground, and each observation
is a single value + the relative time. The assimlation model also assumes the desired output grid is given to it as
well.

```python
import torch
import numpy as np
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
Training data will be available through HuggingFace Datasets for the GFS forecasts. The initial set of data is available for [GFSv16 forecasts, raw observations, and FNL Analysis files from 2016 to 2022](https://huggingface.co/datasets/openclimatefix/gfs-reforecast), and for [ERA5 Reanlaysis](https://huggingface.co/datasets/openclimatefix/era5-reanalysis). MetOffice NWP forecasts we cannot
redistribute, but can be accessed through [CEDA](https://data.ceda.ac.uk/).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Bieker"/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/graph_weather/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jack-kelly.com"><img src="https://avatars.githubusercontent.com/u/460756?v=4?s=100" width="100px;" alt="Jack Kelly"/><br /><sub><b>Jack Kelly</b></sub></a><br /><a href="#ideas-JackKelly" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/byphilipp"><img src="https://avatars.githubusercontent.com/u/59995258?v=4?s=100" width="100px;" alt="byphilipp"/><br /><sub><b>byphilipp</b></sub></a><br /><a href="#ideas-byphilipp" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://iki.fi/markus.kaukonen"><img src="https://avatars.githubusercontent.com/u/6195764?v=4?s=100" width="100px;" alt="Markus Kaukonen"/><br /><sub><b>Markus Kaukonen</b></sub></a><br /><a href="#question-paapu88" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MoHawastaken"><img src="https://avatars.githubusercontent.com/u/55447473?v=4?s=100" width="100px;" alt="MoHawastaken"/><br /><sub><b>MoHawastaken</b></sub></a><br /><a href="https://github.com/openclimatefix/graph_weather/issues?q=author%3AMoHawastaken" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.ecmwf.int"><img src="https://avatars.githubusercontent.com/u/47196359?v=4?s=100" width="100px;" alt="Mihai"/><br /><sub><b>Mihai</b></sub></a><br /><a href="#question-mishooax" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vitusbenson"><img src="https://avatars.githubusercontent.com/u/33334860?v=4?s=100" width="100px;" alt="Vitus Benson"/><br /><sub><b>Vitus Benson</b></sub></a><br /><a href="https://github.com/openclimatefix/graph_weather/issues?q=author%3Avitusbenson" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dongZheX"><img src="https://avatars.githubusercontent.com/u/36361726?v=4?s=100" width="100px;" alt="dongZheX"/><br /><sub><b>dongZheX</b></sub></a><br /><a href="#question-dongZheX" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sabbir2331"><img src="https://avatars.githubusercontent.com/u/25061297?v=4?s=100" width="100px;" alt="sabbir2331"/><br /><sub><b>sabbir2331</b></sub></a><br /><a href="#question-sabbir2331" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rnwzd"><img src="https://avatars.githubusercontent.com/u/58804597?v=4?s=100" width="100px;" alt="Lorenzo Breschi"/><br /><sub><b>Lorenzo Breschi</b></sub></a><br /><a href="https://github.com/openclimatefix/graph_weather/commits?author=rnwzd" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gbruno16"><img src="https://avatars.githubusercontent.com/u/72879691?v=4?s=100" width="100px;" alt="gbruno16"/><br /><sub><b>gbruno16</b></sub></a><br /><a href="https://github.com/openclimatefix/graph_weather/commits?author=gbruno16" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
