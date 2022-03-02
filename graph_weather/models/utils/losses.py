import torch

"""

For the loss, paper uses MSE loss, but after some processing steps:

Re-scale each physical variable such that it has unit-variance in 3 hour temporal difference.
i.e. divide temperature data at all pressure levels by sigma_t_3hr, where sigma^2_T,3hr is the variance of the 3 hour change
in temperature, averaged across space (lat/lon + pressure levels) and time (100 random temporal frames).
Motivations: 1. interested in dynamics of system, so normalizing by magnitude of dynamics is appropriate
2. Physicallymeaningful unit of error should count the same whether it is happening at lower or upper levels of the atmosphere

They also rescale by nomical static air density at each pressure level, but did not have strong impact on performance.

When summing loss across the lat/lon grid, use a weight proportional to each pixels area i.e. cos(lat) weighting



"""
