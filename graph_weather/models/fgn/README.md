# Functional Generative Network (FGN)

## Overview

This is an unofficial implementation of the Functional Generative Network
outlined in [Skillful joint probabilistic weather forecsting from marginals](https://arxiv.org/abs/2506.10772).

This model is heavily based on GenCast, and is designed to make ensemble weather forecasts through a combination of
mutliple trained models, and noise injected into the model parameters during inference.

As it does not use diffusion, it is significantly faster to run than GenCast, while outperforming it on nearly all metrics.