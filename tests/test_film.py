import torch
from graph_weather.models.layers.film import FiLMGenerator, FiLMApplier


def test_film_shapes():
    batch = 4
    feature_dim = 16
    num_steps = 10
    hidden_dim = 8
    lead_time = 3

    gen = FiLMGenerator(num_steps, hidden_dim, feature_dim)
    apply = FiLMApplier()

    gamma, beta = gen(batch, lead_time, device="cpu")

    assert gamma.shape == (batch, feature_dim)
    assert beta.shape == (batch, feature_dim)

    x = torch.randn(batch, feature_dim, 8, 8)
    out = apply(x, gamma, beta)
    assert out.shape == x.shape
