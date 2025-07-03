import torch


class FunctionalGenerativeNetwork(torch.nn.Module):
    """Functional Generative Network (FGN) for weather prediction.

    This class defines a generative model that predicts future weather states
    based on previous observations and noise levels.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        processor: torch.nn.Module,
        decoder: torch.nn.Module,
        noise_dimension: int,
    ):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder

    def forward(self, previous_weather_state: torch.Tensor, num_ensemble: int = 1) -> torch.Tensor:
        """
        Predict the next weather state given the previous weather state

        Run multiple predictions on the same inputs with different noise vectors to get an ensemble

        Args:
            previous_weather_state: Torch tensor
                The previous weather state, shape (batch_size, num_channels, height, width).
            num_ensemble: number of ensemble predictions to make, default is 1

        Returns:
            torch.Tensor: The predicted future weather state, shape (batch_size, num_ensemble, num_channels, height, width).
        """
        predictions = []
        for ensemble in range(num_ensemble):
            noise_vector = torch.randn(
                (previous_weather_state.shape[0], self.noise_dimension),
                device=previous_weather_state.device,
            )
            encoded_state = self.encoder(previous_weather_state)
            # TODO Append in the sin/cos day of year here to the encoded state
            # TODO Processor is only one with the conditional state
            processed_state = self.processor(encoded_state, noise_vector)
            prediction = self.decoder(processed_state)
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)
