"""Training script."""

import json
import time
from typing import Optional

import dask
import torch
import typer
from dask.cache import Cache
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from graph_weather.data.gencast_dataloader import GenCastDataset, BatchedGenCastDataset
from graph_weather.models.gencast import Denoiser, WeightedMSELoss

# comment these the next two lines out to disable Dask's cache
cache = Cache(1e10)  # 10gb cache
cache.register()


atmospheric_features = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
single_features = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    # "sea_surface_temperature",
    "total_precipitation_12hr",
]
static_features = [
    "geopotential_at_surface",
    "land_sea_mask",
]

model_config = {
    "hidden_dims": [10, 10],
    "num_blocks": 2,
    "num_heads": 2,
    "splits": 2,
    "num_hops": 1,
    "use_edges_features": False,
}


def print_json(obj):
    print(json.dumps(obj))


def main(
    resolution: Annotated[int, typer.Option()] = 64,
    max_year: Annotated[int, typer.Option(min=0, max=2022)] = 2018,
    device: Annotated[str, typer.Option()] = "cpu",
    sparse: Annotated[bool, typer.Option()] = True,
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 1,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    shuffle: Annotated[Optional[bool], typer.Option()] = False,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = False,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    dask_threads: Annotated[Optional[int], typer.Option()] = 8,
):
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    data_params = {
        "batch_size": batch_size,
    }

    obs_path = f"gs://weatherbench2/datasets/era5/1959-2022-6h-{resolution}x{resolution//2}_equiangular_conservative.zarr"

    if shuffle is not None:
        data_params["shuffle"] = shuffle
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        data_params["multiprocessing_context"] = "forkserver"
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="synchronous", num_workers=dask_threads)

    run_start_time = time.time()
    print_json(
        {
            "event": "run start",
            "time": run_start_time,
            "data_params": str(data_params),
            "locals": _locals,
        }
    )

    t0 = time.time()
    print_json({"event": "setup start", "time": t0})
    dataset = GenCastDataset(
        obs_path=obs_path,
        atmospheric_features=atmospheric_features,
        single_features=single_features,
        static_features=static_features,
        max_year=max_year,
        time_step=2,
    )

    training_generator = DataLoader(dataset, **data_params)
    """
    dataset_2 = BatchedGenCastDataset(
        obs_path=obs_path,
        atmospheric_features=atmospheric_features,
        single_features=single_features,
        static_features=static_features,
        max_year=max_year,
        time_step=2,
        batch_size=batch_size
    )

    training_generator = dataset_2
   """
    _ = next(iter(training_generator))  # wait until dataloader is ready
    t1 = time.time()
    print_json({"event": "setup end", "time": t1, "duration": t1 - t0})

    t0 = time.time()
    print_json({"event": "model initialization", "time": t0})

    denoiser = Denoiser(
        grid_lon=dataset.grid_lon,
        grid_lat=dataset.grid_lat,
        input_features_dim=dataset.input_features_dim,
        output_features_dim=dataset.output_features_dim,
        **model_config,
        device=device,
        sparse=sparse,
    )

    criterion = WeightedMSELoss(
        grid_lat=torch.tensor(dataset.grid_lat, device=device),
        pressure_levels=torch.tensor(dataset.pressure_levels, device=device),
        num_atmospheric_features=len(dataset.atmospheric_features),
        single_features_weights=torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1], device=device),
    )

    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=1e-3, weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100)

    t1 = time.time()
    print_json({"event": "model initialization end", "time": t1, "duration": t1 - t0})

    for epoch in range(num_epochs):
        e0 = time.time()
        print_json({"event": "epoch start", "epoch": epoch, "time": e0})

        for i, sample in enumerate(training_generator):
            tt0 = time.time()
            print_json({"event": "training start", "batch": i, "time": tt0})

            # Training step
            corrupted_targets, prev_inputs, noise_levels, target_residuals = sample
            print(corrupted_targets.shape)
            denoiser.zero_grad()
            preds = denoiser(
                corrupted_targets=torch.tensor(
                    corrupted_targets, dtype=torch.float32, device=device
                ),
                prev_inputs=torch.tensor(prev_inputs, dtype=torch.float32, device=device),
                noise_levels=torch.tensor(noise_levels, dtype=torch.float32, device=device),
            )
            loss = criterion(
                preds,
                torch.tensor(noise_levels, dtype=torch.float32, device=device),
                torch.tensor(target_residuals, dtype=torch.float32, device=device),
            )
            loss.backward()

            optimizer.step()
            scheduler.step(epoch + i / (365 * 4 // batch_size))

            tt1 = time.time()
            
            print_json(
                {
                    "event": "training end",
                    "batch": i,
                    "time": tt1,
                    "duration": tt1 - tt0,
                    "loss": float(loss),
                }
            )

        e1 = time.time()
        print_json({"event": "epoch end", "epoch": epoch, "time": e1, "duration": e1 - e0})

    run_finish_time = time.time()
    print_json(
        {"event": "run end", "time": run_finish_time, "duration": run_finish_time - run_start_time}
    )


if __name__ == "__main__":
    typer.run(main)
