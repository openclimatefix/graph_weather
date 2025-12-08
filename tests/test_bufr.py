import pytest
import pandas as pd
from pathlib import Path

from graph_weather.data.bufr_process import BUFR_Process
from graph_weather.data.schema_for_bufr.adpupa import adpupa_obs
from graph_weather.data.nnja_ai import load_nnja_dataset    

# @pytest.mark.skip(reason="Requires local historical BUFR + NNJA-AI parquet files.")
def test_adpupa_bufr_vs_parquet():
    """
    Integration test:
    - Downloads known historical ADPUPA BUFR + NNJA-AI Parquet
    - Decodes BUFR dataclasses
    - Loads Parquet pandas
    - Compares field-by-field

    """

    bufr_path = Path("gdas.t00z.1bamua.tm00.bufr")

    nnja_df = load_nnja_dataset(
        dataset_name="ADPUPA",
        time="2025-12-08T00:00"
    ).to_dataframe()

    assert bufr_path.exists(), f"Missing test BUFR: {bufr_path}"

    processor = BUFR_Process("adpupa")
    decoded = processor.decode_file(str(bufr_path))

    assert len(decoded) > 0, "No ADPUPA soundings decoded from BUFR."
    obs: adpupa_obs = decoded[0]
    # Compare station info
    expected_station = nnja_df["station_id"].iloc[0]
    assert obs.station_id == expected_station

    expected_datetime = nnja_df["datetime"].iloc[0]
    assert obs.datetime == expected_datetime

    assert abs(obs.location.lat - nnja_df["lat"].iloc[0]) < 1e-6
    assert abs(obs.location.lon - nnja_df["lon"].iloc[0]) < 1e-6
    
    # Compare pressure levels
    
    expected_levels = nnja_df.sort_values("level_index")
    actual_pressures = [lvl.pressure_hPa for lvl in obs.mandatory_levels]
    expected_pressures = expected_levels["pressure_hPa"].tolist()

    assert len(actual_pressures) == len(expected_pressures)

    for a, e in zip(actual_pressures, expected_pressures):
        if e is None and a is None:
            continue
        assert abs(a - e) < 1e-3
