from dataclasses import dataclass 
from typing import Optional 

@dataclass
class GeoPoint:
    lat : float 
    lon : float 
    elevation_m : Optional[float] = None