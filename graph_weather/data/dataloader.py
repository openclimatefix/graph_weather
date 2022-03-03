"""

The dataloader has to do a few things for the model to work correctly

1. Load the land-0sea mask, orography dataset, regridded from 0.1 to the correct resolution
2. Calculate the top-of-atmosphere solar radiation for each location at fcurrent time and 10 other
 times +- 12 hours
3. Add day-of-year, sin(lat), cos(lat), sin(lon), cos(lon) as well
3. Batch data as either in geometric batches, or more normally
4. Rescale between 0 and 1, but don't normalize

"""
