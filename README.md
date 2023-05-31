# EFAS skill assessment

Analysis of the skill of [EFAS (Europen Flood Awareness System)](https://www.efas.eu/en) formal flood notifications since the deployment of EFAS v4 (October 2020).

## Data

The analysis is limited to the EFAS fixed reporting points with a catchment area larger than 500 km² (2357 points). The original datasets used for the study are:

* EFAS v4 discharge reanalysis (_water balance_). This discharge data was downloaded from the `[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home) (CDS) for the complete EFAS domain, and then the time series specific to each reporting point was extracted and saved as NetCDF files in the [_data/discharge/reanalysis/_](https://github.com/casadoj/EFAS_skill/tree/cleaning/data/discharge/reanalysis) folder. Due to file size limitations, the original files downloaded from the CDS are not in the repository.
* EFAS v4 discharge forecast. This data was extracted 
