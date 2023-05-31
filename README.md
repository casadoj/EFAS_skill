# EFAS skill assessment
***

Analysis of the skill of [EFAS (Europen Flood Awareness System)](https://www.efas.eu/en) formal flood notifications since the deployment of EFAS v4 (October 2020).


## Structure of the repository

The repository contains six folders:

* [conf](https://github.com/casadoj/EFAS_skill/tree/cleaning/conf) contains the configuration file (_config.yml_) used by all notebooks.
* [data](https://github.com/casadoj/EFAS_skill/tree/cleaning/data) contains the original data used in the analysis (whenever the size is suitable to be stored in GitHub).
* [docs](https://github.com/casadoj/EFAS_skill/tree/cleaning/docs) contains several documents associated to the development of the repository: the EGU documents, presentations in meetings...
* [env](https://github.com/casadoj/EFAS_skill/tree/cleaning/env) contains the file _environment.yml_ with the Conda environment used to run this repository.
* [notbook](https://github.com/casadoj/EFAS_skill/tree/cleaning/notebook) contains the notebooks used to develop the analysis.
* [py](https://github.com/casadoj/EFAS_skill/tree/cleaning/py) contains Python files with functions created during the analysis.
* [results](https://github.com/casadoj/EFAS_skill/tree/cleaning/results) is used to save datasets and plots produced by running the notebooks.


## Data

The analysis is limited to the [EFAS fixed reporting points](https://github.com/casadoj/EFAS_skill/blob/cleaning/data/reporting_points/Station-2022-10-27v12.csv) with a catchment area larger than 500 km² (2357 points).

The original datasets used for the study are:

* EFAS v4 discharge reanalysis (_water balance_). This discharge data was downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home) (CDS) for the complete EFAS domain, and then the time series specific to each reporting point was extracted and saved as NetCDF files in the [_data/discharge/reanalysis/_](https://github.com/casadoj/EFAS_skill/tree/cleaning/data/discharge/reanalysis) folder. Due to file size limitations in GitHub, the original files downloaded from the CDS are not in the repository.
* EFAS v4 discharge forecast. This data was extracted by Corentin from the Meteorological Archival and Retrival System (MARS) and provided as NetCDF files for each forecast date and model (COSMO, ECMWF-HRES, ECMWF-ENS, DWD). Due to the size of these files, the original files are not included in the repository.
* The [discharge return periods](https://github.com/casadoj/EFAS_skill/blob/cleaning/data/thresholds/return_levels.nc) associated to each reporting point. Even though the data set contains several return periods (1.5, 2, 5, 10, 20 years, ...), the analysis only uses the 5-year return period.









