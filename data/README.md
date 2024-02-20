# Data

## Discharge

This folder contains discharge time series at the reporting points.

* The subfolder [observed](./observed) is meant to load observed discharge time series to be treated in [notebook 1](../notebook/1_observed_discharge.ipynb). Since the analyses focused only on simulated dischare, this data is not required.
* The subfolder [reanalysis](./reanalysis) contains the discharge time series extracted fron the EFASv4 long run, which should have previously been downloaded into your local machine and referred to in the configuration file (parameter `discharge>input>reanalysis`). This data is produced in [notebook_2](../notebook/2_reanalysis_preprocessing.ipynb).

## GIS

This folder contains shape files used throughout the code. Particulary, a shape file of the fixed reporting points and a shape file of rivers (only required for visualization).

## Reporting points

This folder contains the table of attributes of the EFAS fixed reporting points. It is of special importance the inclusion of the field `KGE` as it may be used to filter the points used in the optimization of the notification criteria.

## Thresholds

This folder contains a NetCDF file with maps of discharge associated with an array of return periods. [Notebook 2](../notebook/2_reanalyis_preprocessing) extracts from these stack of maps the specific discharge thresholds for each fixed reporting point.