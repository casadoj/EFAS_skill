# Index

## Observed discharge

The notebook [*1_observed_discharge.ipynb*](https://github.com/casadoj/EFAS_skill/blob/main/notebook/1_observed_discharge.ipynb) was meant to analyse the observed discharge time series in the hDMS database (Hydrological Data Management Service). It could be removed, since it has not been used in the end.

## Preprocessing reanalysis discharge

The notebook [*2_reanalysis_preprocessing.ipynb](https://github.com/casadoj/EFAS_skill/blob/main/notebook/2_reanalysis_preprocessing.ipynb) processes the raw EFAS reanalysis discharge data to extract the data necessary for the following steps in the skill analysis.

The raw discharge data was downloaded from the Climate Data Store (CDS) and consists of NetCDF files for every year of the analysis. These NetCDF files contain values for the complete EFAS domain, but the succeeding analysis only require the time series for specific points: the selected reporting points. The code extracts these timeseries and saves the result in a folder of the repository.

In a second step, the discharge timeseries are compared against a discharge return period to create new binary time series of exceedance/non-exceedance over the specified discharge threshold. To account for events in which the peak discharge is close to the threshold, there's an option to create a 3-class exceedance timeseries: 0, non-exceendance; 1, exceedance over the reduced threshold ($0.95\cdot Q_{rp}$); 2, exceedance over the actual threshold ($Q_{rp}$). By default, the reducing factor is $0.95$, but this value can be changed in the parameter `reducing_factor` of the configuration file.