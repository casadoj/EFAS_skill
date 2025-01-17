# Index

## 1 Observed discharge

The notebook [*1_observed_discharge.ipynb*](1_observed_discharge.ipynb) was meant to analyse the observed discharge time series in the hDMS database (Hydrological Data Management Service). It could be removed, since it has not been used in the end.

## 2 Preprocessing reanalysis discharge

The notebook [*2_reanalysis_preprocessing.ipynb*](2_reanalysis_preprocessing.ipynb) processes the raw EFAS reanalysis discharge data to extract the data necessary for the following steps in the skill analysis.

The raw discharge data was downloaded from the Climate Data Store (CDS) and consists of NetCDF files for every year of the analysis. These NetCDF files contain values for the complete EFAS domain, but the succeeding analysis only require the time series for specific points: the selected reporting points. The code extracts these timeseries and saves the result in a folder of the repository.

In a second step, the discharge timeseries are compared against a discharge return period to create new binary time series of exceedance/non-exceedance over the specified discharge threshold. To account for events in which the peak discharge is close to the threshold, there's an option to create a 3-class exceedance timeseries: 0, non-exceendance; 1, exceedance over the reduced threshold ($0.95\cdot Q_{rp}$); 2, exceedance over the actual threshold ($Q_{rp}$). By default, the reducing factor is $0.95$, but this value can be changed in the parameter `reducing_factor` of the configuration file.

## 3 Preprocessing forecast discharge

The notebook [*3_forecast_preprocessing.ipynb*](3_forecast_preprocessing.ipynb) computes the probability of the forecasted discharge of exceeding a threshold associated with a specific return period.

The input data are:
* The discharge forecast for the complete set of reporting points, numerical weather prediction (NWP) model and the complete study period. This data is saved in NetCDF format in a hard disk; due to its size it cannot be included in the GitHub repository.
* The discharge thresholds associated to each reporting point, from which a specific return period (by default 5 years) will be used.

The output is the probability of exceeding the specified return period. This is a matrix of multiple dimensions: station, NWP model, date-time, and lead time. Actually, 2 matrixes are computed, one related to the probability of exceeding the specified threshold ($Q_{rp}$), and another with the probability of exceeding a slighly lower threshold ($0.95\cdot Q_{rp}$). This lower threshold is used to avoid false positives or false negatives in forecasts very close to the observation, but on opposite sides of the discharge threshold.

The results are saved inside the repository as NetCDF, saving one file per reporting point.

## 4 Calculating the confusion matrix

The notebook [*4_confusion_matrix.ipynb*](4_confusion_matrix.ipynb) computes the hits (true positives, $TP$), misses (false negatives, $FN$) and false alarms (false positives, $FP$) for all the reporting points exceeding a minimum catchment area and the complete study period.

The input data are the time series of probability of exceedance over a discharge threshold both for the reanalysis ("observed") and the forecast.

The confusion matrix ($TP$, $FN$, $FP$) is computed for all the possible combinations of two notificatin criteria: persistence and probability threshold. The values of these two criteria to be tested are defined by the user in the configuration file:
* `hits>criteria>persistence`: is a list of the persistence values to be tested
* `hits>criteria>probability`: is a tuple that defines the probability values to be tested by defining minimum, maximun and the step.

This notebook can be run under two different experiments (`hits>experiment` in the configuration file):

* **NWP** computes the hits individually for each of the four Numerical Weather Prediction (NWP) models used in EFAS.
* **COMB** computes the hits for different combinations of the four NWP above:
    * *1_deterministic_+_1_probabilistic*: one of the deterministic and one of the probabilistic models must detect the event.
    * *model_mean*: a total probability matrix is computed as a simple mean over models.
    * *member_weighted*: the total probability matrix is the weighted mean of the models, where each model gets a weight relative to the number of member it contains.
    * *brier_weighted*: the total probability matrix is the weighted mean of the models, where each model gets a weight relative to its probabilistic skill, measured in terms of Brier score.

In any case, the result is a new set of NetCDF files (one for station) that contains matrixes of hits, misses and false alarms for every combination of the criteria.

## 5 Selection of reporting points

The notebook [*5_select_points.ipynb*](5_select_points.ipynb) does a selection of reporting points based on the correlation between the reanalysis discharge time series. The selection is done on a catchment basis. From every pair of reporting points with a Spearman correlation coefficient larger than a given value, either the upstream or downstream one is kept depending on the value of the attribute `upstream` in the configuration file.

As a result, the notebook generates a folder for each catchment with a series of plots (hydrograph with flood events, correlation matrix, maps of reporting points...), a CSV file with the original and selected number of reporting points and observed events, and a PARQUET file with the table of attributes of the selected reporting points.

## 6 Compute skill

This notebooks computes the notification skill in terms of precision, recall and the target metric —the $f_\beta$ score—, and optimises the notification criteria to maximise this latter metric.

The main inputs are the confusion matrices computed in [notebook 4](4_confusion_matrix.ipynb) and the reporting points selected in [notebook 5](5_select_points.ipynb).

1. Once the skill metrics are computed for all combinations of model, lead time, probability threshold and persistence, the first step is a visual analysis of the results. Several plots compare the influence of each of these dimensions on the notification skill of the system. 
2. After the exploration, it searches for optimal notification criteria (probability threshold and persistence) for each lead time and model. The results of the optimization are compared with the benchmark in several plots, includint maps, line plots and the Röbber diagram.
3. Finally, the notebooks explores the influence of catchment area in the notification skill. First, it analyses the change in skill with catchment area for a fixed notification criteria. Second, the optimization in point 2 is repeated for several catchment areas, trying to find whether a area-depending threshold enhances skill.

The outputs of the notebook are numerous:

* Plots including maps, histograms, line plots and the Röbber diagram.
* A `pickle` file that contains a dictionary with the optimised criteria for each lead time and model.
* A `parquet` file with the selected reporting points and their skill with the optimal criteria.
* A `NetCDF` file with the skill matrix.

This notebook can be run both for the **NWP** experiment —individual analysis of the Numerical Weather Prediction models— of the **COMB** experiment —analysis of different combination of those models—. The parameters controling the computations are in the `skill` section of the configuration file.

## 7 Summarize results

This notebook creates a table comparing the optimised criteria and the associated skill for every `leadtime` and `model` defined in the configuration file. If several f-scores were tested, it will load all the results and compare results depending on the target f-score.

The inputs are the points selected in [notebook 5](5_select_points.ipynb), the confusion matrices calculated in notebook [notebook_4](4_confusion_matrix.ipynb), and the notification criteria optimised in [notebook_6](6_skill.ipynb).

The output of this notebook is a comparative table exported as a CSV file.