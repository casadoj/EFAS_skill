# EFAS skill assessment

Analysis of the skill of [EFAS (Europen Flood Awareness System)](https://www.efas.eu/en) formal flood notifications since the deployment of EFAS v4 (October 2020).


## 1 Structure of the repository

The repository contains seven folders:

* [conf](https://github.com/casadoj/EFAS_skill/tree/main/conf) contains the configuration file (_config.yml_) used by all notebooks.
* [data](https://github.com/casadoj/EFAS_skill/tree/main/data) contains the original data used in the analysis (whenever the size is suitable to be stored in GitHub).
* [docs](https://github.com/casadoj/EFAS_skill/tree/main/docs) contains several documents associated to the development of the repository: the EGU documents, presentations in meetings...
* [env](https://github.com/casadoj/EFAS_skill/tree/main/env) contains the file _environment.yml_ with the Conda environment used to run this repository.
* [notebook](https://github.com/casadoj/EFAS_skill/tree/main/notebook) contains the notebooks used to develop the analysis.
* [py](https://github.com/casadoj/EFAS_skill/tree/main/py) contains Python files with functions created during the analysis.
* [results](https://github.com/casadoj/EFAS_skill/tree/main/results) is used to save datasets and plots produced by running the notebooks.


## 2 Data

The analysis is limited to the [EFAS fixed reporting points](https://github.com/casadoj/EFAS_skill/blob/data/reporting_points/Station-2022-10-27v12.csv) with a catchment area larger than 500 km² (2357 points).

The original datasets used for the study are:

* EFAS v4 discharge reanalysis (_water balance_). This discharge data was downloaded from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home) (CDS) for the complete EFAS domain, and then the time series specific to each reporting point was extracted and saved as NetCDF files in the [_data/discharge/reanalysis/_](https://github.com/casadoj/EFAS_skill/tree/main/data/discharge/reanalysis) folder. Due to file size limitations in GitHub, the original files downloaded from the CDS are not in the repository.
* EFAS v4 discharge forecast. This data was extracted by Corentin from the Meteorological Archival and Retrival System (MARS) and provided as NetCDF files for each forecast date and model (COSMO, ECMWF-HRES, ECMWF-ENS, DWD). Due to the size of these files, the original files are not included in the repository.
* The [discharge return periods](https://github.com/casadoj/EFAS_skill/blob/data/thresholds/return_levels.nc) associated to each reporting point. Even though the data set contains several return periods (1.5, 2, 5, 10, 20 years, ...), the analysis only uses the 5-year return period.


## 3 Methods

The whole analysis consists of 4 (5) major steps:

### 3.1 Preprocess the discharge reanalysis

This step is carried out in this [notebook](notebook/2_reanalysis_preprocessing.ipynb). The "observed" discharge time series for each reporting point is compared against its defined return period ($Q_{rp}$) to produce time series of exceedance over threshold. In principle, the time series of exceedance should be binary (0, non-exceedance; 1, exceedance); however, to allow for minor deviations between "observed" and forecasted discharge, a reducing factor ($\lambda$) can be used to create ternary time series of exceedance:

* 0: $Q \lt \lambda \cdot Q_{rp}$
* 1: $\lambda \cdot Q_{rp} \le Q \lt Q_{rp}$
* 2: $Q_{rp} \le Q$).

Parameters in the [configuration file](config/config_COMB_leadtime_ranges.yml) specifically involved in this step:

* `discharge>return_period>threshold`: return period (years) associated to the discharge threshold ($Q_{rp}$).
* `discharge>return_period>reducing_factor`: it not None, a value between 0-1 that reduces the discharge threshold ($Q_{rp}$) in order to produce the 3-class exceedance time series explained above.
* `discharge>return_period>input`: location of the NetCDF file with the discharge associated to several return period for all the fixed reporting points.
* `exceedance>output>forecast` is the directory where the output of this step will be saved.


### 3.2 Preprocess the forecast discharge

This [notebook](notebook/3_forecast_preprocessing.ipynb) preprocesses the discharge forecasts. The objective is the same as in the previous step, i.e., to create a data set of exceedances over threshold, but in this case for the forecasts. The procedure is, however, a bit more complex since it involves overlapping forecasts from 4 numerical weather predictors (NWP) that, in some cases, have several runs (members) in every forecast.

As in the reanalysis, the output of the forecast preprocessing are NetCDF files with the time series of exceedance over threshold. Depending on whether the `reducing_factor` is enabled or not, the NetCDF files will contain one or two variables: the exceedance over the discharge threshold ($Q_{rp}$), and, if applicable, the exceedance over the reduced discharge threshold ($\lambda \cdot Q_{rp}$). In any case, the dataset contains values in the range 0-1 with the proportion of model runs (members) that exceeded the specific discharge threshold. For the deterministic NWP (DWD and ECMWF-HRES) values can only be either 0 or 1.

Parameters in the [configuration file](config/config_COMB_leadtime_ranges.yml) specifically involved in this step:

* `discharge>return_period>threshold`: return period (years) associated to the discharge threshold ($Q_{rp}$).
* `discharge>return_period>reducing_factor`: if not None, a value between 0-1 that reduces the discharge threshold ($Q_{rp}$) in order to produce the 3-class exceedance time series explained above.
* `discharge>return_period>input`: location of the NetCDF file with the discharge associated to several return periods for all the fixed reporting points.
* `exceedance>output>forecast` is the directory where the output of this step will be saved.

### 3.3 Confusion matrix

This [notebook](notebook/4_confusion_matrix.ipynb) compares the exceedance over threshold for both the reanalyses (observation) and the forecast, and computes the entries of the confusion matrix (hits, misses, false alarms) that will be later on used to compute skill.

![Figure 1. Confusion matrix for an imbalanced classification, such as that of flood forecasting.](confusion_matrix.JPG)
>***Figure 1**. Confusion matrix for an imbalanced classification, such as that of flood forecasting.*

The first step in this section is to **reshape the forecast exceedance matrix**. Originally this matrix has, for each station and NWP model, the dimensions _forecast_ (in date and time units and a frequency of 12 hours) and _leadtime_ (in hours with frequency 6 hours). These dimensions cannot be directly compared with the _datetime_ dimension in the reanalysis dataset (date and time units and a frequency of 6 hours). Hence, the forecast dataset needs to be reshaped into two new dimensions: _datetime_ (same units and frequency as _datetime_ in the reanalysis data) and _leadtime_ (in hours but with frequency 12 h, instead of 6 h as originally). A thorough explanation of this step can be found in this [document](notebooks/extra/explanation_of_the_procedure.ipynb).

If the exceedance datasets are ternary (see Sections 3.1 and 3.2), the second step in this section is to **recompute the exceedance** to convert these ternary datasets into binary. The combination of 2 ternary time series has 9 possible outcomes. In a nutshell, only two cases are interesting: when one of the time series is over the discharge threshold ($Q_{rp}$) and the other one is just over the reduced discharge threshold ($\lambda \cdot Q_{rp}$). These two cases would be either a miss or a false alarm in a binary analysis; instead, in the ternary analysis they will be both considered as hits.

The third step is to **compute total exceedance probability** out of the probabilities from each of the 4 NWP. Four aproaches are tested:

* _1 deterministic and 1 probabilistic_: this is the procedure currently used in EFAS, in which a notification is sent if both a deterministic NWP and an probabilistic NWP predict the flood with an exceedance probability over a threshold.
* _Model mean_: the total exceedance probability is a simple mean over the probability of all the models. This approach gives the same weight to every model.
* _Member weighted_: the total exceedance probability is a mean over all model runs. In this approach the probabilistic models prevail over the deterministic, since they have more than one run.
* _Performance weighted_: the weighted mean is done using a weighing matrix based of the Brier score, which gives different weigths to every model at every lead time.

**Forecasted events** (i.e. notifications) are computed by comparing the total exceedance probability matrix against a vector of possible probability thresholds. It is in this step that we include _persistence_ as a notification criteria. The forecasted events are calculated for the series of persistence values specified in the configuration file.

Finally, the **hits, misses, and false alarms** are computed from the comparison between the "observed" and the forecasted events. The results are saved as NetCDF file, one for each reporting point. Every NetCDF file contains 3 matrixes ($TP$ for true positives or hits, $FN$ for false negatives of misses, $FP$ for false positives or false alarms) with 4 dimensions (_approach_, _probability_, _persistence_, _leadtime_).

Parameters in the [configuration file](config/config_COMB_leadtime_ranges.yml) specifically involved in this step:

* `hits>criteria>probability`: an array of probability values (in the range 0-1) that will be tested.
* `hits>criteria>persistence`: an array of persistence values to be tested. Every persistence criterion is a pair of values (`[x, y]`) representing the number of $x$ positive forecast of a window of width $y$. For instance, a persistence of `[2, 3]` means that a notification would be sent if 2 out of 3 forecast predict the event.
* `hits>window` and `hits>center`: width, and location of the rolling window used to compute hits. This rolling window is a buffer applied on the predicted events to account for events predicted with a time shift with respect to the observation. The width must be an integer representing the amount of timesteps (12 h each), and the centre is a boolean.
* `hit>seaonality`: a boolean parameter to set if the analysis should be seasonal or not. _(not working yet)_
* `hits>output`: directory where the resulting NetCDF files will be stored.

### 3.4 Selection of reporting points

In a first attempt, we tried to remove the spatial colinearity between reporting points. The idea was that the reporting points in the same catchment might be highly correlated, so including all of them in the skill analysis would not be correct. With that idea in mind, there is a [notebook](notebook/5_select_points.ipynb) that analyses the reporting points in a catchment basis and filters out highly correlated points. 

In the end, this step has been removed from the pipeline due to the limited amount of data that we have, which would be even smaller if we removed more reporting points. 

This filter could be done in order to keep either smaller or larger catchments, in either case, this filter would have hinder the skill analysis based on catchment area that will be part of the final results.

### 3.5 Skill assessment

This is the [notebook](notebooks/6_skill.ipynb) in which we analyse the skill of EFAS notifications in the last 2 years and derive ways of changing the notification criteria in order to optimize skill. The outcome of this process is a set of plots and a few datasets including the optimized criteria and the table of reporting points including their skill for the optimial criteria.

In this section we **compute skill** out of the hits, misses and false alarms derived in the previous section. Skill is measured in three different ways: $recall$, $precision$ and a combination of those named $f_{score}$. The $\beta$ coefficient in the $f_{score}$ is one of the parameters to be set in the configuration file. The default values is 1, for which the same importance is given to both $precision$ and $recall$. If $precision$ is deemed more importance, $\beta$ should be lower than 1, and the other way around if $recall$ is more important.

$$recall = \frac{TP}{TP + FN}$$

$$precision = \frac{TP}{TP + FP}$$

$$f_{beta} = \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + \beta^2 \cdot FN + FP}$$

Two plots are generated that show, respectively, the evolution of the hits and the skill regarding persistence, lead time, probability threshold and approach. A third plot shows especifically the evolution of skill for the fixed lead time (default 60 h) and catchment area (default 2000 km²) that will be used in the optimization.

After the previous exploration, **the criteria are optimized for a fixed lead time and catchment area**. A new set of criteria is derived for each of the approaches used to compute total exceedance probability (see Section 3.3). With these new sets of criteria maps and lineplots are generated to show the results and improvements compared to the current notification criteria.

Finally, we analyse the **behaviour of the skill with varying catchment area** (for a fixed lead time) **and varying lead time** (for a fixed catchment area). Not only we compare the new optimal criteria against the current, but we rerun a optimization in which we look for the optimal probability threshold for each cathcment/lead time value. The objective of this second optimization is only exploratory, to check whether there is ground for improvement in the skill of the system with more complex notification criteria.

Parameters in the [configuration file](config/config_COMB_leadtime_ranges.yml) specifically involved in this step:

* `skill>current_criteria`: as a benchmark, the current _approach_, _probability_ and _persistence_ criteria should be provided.
* `sekill>leadtime`: minimum leadtime value for which the notification criteria will be optimized. By default is 60 h, to keep the current procedure of not sending notifications with less than 2 days in advance.
* `skill>area`: minimum catchment area for wich the notification criteria will be optimized. By default is 2000 km², so that the results of the optimization can be compared with the benchmark.
* `skill>beta`: the coefficient of the $f_{beta}$ score used as the target metric in the optimization. 
* `skill>optimization>kfold`: the number of split samples used in the optimization process. If None, the optimization will be done on the training set of stations and validated on the test set. If an integer is provided, the optimization will be repeated on $k_{fold}$ equal-size subsamples of the training set, averaged, and then validated on the test set.
* `skill>optimization>train_size`: a value between 0 and 1 that defines the proportion of reporting points to be included in the training sample.
* `skill>optimization>tolerance`: a float number that defines the skill difference required to consider one criteria better than the other. If two sets of criteria get a $f_{score}$ closer than this tolerance, the two sets are considered equally-performing; in that case, the definition of the best set depends on the followint paramenter.
* `skill>optimization>minimize_spread`: it controls how to define the best value of the criteria out of those performing equally. If True, the selected value is the one with a minimum difference between precision and recall. If False, the minimum value is selected as the best. This parameter needs to be defined for each criterion, i.e., `probability` and `persistence`.

There is a final [notebook](noteboooks/7_summarize_results.ipynb) that imports the datasets of hits and the optimized criteria and exports a table that summarizes the results of the analysis.

### 3.6 Extras

There are [2 extra notebooks](notebooks/extra/) that were used to explain the whole procedure and generate plots regarding specific events.

## 4 Results

This [Confluence page](https://efascom.smhi.se/confluence/pages/viewpage.action?spaceKey=EJ&title=EFAS+skill+assessment) is a report of the complete study, including the analysis of the results. A PDF version can be found in the folder [docs](https://github.com/casadoj/EFAS_skill/blob/main/docs/Confluence_report.pdf).














