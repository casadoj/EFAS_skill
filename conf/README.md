# Structure of the configuration file

This document explains the structure of the configuration file. Each section of the configuration file is outlined below with a description of its purpose and the meaning of its keys.

## Reporting_points

This section configures the selection of reporting points that will be used in the skill assessment.

    input: 
        CSV: table of fixed reporting points
        GIS: shapefile of fixed reporting points used to correct river and catchment names, and add the Pfafstetter code
    rivers: optional shapefile of rivers. Only used for plots
    area: minimum catchment area (km²)
    catchments:catchments on which the skill analysis will focus
    KGE: minimum values of KGE that a reporting point must meet to be included in the skill assessment
    selection:
        rho: the maximum correlation coefficient allowed between 2 reporting points
        upstream: whether to give priority to stations upstream (True) or downstream (False) reporting points
    output: directory where the output table of reporting points will be saved
    
## Discharge

This section defines the raw data in the analysis —reanalysis and forecast discharge—, the study period, and the return period that will be considered as the event threshold.

    input:
        reanalysis: directory were the original reanalysis discharge files are stored
        forecast: directory were the original forecast discharge files are stored
    study_period:
        start: start date of the analysis (e.g. 2020-01-01)
        end: end date of the analysis (e.g. 2022-12-31 18:00 )
    return_period:
        input: NetCDF file with the return periods of discharge
        threshold: return period whose exceedance is considered a flood event
        reducing_factor: factor reducing the discharge associated to the previous return period (not necessary)
    output:
        reanalysis: directory where the preprocessed reanalysis discharge will be saved
            
## Exceedance

Where to save the files of probability of exceeding the discharge threshold.

    output:
        reanalysis: directory where the reanalysis exceedance probability will be saved
        forecast: directory where the forecast exceedance probability will be saved

## Hits

This section configures how the contingecy table —hits, misses and false alarms— will be computed. It is here were you define the combinations of notification criteria —probability and persistence— that will be tested.

    experiment: type of analysis to be carried out: NWP, individual assessment of meteo models; COMB, assessment of the combination of models
    criteria:
        probability: search values of the probability criterion: minimum, maximum and step
        persistence: search values for the persistence criterion
    leadtime: integer or list of lead times between which hits will be computed
    window: width of the rolling window used to compute hits (no. of timesteps) to allow for a time shift between forecast and observation
    center: whether the previous rolling window is centered or not
    output: directory where the NetCDF files of hits/misses/false alarms will be stored
    
## Skill

This section configures the computation of notification skill, and specially the parameters of the criteria optimization.

    current_criteria: benchmark criteria
        approach:
        probability:
        persistence:
    leadtime: minimum leadtime (hours) for which notifications will be issued
    area: minimum catchment area (km²) for which notifications will be issued
    beta: coefficient of the fbeta-score (by default is 1)
    optimization:
        kfold: number of splits in the cross-validation optimization. If None, cross-validation will not be applied
        train_size: proportion of the complete set of reporting points in each cross-validation subset
        stratify: whether the sampling should be done randomly (False) or based on the number of "observed" events (True)
        tolerance: criteria with a skill difference lower than this value will be considered equally-performing
        minimize_spread: whether to minimize the difference between precision and recall among criteria performing similarly in terms of fscore
            probability:
            persistence:
    output: directory where results will be saved