import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold
from compute import hits2skill
from convert import dict2da



def find_best_criterion(skill, dim='probability', metric='f1', tolerance=1e-2, min_spread=True):
    """It searches for the value of a dimension in a dataset that maximizes a skill metric.
    
    Inputs:
    -------
    skill:      xr.Dataset. It contains the arrays of skill for several metrics. At least, it should have the variables  for the chosen target metric (see attribute 'metric'), recall and precision. The function works regardless of the number of dimensions, as long as one of them matches with the dimension to be optimized, defined in the attribute 'dim'
    dim:        string. Name of the dimension in 'skill' that will be optimized
    metric:     string. Name of the skill metric for which the criterium will be optimize. This name should be one of the variables in the Dataset 'ds'. By default, f1
    tolerance:  float. Minimum value of improving skill that is considered in the optimization. All the values of the dimension 'dim' whose skill differs less than this tolerance from the maximum skill are considered candidates. The selection of the best candidate among these values depends on the attribute `min_spread`.
    min_spread: boolean. If True, the selection of the best 'dim' value is based, for those values within the tolerance, on the minimum difference between precision and recall; therefore, if True, the DataArrays 'recall' and 'precision' are required. If False, the minimum among the candidates is selected as the best
    
    Output:
    -------
    xr.Dataset. Matrix that contains 4 variables ('dim', recall, precision, 'metric') correspoding to the optimized values of the dimension 'dim' and the skill corresponding to that value measured in terms of recall, precision and the selected target 'metric'. It has one dimension less than the original Dataset 'ds', since the  dimension 'dim' was removed and optimized.
    """

    # compute skill loss with respect to the maximum
    delta_metric = skill[metric].max(dim) - skill[metric]
    
    # select candidates as the values for which the skill is close enough (within the tolerance) to the maximum skill
    candidates = skill.where(delta_metric < tolerance, drop=True)
    
    if min_spread:
        # compute the precision-recall difference for the candidates
        diff_RP = abs(candidates['recall'] - candidates['precision'])
        # select the value of "dim" that minimize the precision-recall difference
        best_dim = diff_RP.idxmin(dim)
        # extract the skill associated to that value
        best_skill = candidates.where(diff_RP == diff_RP.min(dim)).max(dim)
    else:
        # select the minimum as the best candidate
        mask = ~candidates[metric].isnull()
        best_dim = mask.idxmax(dim)
        # extract the skill associated to that value
        best_skill = candidates.sel({dim: best_dim}).drop(dim)
        
    # merge all the results in a single Dataset
    best_skill[dim] = best_dim
    
    return best_skill



def find_best_criteria(skill, dims=['probability', 'persistence'], metric='f1', tolerance=1e-2, min_spread=[True, False]):
    """It searches for the combination of criteria that maximizes a skill metric.
    
    Inputs:
    -------
    skill:      xr.Dataset. It contains the arrays of skill for several metrics. At least, it should have the variables  for the chosen target metric (see attribute 'metric'), recall and precision.
    dims:       list or string. Name(s) of the dimension(s) in 'skill' that will be optimized
    metric:     string. Name of the skill metric for which the criterium will be optimize. This name should be one of the variables in the Dataset 'ds'. By default, f1
    tolerance:  float. Minimum value of improving skill that is considered in the optimization. For all the highest values of the dimension 'dim' that differ less than this tolerance from the maximum skill, the value that minimizes the difference between recall and precision will be selected.
    min_spread: list or boolean. If True, the selection of the best 'dim' value is based, for those values within the tolerance, on the minimum difference between precision and recall; therefore, if True, the DataArrays 'recall' and 'precision' are required. If False, the minimum among the candidates is selected as the best
    
    Output:
    -------
    skill:       xr.Dataset. A dataset similar to the input 'skill', in which the dimensions 'dims' have been removed and transformed to variables containing the optimized value of each dimension
    """
    
    if isinstance(dims, str):
        dims = [dims]
    if isinstance(min_spread, bool):
        min_spread = [min_spread] * len(dims)
        
    for dim, spread in zip(dims, min_spread):
        skill = find_best_criterion(skill, metric=metric, dim=dim, tolerance=tolerance, min_spread=spread)
        
    return skill



def find_best_criteria_cv(hits, station_events, dims=['probability', 'persistence'], kfold=5, train_size=.8, stratify=True, random_state=0, beta=1, tolerance=1e-2, min_spread=True):
    """A cross-validation version of the function of the function 'find_best_criteria'. It selects the criteria that maximizes the skill over a 'kfold' number of subsamples of the stations
    
    Inputs:
    -------
    hits:                 xarray.Dataset (id, persistence, approach, probability). A boolean matrix of hits, misses and false alarms. It must contain three variables: 'tp' hits, 'fn' misses, 'fp' false alarms
    station_events:       pd.Series. The number of observed events in the set of the stations used for the optimization. It will be used as a covariable in the stratified sampling in order to keep the proportion of events in each of the subsets
    dims:        list or string. Name(s) of the dimension(s) in 'skill' that will be optimized
    kfold:                int. Number of subsets of the stations to be produced
    train_size:           float. It should be between 0.0 and 1.0 and represents the proportion of the dataset to include in the train split
    stratify:             bool. If True, the split sampling is done in a stratified way, so that the proportion of classes in 'station_events' is kept. If False, the sampling is random.
    ramdon_state:         int. The seed in the random selection of samples
    beta:                 float. A coefficient of the f score that balances the importance of misses and false alarms. By default is 1, so misses and false alarms penalize the same. If beta is lower than 1, false alarms penalize more than misses, and the other way around if beta is larger than 1 
    tolerance:            float. Minimum value of improving skill that is considered in the optimization. For all the highest values of the dimension 'dim' that differ less than this tolerance from the maximum skill, the value that minimizes the difference between recall and precision will be selected.
    min_spread: boolean. If True, the selection of the best 'dim' value is based, for those values within the tolerance, on the minimum difference between precision and recall; therefore, if True, the DataArrays 'recall' and 'precision' are required. If False, the minimum among the candidates is selected as the best
    
    Outputs:
    --------
    skill:                xr.DataArray (persistence, approach, probability, kfold). The skill of each of the cross-validation subsets
    best_criteria:        dict. Best set of criteria for each approach  
    """
    
    # compute skill on 'kfold' sets of samples
    skill = {}
    if stratify:
        kfolds = StratifiedShuffleSplit(n_splits=kfold, train_size=train_size, random_state=random_state)
    else:
        kfolds = KFold(n_splits=kfold, random_state=random_state, shuffle=True)
    for i, (train, val) in enumerate(kfolds.split(station_events.index, station_events.values)):

        # convert indexes into station ID
        train = station_events.index[train]

        # subset of the 'hits' dataset with the stations selected for the optimization
        hits_train = hits.sel(id=train).sum('id', skipna=False)

        # skill dataset for optimizing criteria
        skill[i] = hits2skill(hits_train, beta=beta)#.sel(leadtime=min_leadtime).drop('leadtime')

    # concatenate the 'skill_cv' dictionary as xarray.DataArray
    skill = dict2da(skill, dim='kfold')

    # find the best criteria for the average over station sets
    best_criteria = find_best_criteria(skill.mean('kfold'), metric=f'f{beta}', dims=dims, tolerance=tolerance, min_spread=min_spread)
    
    return skill, best_criteria