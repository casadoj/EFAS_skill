import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm_notebook
from convert import dict2da
from typing import Union, List, Tuple, Dict, Literal


# models
models = {'COS': {'members': 20, 'leadtimes': 22},
          'DWD': {'members': 1, 'leadtimes': 28},
          'EUD': {'members': 1, 'leadtimes': 40},
          'EUE': {'members': 51, 'leadtimes': 40},}



def identify_events(
    discharge: pd.DataFrame,
    upper_bound: pd.Series,
    lower_bound: pd.Series = None
) -> pd.DataFrame:
    """
    It identifies the start of flood events given the discharge series and the thresholds.
    
       * If only the 'upper_bound' is provided, an event is considered a period during which discharge exceeds that upper bound.
    
       * If both 'upper_bound' and 'lower_bound' are provided, an event is considered a period of time during which discharge exceeds for some time the upper bound and never goes below the lower bound.
    
    Parameters:
    -----------
    discharge: pd.DataFrame
        Discharge timeseries for the multiple stations under analysis
    upper_bound: pd.Series
        Discharge upper threshold above which a flood event starts
    lower_bound: pd.Series
        Discharge lower threshold below which a flood event finishes. If not provided, the 'upper_bound' is used to define the end of an event
    
    Returns:
    --------
    events: pd.DataFrame
        Boolean table with Trues at the timesteps when a flood event starts
    """
    
    # make sure the order of the stations is the same in 'discharge' and 'upper_bound'
    discharge, upper_bound = discharge.align(upper_bound, join='inner', axis=1)
    if not all(discharge.columns == upper_bound.index):
        print('ERROR. The columns in "discharge" and the index in "upper_bound" do not match')
        
    # find timesteps at which discharge starts exceeding the upper bound
    exceed_up = (discharge > upper_bound)
    mask_up = (exceed_up.astype(int).diff() == 1)

    if lower_bound is None:
        events = mask_up
    else:
        # make sure the order of the stations is the same in 'discharge' and 'upper_bound'
        discharge, lower_bound = discharge.align(lower_bound, join='inner', axis=1)
        if not all(discharge.columns == lower_bound.index):
            print('ERROR. The columns in "discharge" and the index in "lower_bound" do not match')
            
        # find timesteps at which discharge goes below the lower bound
        exceed_low = (discharge > lower_bound)
        mask_low = (exceed_low.astype(int).diff() == -1)

        events = pd.DataFrame(False, index=discharge.index, columns=discharge.columns)    
        for stn in events.columns:
            if mask_up[stn].sum() > 0:
                # preliminar list of events in the station
                events_stn_pre = events[mask_up[stn]].index
                # definite list of events in the station
                events_stn = [events_stn_pre[0]]
                for event in events_stn_pre[1:]:
                    if any(mask_low[stn].loc[events_stn[-1]:event]):
                        events_stn.append(event)
                # if len(events_preliminar) > len(events):
                #     print(stn)
                #     break
                events.loc[events_stn, stn] = True
                
    return events



def discharge2exceedance(
    model_files: Dict,
    thresholds: xr.DataArray,
    verbose: bool = True
) -> xr.DataArray:
    """
    From a list of NetCDFfiles corresponding to consecutive forecasts, it extracts the data corresponding to a station and it creates a boolean matrix of exceedance (1) non-exceedance (0).
    
    Parameters:
    -----------
    model_files: Dict
        The keys are the different NWP (numerical weather predicitons) models and the values a list of file names
    threshold: xr.DataArray
        Matrix of 1 dimension (station ID) containint the discharge threshold
    
    Returns:
    --------
    exceedance: xarray.DataArray
        A boolean matrix of exceedance/non-exceedance of the threshold
    """
    
    exceedance = {}
    for model, files in tqdm_notebook(model_files.items()):
        dct = {}
        for file in tqdm_notebook(files):
            
            if verbose:
                print(f'{model}\t{file}', end='\r')

            # open dataaray with dicharge data
            dis = xr.open_dataarray(file).isel(time=slice(1, None))
            dis['time'] = dis.time - np.timedelta64(6, 'h')
            # limit the forecast to its maximum leadtime
            if len(dis.time) > models[model]['leadtimes']:
                dis = dis.isel(time=slice(None, models[model]['leadtimes']))
            # reformat the 'time' dimension into 'leadtime'
            dis = dis.rename({'time': 'leadtime', 'stations': 'id'})
            dis['leadtime'] = [timedelta(hours=(i + 1) * 6) for i in range(len(dis.leadtime))]

            # compute mean exceedance over the threshod
            # this steps also select the stations
            exc = dis > thresholds
            if 'member' in exc.dims:
                exc = exc.mean('member')

            # save in the dictionary
            forecast = datetime.strptime(file[-13:-3], '%Y%m%d%H')
            dct[forecast] = exc

        # join the exceedance of all the files of a model into one dataset
        exceedance[model] = xr.Dataset(dct).to_array(dim='forecast', name=model)
        
        print()
        
    # join all the models into one dataset
    exceedance = xr.Dataset(exceedance).to_array(dim='model', name='exceedance')
    
    return exceedance



def recompute_exceedance(
    obs: xr.DataArray,
    pred_high: xr.DataArray,
    pred_low: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    It recomputes exceedances given DataArrays of observed and predicted exceedance based on 2 discharge thresholds
    
    Parameters:
    -----------
    obs: xr.DataArray
        Observed exceedance over the discharge thresholds. It has 3 possible values: 2, exceedance over the higher threshold; 1, exceedance over the lower threshold; 0, non-exeedance
    pred_high: xr.DataArray
        Probability of exceedance (0-1) over the higher discharge threshold
    pred_low: xr.DataArray
        Probability of exceedance (0-1) over the lower discharge threshold
    
    Returns:
    --------
    exceed_obs: xr.DataArray
        The input observed exeedance recomputed with only 2 classes: 1, exceedance; 0, non-exceedance
    exceed_pred: xr.DataArray
        Probability of exceedance (0-1) as a combination of the exceedances for the higher and lower thresholds
    """
    
    # create empty xarray.DataArrays for observed and predicted exceedance
    exceed_obs = np.zeros_like(obs)
    exceed_pred = np.zeros_like(pred_high)

    # if observation exceeds the higher threshold
    mask = (obs == 2).data
    exceed_obs[mask] = 1
    exceed_pred[mask] = pred_low.data[mask]

    # if observation exceeds the lower threshold and some predictions exceed the higher
    mask = ((obs == 1) & (pred_high > 0)).data
    if len(mask.shape) == 4:
        exceed_obs[mask.any(axis=(2, 3))] = 1
    elif len(mask.shape) == 3:
        exceed_obs[mask.any(axis=2)] = 1
    exceed_pred[mask] = pred_low.data[mask]

    # if observation exceeds the lower threshold and none of the predictions exceed the higher
    mask = ((obs == 1) & (pred_high == 0)).data
    if len(mask.shape) == 4:
        exceed_obs[mask.all(axis=(2, 3))] = 0
    elif len(mask.shape) == 3:
        exceed_obs[mask.all(axis=2)] = 0
    exceed_pred[mask] = pred_high.data[mask]

    # if observation does not exceed the lower threshold
    mask = (obs == 0).data
    exceed_obs[mask] = 0
    exceed_pred[mask] = pred_high.data[mask]
    
    # convert into xarray.DataArrays
    exceed_obs = xr.DataArray(exceed_obs, dims=obs.dims, coords=obs.coords)
    exceed_pred = xr.DataArray(exceed_pred, dims=pred_high.dims, coords=pred_high.coords)
    
    return exceed_obs, exceed_pred



def exceedance2events(
    da: xr.DataArray, 
    probability: Union[float, xr.DataArray] = None, 
    persistence: Tuple = (1, 1), 
    leadtime: Union[int, List] = None
) -> xr.DataArray:
    """
    It defines predicted events out of a DataArray of exceendances over a probability threshold. 
    The persistence criterion defines the number of forecasts that must predict an exceedance in order to be considered an event.
    
    Parameters:
    -----------
    da: xr.DataArray
        A matrix of exceedances over probability threshold, or a matrix of probability of exceedance (in that case the attribute 'probability' is required). It must have a dimension called 'leadtime', over which the function will compute persistence
    probability: float or xr.DataArray
        Probability thresholds used to convert a 'da' of exceedance probability into a DataArray of exceedances over threshold. If None, the function implies that 'da' is already a DataArray of exceedances over threshold
    persistence: tuple
        Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
    leadtime: int or list
        A list of lead times (in hours) between which the events will be computed. If an integer, the computation will done from that lead time to the longest lead time in 'da.leadtime'. If None, the computation will be done for every leadtime in 'da.leadtime'
    
    Returns:
    --------
    events: xr.DataArray
        A matrix of predicted events. If 'leadtime' is None or an integer, the dimension 'leadtime' in the input DataArray is collapsed to a single value. If 'leadtime' is a list, the dimension will have as many values as the length of the list.
    """

    # invert 'leadtime' order from longer to shorter lead times
    da = da.isel(leadtime=slice(None, None, -1))

    # compute exceedance over the probability threshold
    if probability is None:
        exceedance = da
    else:
        exceedance = (da >= probability).astype(int)

    # compute persistence (rolling sum over a window exceeds a number of forecast positives)
    events = (exceedance.rolling({'leadtime': persistence[1]}, center=False, min_periods=1).sum() >= persistence[0]) & exceedance
    events = events.isel(leadtime=slice(None, None, -1))

    if leadtime is None:
        events_agg = events.copy()
        for lt in events_agg.leadtime.data:
            events_agg.loc[dict(leadtime=lt)] = events.sel(leadtime=slice(lt, None)).any('leadtime').astype(int)
    else:
        if isinstance(leadtime, int):
            leadtimes = [leadtime, None]
        else:
            leadtimes = leadtime + [None]
        events_agg = {}
        for lt, LT in zip(leadtimes[:-1], leadtimes[1:]):
            aux = events.sel(leadtime=slice(lt + 1, LT))
            events_agg[aux.leadtime.data.min()] = aux.any('leadtime').astype(int)
        events_agg = dict2da(events_agg, dim='leadtime')
            
    return events_agg
        
        
        
# def compute_events2D(da, probability=None, persistence=(1, 1), resample=None, min_leadtime=None):
#     """It defines predicted events out of a DataArray of exceendances over a probability threshold. 
#     The persistence criterion defines the number of forecast that must predict an exceedance in order to be considered an event.
    
#     Inputs:
#     -------
#     da:           xr.DataArray. A matrix of exceedances over probability threshold. It must have a dimension called 'leadtime', over which the function will compute persistence
#     probability:  float or xr.DataArray. Probability thresholds over which a forecast is considered an event
#     persistence:  tuple (a, b). Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
#     resample:     string. Time resolution at which resample the leadtime data, e.g., '12h'.
#     min_leadtime: int. Minimun number of leadtime hours required to raise the notification for an event
    
#     Output:
#     -------
#     As objetcs:
#     events:       xr.DataArray. A matrix of predicted events. The dimension 'leadtime' in the input DataArray (length 20 in the usual case) is collapsed to a single value.
#     As method:
#     exceedance:   xr.DataArray. Matrix of cells that exceed the 'probability' threshold
#     """
    
#     # invert 'leadtime' order from longer to shorter lead times
#     da = da.isel(leadtime=slice(None, None, -1))

#     # compute exceedance over the probability threshold
#     if probability is None:
#         exceedance = da
#     else:
#         exceedance = (da >= probability).astype(int)
#     compute_events2D.exceedance = exceedance.isel(leadtime=slice(None, None, -1))
    
#     # define custom function to apply to rolling window
#     def compute_persistence(da):
#         return da.fillna(0).any('dt').sum('lt')

#     # apply custom function to 2D window of size (2,2)
#     exc_rolling = exceedance.rolling(leadtime=persistence[1], datetime=3, center={'leadtime': False, 'datetime': True}, min_periods=1)
#     exc_rolling = exc_rolling.construct(leadtime='lt', datetime='dt')
#     events = xr.DataArray(np.zeros(exceedance.shape), dims=exceedance.dims, coords=exceedance.coords)
#     for i in range(exc_rolling.shape[0]):
#         for j in range(exc_rolling.shape[1]):
#             loc = dict(leadtime=i, datetime=j)
#             mask = (compute_persistence(exc_rolling[loc]) >= persistence[0]) & exceedance[loc].astype(bool)
#             events[loc] = events[loc].where(~mask, other=1)
#     events = events.isel(leadtime=slice(None, None, -1))
    
#     if resample is not None:
#         # convert 'leadtime' from integer hours to timedelta
#         events['leadtime'] = pd.to_timedelta(events.leadtime, 'h')
#         # resample
#         events = events.resample({'leadtime': resample}).any().astype(int)
#         # reconvert 'leadtime' back to integer hours
#         events['leadtime'] = (events.leadtime / np.timedelta64(1, 'h')).astype(int)
#         return events.sel(leadtime=slice(min_leadtime, None))
#     else:
#         # check if there's any predicted event
#         return events.sel(leadtime=slice(min_leadtime, None)).any('leadtime').astype(int)



def buffer_events(
    da: xr.DataArray, 
    center: bool = True, 
    w: int = 5
) -> xr.DataArray:
    """
    It creates a buffer around the matrix of predicted events to allow for short lags between observation and prediction. 
    It applys a rolling sum of window 'w' to the input matrix. The window function can be centered or not.
    
    Parameters:
    -----------
    da: xr.DataArray
        Matrix of predicted events
    center: bool
        Whereas the rolling sum must be centered or right sided
    w: int
        Width of the rolling sum window
    
    Returns:
    --------
    buffer: xr.DataArray
        Matrix with the same size as the input matrix, but in which the events have been 'enlarged'
    """
    
    if center:
        mp = int(w / 2) + 1
        buffer = (da.rolling({'datetime': w}, center=True, min_periods=mp).sum() > 0).astype(int)
    else:
        mp = 1 # int(w / 2)
        buffer = (da.rolling({'datetime': w}, center=False, min_periods=mp).sum() > 0).astype(int)
        
    return buffer



def count_events(da: xr.DataArray) -> xr.DataArray:
    """
    Given a boolean DataArray of exceedances over probability threshold, it counts the number of events in the timeseries
    
    Parameters:
    -----------
    da:       xr.DataArray
        Boolean matrix of exceedances over probability threshold. It must contain a 'datetime' dimension
    
    Returns:
    --------
    n_events: xr.DataArray
        A matrix with the counts of events over the dimension 'datetime' (which collapses)"""
    
    # compute onsets: difference equal to 1
    onsets = (xr.concat([da.isel(datetime=[0]), da.diff('datetime')], dim='datetime') == 1).astype(int)
    # count events
    n_events= onsets.sum('datetime')#.data
    
    return n_events



def events2hits(
    obs: xr.DataArray,
    pred: xr.DataArray,
    center: bool = True,
    w: int = 1
) -> xr.Dataset:
    """
    It computes the hits, misses and false alarms between two matrixes of observations and predictions.
    To allow for some lags in the predictions, a buffer can be applied by giving the attribute 'w' a value larger than 1
    
    Parameters:
    -----------
    obs: xr.DataArray
        Boolean matrix of observed exceedances over threshold
    pred: xr.DataArray
        Boolean matrix of predicted exceedances over threshold
    center: bool
        Whereas the rolling sum must be centered or right sided
    w: int
        Width of the rolling sum window
    
    Returns:
    --------
    As an object:
    hits: xr.Dataset
        Contains three variables: TP, true positives; FN, false negatives; FP, false positives
    
    As methods:
    buffer:  xr.DataArray
        The buffered matrix of predicted events
    tp:      xr.DataArray
        A timeseries of correctly predicted events
    """
   
    # check that both Dataset have the same length in the matching dimensions
    dims = list(set(obs.dims).intersection(pred.dims))
    for dim in dims:
        dim_min = max(obs[dim].min(), pred[dim].min())
        dim_max = min(obs[dim].max(), pred[dim].max())
        obs.sel({dim: slice(dim_min, dim_max)})
        pred.sel({dim: slice(dim_min, dim_max)})
    
    # buffer the predicted events
    buff = buffer_events(pred, center=center, w=w)
    events2hits.buffer = buff

    # compute the true positive timeseries
    tp = buff.where(obs == 1) # apply observed mask on the buffered prediction
    tp = (tp == 1).astype(int) # ones in the masked array are true positives
    events2hits.true_positives = tp
    
    # number of observed and predicted events
    n_obs, n_pred = [count_events(da) for da in [obs, buff]]
    
    # compute performance metrics
    TP = count_events(tp)
    TP = xr.ufuncs.minimum(TP, n_obs)
    FN = n_obs - TP
    FP = xr.ufuncs.maximum(0, n_pred - TP) #max(0, n_pred - TP)

    return xr.Dataset({'TP': TP, 'FN': FN, 'FP': FP})
            
            
            
def hits2skill(
    hits: xr.Dataset,
    beta: Union[float, List[float]] = 1
) -> xr.Dataset:
    """
    It computes skill metrics (recall, precision and f1) out of a Dataset of hits, misses and false alarms.
    
    Parameters:
    -----------
    hits: xr.Dataset
        It contains three DataArrays with names 'TP' (hits), 'FN' (misses) and 'FP' (false alarms)
    beta: float or list of floats
        A coefficient (or list of coefficients) of the f score that balances the importance of misses and false alarms. By default is 1, so misses and false alarms penalize the same. If beta is lower than 1, false alarms penalize more than misses, and the other way around if beta is larger than 1 
    
    Returns:
    --------
    skill: xr.Dataset
        It contains three DataArrays with the metrics 'recall', 'precision', and 'fbeta' scores.
    """
    
    skill = xr.Dataset({'recall': hits.TP / (hits.TP + hits.FN),
                        'precision': hits.TP / (hits.TP + hits.FP)})
    if isinstance(beta, float) or isinstance(beta, int):
        beta = [beta]
    for b in beta:
        if isinstance(b, int):
            score = f'f{b}'
        else:
            score = f'f{b}'
        skill[score] = (1 +  b**2) * hits.TP / ((1 +  b**2) * hits.TP + b**2 * hits.FN + hits.FP)
    
    return skill



def define_area_ranges(
    area_min: int, 
    area_max: int, 
    scale: Literal['linear', 'log', 'semilog'] = 'semilog'
) -> np.array:
    """
    Define an array of catchment area ranges
    
    Parameters:
    -----------
    area_min: int
        Minimum catchment area
    area_max: int
        Maximum catchment area
    scale: str
        Type of scale: 'linear', 'log', 'semilog'
    
    Returns:
    --------
    areas: np.array
    """
    
    # linear scale
    if scale == 'linear':
        areas = np.arange(area_min, area_max, min_area)
    # logarithmic scale
    elif scale == 'log':
        areas = np.logspace(np.log10(area_min), np.log10(area_max), 100)
    # pseudo-logarithmic scale
    elif scale == 'semilog':
        min_order_magnitude = len(str(area_min)) - 1
        max_order_magnitude = len(str(int(area_max))) + 1
        areas = np.empty((0,))
        for order in np.arange(min_order_magnitude, max_order_magnitude + 1):
            areas = np.hstack((areas, np.array([1, 1.5, 2, 3, 5, 7]) * 10**order))
        areas = areas[(areas >= area_min) & (areas <= areas[areas > area_max][0])]
    else:
        return 'ERROR. Scale must be one of the following: "linear", "log", "semilog".'
    
    return areas.astype(int)



def summarize_by_area(
    station_area: pd.Series, 
    station_events: pd.Series, 
    area_ranges: np.array
) -> pd.DataFrame:
    """
    It calculates the amount of stations and observed events at different catchment area thresholds
    
    Parameters:
    -----------
    station_area: pd.Series
        Catchment area of each of the stations contained in the dimension 'id' of the dataset 'hits'
    station_events: pd.Series
        Number of observed events of each of the stations contained in the dimension 'id' of the dataset 'hits' 
    area_ranges: np.array
        Values of catchment area in which the results will be discretized
    
    Returns:
    --------
    summary: pd.DataFrame
        Summary of number of stations and observed events by catchment area
    """
    
    assert station_area.size == station_events.size, 'The size of the input pd.Series "station_area" and "station_events" must be equal.'
    
    # DataFrame to save the number of stations and observed events for each area range
    summary = pd.DataFrame(index=area_ranges, columns=['n_stations', 'n_events_obs'])

    # compute the previous DataFrame and Dataset
    for area in summary.index:

        # select stations in the catchment area range
        mask_area = (station_area >= area)
        stn_area = station_area.loc[mask_area].index.to_list()
        summary.loc[area, 'n_stations'] = len(stn_area)
        summary.loc[area, 'n_events_obs'] = station_events.loc[stn_area].sum()

    return summary



def hits_by_area(
    hits: xr.Dataset,
    station_area: pd.Series,
    area_ranges: np.array
) -> xr.Dataset:
    """
    Given a Dataset of hits by station ID and the area of the stations, it computes the hits grouped by catchment area threshold
    
    Parameters:
    -----------
    hits: xr.Dataset
        It contains 3 variables: 'TP' true positives, 'FN' false negatives, 'FP' false positives
    station_area: pd.Series
        Catchment area of each of the stations contained in the dimension 'id' of the dataset 'hits'
    area_ranges: np.array
        Values of catchment area in which the results will be discretized
    
    Returns:
    --------
    hits_area: xr.Dataset
        It contains the same 3 variables as the original dataset 'hits', but aggregated by ranges of catchment area
    """
    
    # Dataset to save hits, misses and false alarms by area range
    dims = list(hits.dims)
    dims.remove('id')
    coords = {dim: hits[dim].data for dim in dims}
    dims += ['area']
    coords['area'] = area_ranges
    hits_area = xr.Dataset({var: xr.DataArray(dims=dims, coords=coords) for var in ['TP', 'FN', 'FP']})
    
    # compute the previous Dataset
    for area in tqdm_notebook(hits_area.area.data):
        # select stations in the catchment area range
        mask_area = (station_area >= area)
        stn_area = station_area.loc[mask_area].index.to_list()
        # extract hits for those stations and aggregate
        hits_area.loc[dict(area=area)] = hits.sel(id=stn_area).sum('id', skipna=False)
        
    return hits_area



def limit_leadtime(da: Union[xr.DataArray, xr.Dataset], exp: Literal['NWP', 'COMB']) -> Union[xr.DataArray, xr.Dataset]:
    """
    Given a DataArray or Dataset in which one of its dimensions is 'leadtime', it converts to NaN values at lond lead times for which either the meteorological model doesn't predict, or that the persistence can no be complied with
    
    Parameters:
    -----------
    da:   xr.DataArray or xr.Dataset
        A xarray object that must contain a dimension named 'leadtime' and a dimension named 'persistence'.
    exp: str
        Type of models to be checked: 'NWP' for individual meteorological models, 'COMB' for combinations of models
    
    Returns:
    --------
    da:   xr.DataArray or xr.Dataset
        Same as the input but long lead times may have been converted to NaN
    """
    
    # convert to -999 values at long leadtimes for which the model has no forecast or the persistence is impossible to be met
    if exp == 'NWP':
        for model in da.model.data:
            last_leadtime = models[model]['leadtimes'] * 6
            for persistence in da.persistence.data:
                n_forecasts = int(persistence.split('/')[0]) - 1
                max_leadtime = last_leadtime - n_forecasts * 12
                if max_leadtime < da.leadtime.max():
                    sel = {'model': model, 'persistence': persistence, 'leadtime': slice(max_leadtime + 1, None)}
                    da.loc[sel] = -999  
    elif exp == 'COMB':
        last_leadtime = max([dct['leadtimes'] for model, dct in models.items()]) * 6
        for persistence in da.persistence.data:
            n_forecasts = int(persistence.split('/')[0]) - 1
            max_leadtime = last_leadtime - n_forecasts * 12
            if max_leadtime < da.leadtime.max():
                sel = {'persistence': persistence, 'leadtime': slice(max_leadtime + 1, None)}
                da.loc[sel] = -999
                
    # convert the -999 just created to NaN
    da = da.where(da != -999, other=np.nan)
    
    return da


def compute_skill(
    TP: Union[int, np.ndarray, pd.Series], 
    FN: Union[int, np.ndarray, pd.Series], 
    FP: Union[int, np.ndarray, pd.Series], 
    beta: float = 1
) -> Tuple[Union[int, np.ndarray, pd.Series], Union[int, np.ndarray, pd.Series], Union[int, np.ndarray, pd.Series]]:
    """
    Given values of true positives, false negatives and false positives, compute the skill metrics recall, precision and f-score
    
    Parameters:
    -----------
    TP: int, np.ndarray or pd.Series
        True positives
    FN: int, np.ndarray or pd.Series
        False negatives
    FP: int, np.ndarray or pd.Series
        False positives
    beta: float
        Parameter of the f-score that balances the importance of recall and precision in the computation of the f-score. The default value (1) gives equal value to precision and recall; values lower than 1 prioritize precision; values larger than 1 prioritize recall
        
    Returns:
    --------
    recall: int, np.ndarray or pd.Series
    precision: int, np.ndarray or pd.Series
    fscore: int, np.ndarray or pd.Series
    """
    
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    fscore = (1 +  beta**2) * TP / ((1 +  beta**2) * TP + beta**2 * FN + FP)
    
    return recall, precision, fscore