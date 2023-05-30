import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tqdm import tqdm_notebook

# models
models = {'COS': {'members': 20, 'leadtimes': 22},
          'DWD': {'members': 1, 'leadtimes': 28},
          'EUD': {'members': 1, 'leadtimes': 40},
          'EUE': {'members': 51, 'leadtimes': 40},}


def identify_events(discharge, upper_bound, lower_bound=None):
    """It identifies the start of flood events given the discharge series and the thresholds.
    
       * If only the 'upper_bound' is provided, an event is considered a period during which discharge exceeds that upper bound.
    
       * If both 'upper_bound' and 'lower_bound' are provided, an event is considered a period of time during which discharge exceeds for some time the upper bound and never goes below the lower bound.
    
    Inputs:
    -------
    discharge:   pandas.DataFrame (timesteps, stations). Discharge timeseries for the multiple stations under analysis
    upper_bound: pandas.Series (stations,). Discharge upper threshold above which a flood event starts
    lower_bound: pandas.Series (stations,). Discharge lower threshold below which a flood event finishes. If not provided, the 'upper_bound' is used to define the end of an event
    
    Output:
    -------
    events:      pandas.DataFrame (timesteps, stations). Boolean table with Trues at the timesteps when a flood event starts
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

        
    
# def compute_exceedance(files, station, threshold):
    # """From a list of files (NetCDF) corresponding to consecutive forecast, it extracts the data corresponding to a station and it creates a boolean matrix of exceedance (1) non-exceedance (0).
    
    # Inputs:
    # -------
    # files:      list. NetCDF files corresponding with EFAS' discharge forecast
    # station:    str. Station ID
    # threshold:  float. Discharge threshold
    
    # Output:
    # -------
    # A xarray.DataArray with a boolean matrix of exceedance/non-exceedance of the threshold. If the forecast is deterministic, the output DataArray will have two dimensions (number of forecast, number of timesteps in each forecast). If the forecast is probabilistic, the output DataArray will have three dimensions (no. forecast, no. memebers, no.timesteps)
    # """
    
    # load an example file to extract dimensions
    # aux = xr.open_dataarray(files[0])
    # n_time = len(aux.time)
    # if 'member' in aux.dims:
        # member = aux.member.data
        # arr = np.empty((len(files), len(member), n_time))
    # else:
        # arr = np.empty((len(files), n_time))
        # if 'member' in locals():
            # del n_member
    # forecast = []
    # leadtime = [timedelta(hours=(i + 1) * 6) for i in range(n_time)]
    # aux.close()

    # read each file and compute exceedance
    # for i, file in enumerate(files):
        # compute exceedance of the threshold
        # da = xr.open_dataarray(file)
        # exc = da.sel(stations=station) >= threshold
        # if len(exc.shape) == 2:
            # arr[i,:,:] = exc
        # elif len(exc.shape) == 1:
            # arr[i,:] = exc
        # da.close()

        # add forecast to the list
        # forecast.append(datetime.strptime(file[-13:-3], '%Y%m%d%H'))
        
    # build the xarray.DataArray
    # if len(arr.shape) == 3:
        # return xr.DataArray(arr, dims=('forecast', 'member', 'leadtime'), coords={'forecast': forecast, 'member': member, 'leadtime': leadtime})
    # elif len(arr.shape) == 2:
        # return xr.DataArray(arr, dims=('forecast', 'leadtime'), coords={'forecast': forecast, 'leadtime': leadtime})
    
    

def compute_exceedance(model_files, thresholds, verbose=True):
    """From a list of files (NetCDF) corresponding to consecutive forecast, it extracts the data corresponding to a station and it creates a boolean matrix of exceedance (1) non-exceedance (0).
    
    Inputs:
    -------
    model_files: dictionary. The keys are the different NWP (numerical weather predicitons) models and the values a list of file names
    threshold:   xarray.DataArray (id,). Matrix of 1 dimension (station ID) containint the discharge threshold
    
    Output:
    -------
    A xarray.DataArray with a boolean matrix of exceedance/non-exceedance of the threshold. This DataArray has 4 dimensions: id, model, forecast, leadtime
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

    
    
# def deterministic_criteria(da, leadtime=4):
#     """Given a boolean DataArray with the set of forecast of a deterministic model, it computes if the the formal notification criteria for deterministic models is fulfilled
    
#     * Lead time >= 48 h (skip fisrt 4 timesteps since the temporal resolution is 6 h)
#     * Any timestep exceeds Q5
    
#     Input:
#     ------
#     da:        DataArray (forecast, leadtime). Set of forecast containing boolean data of exceedance/non-exceedance of the 5 year return period of discharge
#     leadtime:  int. Amount of timesteps to skip at the beginning of each forecast.
    
#     Output:
#     -------
#     da:        DataArray (forecast,). Boolean DataArray that defines whether a forecast fulfills the notification criteria or not
#     """
      
#     return da.isel(leadtime=slice(leadtime, None)).any('leadtime')


# def probabilistic_criteria(da, leadtime=4, probability=.3, persistence=3):
#     """Given a boolean DataArray with the set of forecast of a probabilistic model, it computes if the the formal notification criteria for probabilistic models is fulfilled
    
#     * Lead time >= 48 h (skip fisrt 4 timesteps since the temporal resolution is 6 h)
#     * The 5 year return period is exceeded with a given probability during a number of consecutive forecasts.
    
#     Input:
#     ------
#     da:          DataArray (forecast, member, leadtime). Set of forecast containing boolean data of exceedance/non-exceedance of the 5 year return period of discharge
#     leadtime:    int. Amount of timesteps to skip at the beginning of each forecast
#     probability: float. Probability threshold required to send a notification
#     presistence: int. Number of consecutive forecast that must exceed the probability threshold so that a notification is issued
    
#     Output:
#     -------
#     da:        DataArray (forecast,). Boolean DataArray that defines whether a forecast fulfills the notification criteria or not
#     """
    
#     aux = (da.isel(leadtime=slice(leadtime, None)).mean('member') > probability).any('leadtime')
#     aux = aux.rolling(forecast=persistence).sum() == persistence
    
#     return aux


# def compute_notifications(exceedance, leadtime=4, probability=.3, persistence=3):
#     """Given a dictionary with the forecasts of the 2 deterministic modeles (EUD, DWD) and the 2 probabilistic models (EUE, COS), it computes all the notification criteria and creates a boolean DataArray with the issuance of formal notifications.
    
#     Input:
#     ------
#     exceedance:  dict. Contains a DataArray for each model with the exceedance/not exceedance of the 5 year return period discharge.
#     leadtime:    int. Amount of timesteps to skip at the beginning of each forecast
#     probability: float. Probability threshold required to send a notification
#     presistence: int. Number of consecutive forecast that must exceed the probability threshold so that a notification is issued
        
#     Output:
#     -------
#     da:        DataArray (forecast,). Boolean DataArray that defines whether a formal notification should be issued or not
#     """
    
#     # DETERMINISTIC FORECASTS
    
#     # calculate forecast in each model that comply with the notification criteria
#     EUD = deterministic_criteria(exceedance['EUD'], leadtime)
#     DWD = deterministic_criteria(exceedance['DWD'], leadtime)
#     # combine models
#     deterministic = xr.concat((EUD, DWD), 'model').any('model')
    
#     # PROBABILISTIC FORECASTS
    
#     # calculate forecast in each model that comply with the notification criteria
#     EUE = probabilistic_criteria(exceedance['EUE'], leadtime, probability, persistence)
#     COS = probabilistic_criteria(exceedance['COS'], leadtime, probability, persistence)
#     # combine models
#     probabilistic = xr.concat((EUE, COS), 'model').any('model')
    
#     # COMBINE FORECASTS
#     notifications = xr.concat((deterministic, probabilistic), 'type').all('type')
    
#     return notifications


def area_increment(area):
    """It computes the increment in catchment area between contiguous points.
    
    Input:
    ------
    area:     pd.Series. Catchment area of the points of interest. All the points should belong to the same river.
    
    Output:
    -------
    area_inc: pd.Series. Area increment between contiguous points. The point with largest catchment area has no value.
    """
    
    # sort in descending order
    area = area.sort_values(ascending=False)
    # increment in catchment area between contiguous points
    area_diff = abs(area.diff())
    area_inc = area_diff / area
    
    return area_inc



def filter_points(stations, threshold=.1, verbose=False):
    """From a table of stations including river and catchment area, it removes those whose catchment area does not increase more than a threshold compared with the station directly downstream
    
    Inputs:
    -------
    stations:      pd.DataFrame. Table of original stations. It must contain two fields: 'river' with the river name, 'area' with the catcment area
    threshold:     float. Increase in catchment area required to keep a stations. It must be a value larger than 0
    verbose:       boolean. Whether to show on screen the process or not
    
    Output:
    -------
    stations_sel:  pd.DataFrame. Table of stations that comply with the established threshold
    """
    
    # list of selected stations
    filter_stns = []
    
    # analyse river by river
    for river in stations.river.unique():
        # extract stations in the river
        stns_river = stations.loc[stations.river == river].sort_values('area', ascending=False).copy()
        n_stns_orig = stns_river.shape[0]
        # check stations one by one from downstream to upstream
        for stn in stns_river.index[1:]:
            stns_river['area_inc'] = area_increment(stns_river.area)
            # remove station if it doesn't comply with the threshold
            if stns_river.loc[stn, 'area_inc'] < threshold:
                stns_river.drop(stn, axis=0, inplace=True)
        
        if verbose:
            print(river)
            print('-' * len(river))
            print('no. original stations:\t{0}'.format(n_stns_orig))
            print('no. filtered stations:\t{0}'.format(stns_river.shape[0]), end='\t\t')
        
        # add seleted stations to the list
        filter_stns += stns_river.index.tolist()
        
    # filter stations
    stations_sel = stations.loc[filter_stns]
    # if verbose:
    print('Total no. original stations:\t{0}'.format(stations.shape[0]))
    print('Total no. filtered stations:\t{0}'.format(stations_sel.shape[0]))

    return stations_sel     
        
        

                
                
                
def filter_correlation_matrix(correlation_matrix, rho=.9, inplace=False):
    """This functionis used to filter a correlation matrix based on a certain threshold. It takes in 3 parameters:
    
    correlation_matrix : a DataFrame that represents a correlation matrix.
    rho : a float value between -1 and 1 that represents the threshold of correlation coefficient, default value is 0.9
    inplace : a boolean variable that indicates if the input matrix is to be modified inplace, default value is False
    """
    
    if inplace:
        cm = correlation_matrix
    else:
        cm = correlation_matrix.copy()
    
    # remove the upper diagonal
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if j >= i:
                cm.iloc[i,j] = np.nan
            
    # remove highly correlated stations
    for stn in cm.index:
        if (cm.loc[stn] >= rho).any():
            cm.drop(stn, axis=0, inplace=True)
            cm.drop(stn, axis=1, inplace=True)
            
    if inplace is False:
        return cm
    
    
    
def filter_correlation_matrix(correlation_matrix, rho=.9):
    """This functionis used to filter a correlation matrix based on a certain threshold. It takes in 3 parameters:
    
    correlation_matrix : a DataFrame that represents a correlation matrix.
    rho : a float value between -1 and 1 that represents the threshold of correlation coefficient, default value is 0.9
    """
        
    cm = correlation_matrix.copy()
    
    # remove the upper diagonal
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if j >= i:
                cm.iloc[i,j] = np.nan

    # remove highly correlated stations
    for stn in cm.index[::-1]:
        try:
            mask = cm.loc[stn] > rho
            if mask.sum() > 0:
                cm = cm.loc[~mask, ~mask]
        except:
            continue
    
    return cm




    
    
    
def dataarray_events(events, forecast, leadtime):
    """It creates a binary, 2D DataArray describing the onset of a flood event.
    
    Inputs:
    -------
    events:    numpy.array or list. Array of timestamps representing the moments at which a flood event started
    forecast:  numpy.array or xarray.DataArray. Array of forecast values (timestamps) that will represent the 1st dimension in the output DataArray
    leadtime:  numpy.array or xarray.DataArray. Array of leadtime values (timedelta64) that will represent the 2nd dimension in the output DataArray
    
    Output:
    -------
    events_da: xarray.DataArray (forecasts, leadtimes). Binary DataArray with ones at the times (forecast + leadtime) when a flood event started
    """
    
    ll, ff = np.meshgrid(leadtime, forecast)
    
    # create a binary DataArray with 1 where the
    events_da = xr.DataArray(np.in1d(ff + ll, events).reshape((len(forecast), len(leadtime))),
                             dims=['forecast', 'leadtime'], coords=[forecast, leadtime])
    
    return events_da



def compute_performance(ds, events):
    """It computes performance metrics of a two binary DataArrays.
    
    Inputs:
    -------
    ds:     xr.Dataset (forecast, leadtime). Each variable contains a binary, 2D DataArray with the predicted starts of events
    events: xr.DataArray (forecast, leadtime). Binary, 2D DataArray with the observed starts of events
    
    Output:
    -------
    df:     pandas.DataFrame. Table with performance metrics for every variable in the 'ds' object. 'FP' false positive, 'FN' false negative, 'TP' true positives
    """
    
    for coord in list(ds.coords):
        if (ds[coord] != events[coord]).any():
            print('ERROR. Coordinate {0} does not match between the two datasets.')
        
    df = pd.DataFrame(index=list(ds), columns=['FP', 'FN', 'TP', 'f1', 'recall', 'precision'])
    for variable, da in ds.data_vars.items():
        y_true = events.astype(int).data.flatten()
        y_pred = da.astype(int).data.flatten()
        df.loc[variable, ['FP', 'FN', 'TP']] = confusion_matrix(y_true, y_pred).flatten()[1:]
        df.loc[variable, 'f1'] = f1_score(y_true, y_pred)
        df.loc[variable, 'recall'] = recall_score(y_true, y_pred)
        df.loc[variable, 'precision'] = precision_score(y_true, y_pred)

    return df



def compute_f1(y_true, y_pred, window=0):
    """It computes performance metrics of a two binary DataArrays.
    
    Inputs:
    -------
    y_true: xr.DataArray (forecast, leadtime). Binary, 2D DataArray with the observed starts of events
    y_pred: xr.Dataset (forecast, leadtime). Each variable contains a binary, 2D DataArray with the predicted starts of events

    Output:
    -------
    df:     pandas.DataFrame. Table with performance metrics for every variable in the 'ds' object. 'FP' false positive, 'FN' false negative, 'TP' true positives
    """
    

    for coord in list(y_pred.coords):
        if (y_pred[coord] != y_true[coord]).any():
            print('ERROR. Coordinate {0} does not match between the two datasets.')

    ws = range(-window, window + 1)
    df = pd.DataFrame(index=list(y_pred), columns=ws)

    true = y_true.astype(int).data.flatten()
    for var, da in y_pred.data_vars.items():
        for w in ws:
            pred = da.shift(leadtime=w, fill_value=False).astype(int).data.flatten()
            df.loc[var, w] = f1_score(true, pred)

    return df
            
            
            

        
        
        
def compute_metrics(hits, dims_agg=None, beta=1):
    """It computes a Dataset of metrics (f1, precision and recall) out of a Dataset of hits (variable 'tp'), misses (variable 'fn') and false alarms (variable 'fp'). If desired, the original dataset can be aggregated (summed) over one or more variables
    
    Input:
    ------
    hits:     xarray.Dataset. A boolean matrix of hits, misses and false alarms. It must contain three variables: 'tp' hits, 'fn' misses, 'fp' false alarms
    dims_agg: str or list. Dimensions in 'hits' over which the number of hits/misses/false alarms will be added.
    beta:     float or list of floats. A coefficient (or list of coefficients) of the f score that balances the importance of misses and false alarms. By default is 1, so misses and false alarms penalize the same. If beta is lower than 1, false alarms penalize more than misses, and the other way around if beta is larger than 1 

    Output:
    -------
    skill:    xarray.Dataset. A matrix with the metric values. Three metrics are computed and saved as variables: 'f1', 'precision', 'recall'
    """
    
    if dims_agg is None:
        aux = hits
    else:
        aux = hits.sum(dims_agg)
    if isinstance(beta, int):
        f_score = f'f{beta}'
    elif isinstance(beta, float):
        f_score = f'f{beta:.1f}'
    skill = xr.Dataset({f_score: (1 + beta**2) * aux['tp'] / ((1 + beta**2) * aux['tp'] + beta**2 * aux['fn'] + aux['fp']),
                        'precision': aux['tp'] / (aux['tp']+ aux['fp']),
                        'recall': aux['tp'] / (aux['tp']+ aux['fn'])})
    if 'leadtime' in skill.dims:
        skill['leadtime'] = (skill.leadtime / 3600e9).astype(int)
    
    return skill



def df2da(df, dims, plot=False, **kwargs):
    """It converts a pandas.DataFrame into a xarray.DataArray
    
    Inputs:
    -------
    df:    pandas.DataFrame
    dims:  list (2,). Names of the dimensions for the DataArray. The first dimension corresponds to the columns in the DataFrame, and the second dimension to the index
    plot:  boolean. Whether to plot or not a heat map of the data
    """
    
    da = xr.DataArray(df.transpose(), dims=dims, coords={dims[0]: df.columns.tolist(), dims[1]: df.index.tolist()})
    
    if plot:
        kwargs['xticklabels'] = df.index
        kwargs['yticklabels'] = df.columns
        plot_da(da, **kwargs)
        
    return da



def dict2da(dictionary, dim):
    """It converts a dictionary of xarray.Datarray into a single xarray.DataArray combining the keys in the dictionary in a new dimension
    
    Inputs:
    -------
    dictionary: dict. A dictionary of xarray.DataArray
    dim:        str. Name of the new dimension in which the keys of 'dictionary' will be combined
    
    Output:
    -------
    array:      xr.DataArray.
    """
    
    if isinstance(dictionary, dict) is False:
        return 'ERROR. The input data must be a Python dictionary.'
        
    data = list(dictionary.values())
    coord = xr.DataArray(list(dictionary), dims=dim)

    return xr.concat(data, dim=coord)



def reshape_DataArray(da, trim=False, chunks=None):
    """It converts a DataArray with 'forecast' and 'leadtime' dimensions into another DataArray with a 'datetime' and 'leadtime' dimensions.
    
    Inputs:
    -------
    da:       xarray.DataArray. Original DataArray
    trim:     boolean. Remove timesteps (at the beginning and end) for which all forecasts are not available
    chunks:   dict. A dictionary that specifies the size of Dask chunks in which the DataArray will be segmented
    """
    
    # compute frequencies of leadtime and forecast, and start and end datetime in the input DataArray
    freq_lt, freq_fc = [np.diff(da[dim]).mean() for dim in ['leadtime', 'forecast']]
    freq_lt, freq_fc = [(x / np.timedelta64(1, 'h')).astype(int) if isinstance(x, np.timedelta64) else x.astype(int) for x in [freq_lt, freq_fc]]
    ratio = int(freq_fc / freq_lt)
    st, en = [pd.to_datetime((da.forecast[i] + da.leadtime[i]).data) for i in [0, -1]]

    # define coordinates and create the empty new DataArray
    coords = {dim: da[dim].data for dim in da.dims if dim not in ['leadtime', 'forecast']}
    coords['leadtime'] = (np.arange(1, len(da.leadtime) / ratio + 1) * freq_fc).astype(int)
    coords['datetime'] = pd.date_range(st, en, freq=f'{freq_lt}h')
    da_new = xr.DataArray(coords=coords, dims=list(coords))

    # iteratively fill the new DataArray
    new_shape = [len(da[dim]) for dim in da.dims if dim not in ['leadtime', 'forecast']]
    new_shape += [len(da.forecast) * ratio] # the temporal resolution of the model is double as the frequency of forecasts
    for j, k in enumerate(np.arange(0, len(da.leadtime), ratio)):
        aux = da.isel(dict(leadtime=slice(k, k + 2))).data.reshape(new_shape)
        da_new[dict(leadtime=j, datetime=slice(k, k + new_shape[-1]))] = aux
    
    # if desired, remove timesteps for which at least one forcast is missing
    if trim:
        if 'id'in da_new.dims:
            mask = da_new.mean('id')
        else:
            mask = da_new
        if 'model' in mask.dims:
            mask = mask.mean('model')
        mask = mask.isnull().any('leadtime')
        da_new = da_new.loc[dict(datetime=~mask)]
        
    if chunks is not None:
        da_new = da_new.chunk(chunks)

    return da_new
        
        
  
def compute_events(da, probability=None, persistence=(1, 1), min_leadtime='all'):
    """It defines predicted events out of a DataArray of exceendances over a probability threshold. 
    The persistence criterion defines the number of forecast that musts predict an exceedance in order to be considered an event.
    
    Inputs:
    -------
    da:           xr.DataArray. A matrix of exceedances over probability threshold, or a matrix of probability of exceedance (in that case the attribute 'probability' is required). It must have a dimension called 'leadtime', over which the function will compute persistence
    probability:  float or xr.DataArray. Probability thresholds used to convert a 'da' of exceedance probability into a DataArray of exceedances over threshold. If None, the function implies that 'da' is already a DataArray of exceedances over threshold
    persistence:  tuple (a, b). Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
    min_leadtime: str of int. The minimum leadtime (in hours) above which the events will be notified. If 'all', the computation will be done for every leadtime in 'da.leadtime'
    
    Output:
    -------
    events:       xr.DataArray. A matrix of predicted events. The dimension 'leadtime' in the input DataArray (length 20 in the usual case) is collapsed to a single value.
    """
    
    # invert 'leadtime' order from longer to shorter lead times
    da = da.isel(leadtime=slice(None, None, -1))

    # compute exceedance over the probability threshold
    if probability is None:
        exceedance = da
    else:
        exceedance = (da >= probability).astype(int)
    #compute_events.exceedance = exceedance.isel(leadtime=slice(None, None, -1))

    # compute persistence (rolling sum over a window exceeds a number of forecast positives)
    events = (exceedance.rolling({'leadtime': persistence[1]}, center=False, min_periods=1).sum() >= persistence[0]) & exceedance
    events = events.isel(leadtime=slice(None, None, -1))

    if min_leadtime == 'all':
        events_agg = events.copy()
        for lt in events_agg.leadtime.data:
            events_agg.loc[dict(leadtime=lt)] = events.sel(leadtime=slice(lt, None)).any('leadtime')
        return events_agg.astype(int)
    elif min_leadtime in da.leadtime:
        # check if there's any predicted event
        return events.sel(leadtime=slice(min_leadtime, None)).any('leadtime').astype(int)
    else:
        return "ERROR. The attribute 'min_leadtime' must be 'all' or a value in 'da.leadtime'."
        
        
        
def compute_events2D(da, probability=None, persistence=(1, 1), resample=None, min_leadtime=None):
    """It defines predicted events out of a DataArray of exceendances over a probability threshold. 
    The persistence criterion defines the number of forecast that must predict an exceedance in order to be considered an event.
    
    Inputs:
    -------
    da:           xr.DataArray. A matrix of exceedances over probability threshold. It must have a dimension called 'leadtime', over which the function will compute persistence
    probability:  float or xr.DataArray. Probability thresholds over which a forecast is considered an event
    persistence:  tuple (a, b). Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
    resample:     string. Time resolution at which resample the leadtime data, e.g., '12h'.
    min_leadtime: int. Minimun number of leadtime hours required to raise the notification for an event
    
    Output:
    -------
    As objetcs:
    events:       xr.DataArray. A matrix of predicted events. The dimension 'leadtime' in the input DataArray (length 20 in the usual case) is collapsed to a single value.
    As method:
    exceedance:   xr.DataArray. Matrix of cells that exceed the 'probability' threshold
    """
    
    # invert 'leadtime' order from longer to shorter lead times
    da = da.isel(leadtime=slice(None, None, -1))

    # compute exceedance over the probability threshold
    if probability is None:
        exceedance = da
    else:
        exceedance = (da >= probability).astype(int)
    compute_events2D.exceedance = exceedance.isel(leadtime=slice(None, None, -1))
    
    # define custom function to apply to rolling window
    def compute_persistence(da):
        return da.fillna(0).any('dt').sum('lt')

    # apply custom function to 2D window of size (2,2)
    exc_rolling = exceedance.rolling(leadtime=persistence[1], datetime=3, center={'leadtime': False, 'datetime': True}, min_periods=1)
    exc_rolling = exc_rolling.construct(leadtime='lt', datetime='dt')
    events = xr.DataArray(np.zeros(exceedance.shape), dims=exceedance.dims, coords=exceedance.coords)
    for i in range(exc_rolling.shape[0]):
        for j in range(exc_rolling.shape[1]):
            loc = dict(leadtime=i, datetime=j)
            mask = (compute_persistence(exc_rolling[loc]) >= persistence[0]) & exceedance[loc].astype(bool)
            events[loc] = events[loc].where(~mask, other=1)
    events = events.isel(leadtime=slice(None, None, -1))
    
    if resample is not None:
        # convert 'leadtime' from integer hours to timedelta
        events['leadtime'] = pd.to_timedelta(events.leadtime, 'h')
        # resample
        events = events.resample({'leadtime': resample}).any().astype(int)
        # reconvert 'leadtime' back to integer hours
        events['leadtime'] = (events.leadtime / np.timedelta64(1, 'h')).astype(int)
        return events.sel(leadtime=slice(min_leadtime, None))
    else:
        # check if there's any predicted event
        return events.sel(leadtime=slice(min_leadtime, None)).any('leadtime').astype(int)
        
        
    
def buffer_events(da, center=True, w=5):
    """It creates a buffer around the matrix of predicted events to allow for short lags between observation and prediction. 
    It applys a rolling sum of window 'w' to the input matrix. The window function can be centered or not.
    
    Inputs:
    -------
    da:     xr.DataArray. Matrix of predicted events
    center: boolean. Whereas the rolling sum must be centered or right sided
    w:      int. Width of the rolling sum window
    
    Output:
    -------
    buffer: xr.DataArray. Matrix with the same size as the input matrix, but in which the events have been 'enlarged'
    """
    
    if center:
        mp = int(w / 2) + 1
        buffer = (da.rolling({'datetime': w}, center=True, min_periods=mp).sum() > 0).astype(int)
    else:
        mp = 1 # int(w / 2)
        buffer = (da.rolling({'datetime': w}, center=False, min_periods=mp).sum() > 0).astype(int)
        
    return buffer



def count_events(da):
    """Given a boolean DataArray of exceedances over probability threshold, it counts the number of events in the timeseries
    
    Input:
    ------
    da:       xr.DataArray. Boolean matrix of exceedances over probability threshold. It must contain a 'datetime' dimension
    
    Output:
    -------
    n_events: xr.DataArray. A matrix with the counts of events over the dimension 'datetime' (which collapses)"""
    
    # compute onsets: difference equal to 1
    onsets = (xr.concat([da.isel(datetime=[0]), da.diff('datetime')], dim='datetime') == 1).astype(int)
    # count events
    n_events= onsets.sum('datetime')#.data
    
    return n_events



def compute_hits(obs, pred, center=True, w=1):#, verbose=True):
    """It computes the hits, misses and false alarms between two matrixes of observations and predictions.
    To allow for some lags in the predictions, a buffer can be applied by giving the attribute 'w' a value larger than 1
    
    Inputs:
    -------
    obs:     xr.DataArray. Boolean matrix of observed exceedances over threshold
    pred:    xr.DataArray. Boolean matrix of predicted exceedances over threshold
    center:  boolean. Whereas the rolling sum must be centered or right sided
    w:       int. Width of the rolling sum window
    verbose: boolean. Whether to print or not the summary of results
    
    Output:
    -------
    As an object:
    hits:    xr.Dataset. Contains three variables: TP, true positives; FN, false negatives; FP, false positives
    As methods:
    buffer:  xr.DataArray. The buffered matrix of predicted events
    tp:      xr.DataArray. A timeseries of correctly predicted events
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
    compute_hits.buffer = buff

    # compute the true positive timeseries
    tp = buff.where(obs == 1) # apply observed mask on the buffered prediction
    tp = (tp == 1).astype(int) # ones in the masked array are true positives
    compute_hits.true_positives = tp
    
    # number of observed and predicted events
    n_obs, n_pred = [count_events(da) for da in [obs, buff]]
    
    # compute performance metrics
    TP = count_events(tp)
    TP = xr.ufuncs.minimum(TP, n_obs)
    FN = n_obs - TP
    FP = xr.ufuncs.maximum(0, n_pred - TP) #max(0, n_pred - TP)

    return xr.Dataset({'TP': TP, 'FN': FN, 'FP': FP})



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def size_objects(n=20):
    """
    """

    # get all the objects in the notebook namespace
    all_objects = globals().items()

    # get the size of each object and store it in a dictionary
    sizes = {}
    for key, value in all_objects:
        if key.startswith('_'):
            continue
        sizes[key] = sys.getsizeof(value)

    # sort the objects by their size in descending order
    sorted_objects = sorted(sizes.items(), key=operator.itemgetter(1), reverse=True)
    
    # print the objects and their sizes
    for i, (obj, size) in enumerate(sorted_objects):
        print(obj, ":", size)
        if i == n:
            break
            
            
            
def hits2skill(hits, beta=1):
    """It computes skill metrics (recall, precision and f1) out of a Dataset of hits, misses and false alarms.
    
    Input:
    ------
    hits:       xr.Dataset. It contains three DataArrays with names 'TP' (hits), 'FN' (misses) and 'FP' (false alarms)
    beta:       float or list of floats. A coefficient (or list of coefficients) of the f score that balances the importance of misses and false alarms. By default is 1, so misses and false alarms penalize the same. If beta is lower than 1, false alarms penalize more than misses, and the other way around if beta is larger than 1 
    
    Output:
    -------
    skill:      xr.Dataset. It contains three DataArrays with the metrics 'recall', 'precision', and 'fbeta' scores.
    
    
    """
    
    skill = xr.Dataset({'recall': hits.TP / (hits.TP + hits.FN),
                        'precision': hits.TP / (hits.TP + hits.FP)})
    if isinstance(beta, float) or isinstance(beta, int):
        beta = [beta]
    for b in beta:
        if isinstance(b, int):
            score = f'f{b}'
        else:
            score = f'f{b:.1f}'
        skill[score] = (1 +  b**2) * hits.TP / ((1 +  b**2) * hits.TP + b**2 * hits.FN + hits.FP)
    
    return skill



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


def find_best_criteria_cv(hits, station_events, dims=['probability', 'persistence'], kfold=5, train_size=.8, random_state=0, beta=1, tolerance=1e-2, min_spread=True):
    """A cross-validation version of the function of the function 'find_best_criteria'. It selects the criteria that maximizes the skill over a 'kfold' number of subsamples of the stations
    
    Inputs:
    -------
    hits:                 xarray.Dataset (id, persistence, approach, probability). A boolean matrix of hits, misses and false alarms. It must contain three variables: 'tp' hits, 'fn' misses, 'fp' false alarms
    station_events:       pd.Series. The number of observed events in the set of the stations used for the optimization. It will be used as a covariable in the stratified sampling in order to keep the proportion of events in each of the subsets
    dims:        list or string. Name(s) of the dimension(s) in 'skill' that will be optimized
    kfold:                int. Number of subsets of the stations to be produced
    train_size:           float. It should be between 0.0 and 1.0 and represents the proportion of the dataset to include in the train split
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
    cv = StratifiedShuffleSplit(n_splits=kfold, train_size=train_size, random_state=random_state)
    for i, (train, val) in enumerate(cv.split(station_events.index, station_events.values)):

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



def define_area_ranges(area_min, area_max, scale='semilog'):
    """Define an array of catchment area ranges
    
    Inputs:
    -------
    area_min: int. Minimum catchment area
    area_max: int. Maximum catchment area
    scale:    str. Type of scale: 'linear', 'log', 'semilog'
    
    Output:
    -------
    areas:    np.array.
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



def month2season(month):
    """It provides the season of a month
    
    Inputs:
    -------
    month:  int (1-12)
    
    Output:
    -------
    season:  str. Season of the year
    """
    
    month_to_season = {1: 'winter', 2: 'winter', 3: 'winter',
                       4: 'spring', 5: 'spring', 6: 'spring',
                       7: 'summer', 8: 'summer', 9: 'summer',
                       10: 'autumn', 11: 'autumn', 12: 'autumn'}
    
    return month_to_season[month]

# Use numpy.vectorize to vectorize the mapping function
month2season_vec = np.vectorize(month2season)



def disaggregate_by_season(da, dim='datetime'):
    """Given a DataArray with a datetime dimension, it creates a new dimension named 'season' to store the data corresponding to each of the 4 seasons
    
    Input:
    ------
    da:   xr.DataArray. One of its dimensions must be of type datetime
    dim:  string. Name of the dimension in 'da' of type datetime that will be used to split the 4 seasons
    
    Output:
    -------
    da_season: xr.DataArray. A new DataArray with one extra dimension: 'season'
    """
    
    if dim not in da.dims:
        return 'ERROR. The dimension "dim" is not in the DataArray'
    
    seasons = ['winter', 'spring', 'summer', 'autumn']
    array_seasons = xr.apply_ufunc(month2season_vec, da[dim].dt.month, vectorize=True)
    da_season = {season: da.where(array_seasons == season, drop=True) for season in seasons}
    da_season = xr.concat(da_season.values(), dim='season').assign_coords(season=seasons)
    
    return da_season



def recompute_exceedance(obs, pred_high, pred_low):
    """It recomputes exceedances given DataArrays of observed and predicted exceedance based on 2 discharge thresholds
    
    Inputs:
    -------
    obs:         xr.DataArray ('datetime', 'id'). Observed exceedance over the discharge thresholds. It has 3 possible values: 2, exceedance over the higher threshold; 1, exceedance over the lower threshold; 0, non-exeedance
    pred_high:   xr.DataArray ('datetime', 'id', 'model', 'leadtime'). Probability of exceedance (0-1) over the higher discharge threshold
    pred_low:    xr.DataArray ('datetime', 'id', 'model', 'leadtime'). Probability of exceedance (0-1) over the lower discharge threshold
    
    Outputs:
    --------
    exceed_obs:  xr.DataArray ('datetime', 'id'). The input observed exeedance recomputed with only 2 classes: 1, exceedance; 0, non-exceedance
    exceed_pred: xr.DataArray ('datetime', 'id', 'model', 'leadtime'). Probability of exceedance (0-1) as a combination of the exceedances for the higher and lower thresholds
    """
    
    # create empty xarray.DataArrays for observed and predicted exceedance
    exceed_obs = np.zeros_like(obs)
    exceed_pred = np.zeros_like(pred_high)

    # if observation exceeds Q5
    mask = (obs == 2).data
    exceed_obs[mask] = 1
    exceed_pred[mask] = pred_low.data[mask]

    # if observation exceeds 0.95*Q5 and some predictions exceed Q5
    mask = ((obs == 1) & (pred_high > 0)).data
    exceed_obs[mask.any(axis=(2, 3))] = 1
    exceed_pred[mask] = pred_low.data[mask]

    # if observation exceeds 0.95*Q5 and none of the predictions exceed Q5
    mask = ((obs == 1) & (pred_high == 0)).data
    exceed_obs[mask.all(axis=(2, 3))] = 0
    exceed_pred[mask] = pred_high.data[mask]

    # if observation does not exceed 0.95*Q5
    mask = (obs == 0).data
    exceed_obs[mask] = 0
    exceed_pred[mask] = pred_high.data[mask]
    
    # convert into xarray.DataArrays
    exceed_obs = xr.DataArray(exceed_obs, dims=obs.dims, coords=obs.coords)
    exceed_pred = xr.DataArray(exceed_pred, dims=pred_high.dims, coords=pred_high.coords)
    
    return exceed_obs, exceed_pred



def summarize_by_area(station_area, station_events, area_ranges):
    """It calculates the amount of stations and observed events at different catchment area thresholds
    
    Inputs:
    -------
    station_area:     pd.Series (id,). Catchment area of each of the stations contained in the dimension 'id' of the dataset 'hits'
    station_events:   pd.Series (id,). Number of observed events of each of the stations contained in the dimension 'id' of the dataset 'hits' 
    area_ranges:      np.array (area,). Values of catchment area in which the results will be discretized
    
    Output:
    -------
    summary:          pd.DataFrame(area,2). Summary of number of stations and observed events by catchment area
    """
    
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



def hits_by_area(hits, station_area, area_ranges):
    """Given a Dataset of hits by station ID and the area of the stations, it computes the hits grouped by catchment area threshold
    
    Inputs:
    -------
    hits:             xr.Dataset (id, persistence, approach, probability). It contains 3 variables: 'TP' true positives, 'FN' false negatives, 'FP' false positives
    station_area:     pd.Series (id,). Catchment area of each of the stations contained in the dimension 'id' of the dataset 'hits'
    area_ranges:      np.array (area,). Values of catchment area in which the results will be discretized
    
    Output:
    -------
    hits_area:        xr.Dataset (persistence, approach, probability, area). It contains the same 3 variables as the original dataset 'hits', but aggregated by ranges of catchment area
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