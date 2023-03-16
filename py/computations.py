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

        
    
def compute_exceedance(files, station, threshold):
    """From a list of files (NetCDF) corresponding to consecutive forecast, it extracts the data corresponding to a station and it creates a boolean matrix of exceedance (1) non-exceedance (0).
    
    Inputs:
    -------
    files:      list. NetCDF files corresponding with EFAS' discharge forecast
    station:    str. Station ID
    threshold:  float. Discharge threshold
    
    Output:
    -------
    A xarray.DataArray with a boolean matrix of exceedance/non-exceedance of the threshold. If the forecast is deterministic, the output DataArray will have two dimensions (number of forecast, number of timesteps in each forecast). If the forecast is probabilistic, the output DataArray will have three dimensions (no. forecast, no. memebers, no.timesteps)
    """
    
    # load an example file to extract dimensions
    aux = xr.open_dataarray(files[0])
    n_time = len(aux.time)
    if 'member' in aux.dims:
        member = aux.member.data
        arr = np.empty((len(files), len(member), n_time))
    else:
        arr = np.empty((len(files), n_time))
        if 'member' in locals():
            del n_member
    forecast = []
    leadtime = [timedelta(hours=(i + 1) * 6) for i in range(n_time)]
    aux.close()

    # read each file and compute exceedance
    for i, file in enumerate(files):
        # compute exceedance of the threshold
        da = xr.open_dataarray(file)
        exc = da.sel(stations=station) >= threshold
        if len(exc.shape) == 2:
            arr[i,:,:] = exc
        elif len(exc.shape) == 1:
            arr[i,:] = exc
        da.close()

        # add forecast to the list
        forecast.append(datetime.strptime(file[-13:-3], '%Y%m%d%H'))
        
    # build the xarray.DataArray
    if len(arr.shape) == 3:
        return xr.DataArray(arr, dims=('forecast', 'member', 'leadtime'), coords={'forecast': forecast, 'member': member, 'leadtime': leadtime})
    elif len(arr.shape) == 2:
        return xr.DataArray(arr, dims=('forecast', 'leadtime'), coords={'forecast': forecast, 'leadtime': leadtime})
    
    

# def compute_exceedance_2(model_files, thresholds, verbose=True):
#     """From a list of files (NetCDF) corresponding to consecutive forecast, it extracts the data corresponding to a station and it creates a boolean matrix of exceedance (1) non-exceedance (0).
    
#     Inputs:
#     -------
#     model_files: dictionary. The keys are the different NWP (numerical weather predicitons) models and the values a list of file names
#     threshold:   xarray.DataArray (id,). Matrix of 1 dimension (station ID) containint the discharge threshold
    
#     Output:
#     -------
#     A xarray.DataArray with a boolean matrix of exceedance/non-exceedance of the threshold. This DataArray has 4 dimensions: id, model, forecast, leadtime
#     """
    
#     exceedance = {}
#     for model, files in model_files.items():
#         dct = {}
#         for file in files:
            
#             if verbose:
#                 print(f'{model}\t{file}', end='\r')

#             # open dataaray with dicharge data
#             dis = xr.open_dataarray(file)
#             # limit the forecast to 10 days
#             if len(dis.time) > models[model]['leadtimes']:
#                 dis = dis.isel(time=slice(None, models[model]['leadtimes']))
#             # reformat the 'time' dimension into 'leadtime'
#             dis = dis.rename({'time': 'leadtime', 'stations': 'id'})
#             dis['leadtime'] = [timedelta(hours=(i + 1) * 6) for i in range(len(dis.leadtime))]

#             # compute mean exceedance over the threshod
#             # this steps also select the stations
#             exc = dis > thresholds
#             if 'member' in exc.dims:
#                 exc = exc.mean('member')

#             # save in the dictionary
#             forecast = datetime.strptime(file[-13:-3], '%Y%m%d%H')
#             dct[forecast] = exc

#         # join the exceedance of all the files of a model into one dataset
#         exceedance[model] = xr.Dataset(dct).to_array(dim='forecast', name=model)
        
#         print()
        
#     # join all the models into one dataset
#     exceedance = xr.Dataset(exceedance).to_array(dim='model', name='exceedance')
    
#     return exceedance

    
    
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
            
            
            

        
        
        
def compute_metrics(hits, dims_agg=None):
    """It computes a Dataset of metrics (f1, precision and recall) out of a Dataset of hits (variable 'tp'), misses (variable 'fn') and false alarms (variable 'fp'). If desired, the original dataset can be aggregated (summed) over one or more variables
    
    Input:
    ------
    hits:     xarray.Dataset. A boolean matrix of hits, misses and false alarms. It must contain three variables: 'tp' hits, 'fn' misses, 'fp' false alarms
    dims_agg: str or list. Dimensions in 'hits' over which the number of hits/misses/false alarms will be added.
    
    Output:
    -------
    skill:    xarray.Dataset. A matrix with the metric values. Three metrics are computed and saved as variables: 'f1', 'precision', 'recall'
    """
    
    if dims_agg is None:
        aux = hits
    else:
        aux = hits.sum(dims_agg)
    skill = xr.Dataset({'f1': 2 * aux['tp'] / (2 * aux['tp']+ aux['fp'] + aux['fn']),
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
        
        

# def compute_events(da, probability=None, persistence=(1, 1), min_leadtime=None):
#     """It defines predicted events out of a DataArray of exceendances over a probability threshold. 
#     The persistence criterion defines the number of forecast that must predict an exceedance in order to be considered an event.
    
#     Inputs:
#     -------
#     da:           xr.DataArray. A matrix of exceedances over probability threshold. It must have a dimension called 'leadtime', over which the function will compute persistence
#     persistence:  tuple (a, b). Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
#     min_leadtime: int. Minimum number of hours in advance necessary to raise an event notification
    
#     Output:
#     -------
#     As objetcs:
#     events:       xr.DataArray. A matrix of predicted events. The dimension 'leadtime' in the input DataArray (length 20 in the usual case) is collapsed to a single value.
#     As method:
#     exceedance:   xr.DataArray. Matrix of cells that exceed the 'probability' threshold
#     """
    
#     # compute exceedance over the probability threshold
#     if probability is None:
#         exceedance = da
#     else:
#         exceedance = (da >= probability).astype(int)
#     compute_events.exceedance = exceedance
    
#     # compute persistence (rolling sum over a window exceeds a number of forecast positives)
#     events = exceedance.rolling({'leadtime': persistence[1]}).sum() >= persistence[0]
    
#     # check if there's any predicted event
#     if min_leadtime is None:
#         return events.any('leadtime').astype(int)
#     else:
#         return events.sel(leadtime=slice(min_leadtime, None)).any('leadtime').astype(int)
    
    
    
def compute_events(da, probability=None, persistence=(1, 1), resample=None, min_leadtime=None):
    """It defines predicted events out of a DataArray of exceendances over a probability threshold. 
    The persistence criterion defines the number of forecast that must predict an exceedance in order to be considered an event.
    
    Inputs:
    -------
    da:           xr.DataArray. A matrix of exceedances over probability threshold. It must have a dimension called 'leadtime', over which the function will compute persistence
    persistence:  tuple (a, b). Two values that define the number of positive forecasts (a) out of a series of consecutive forecast (b) needed to consider the prediction as an event
    resample:     string. 
    
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
    compute_events.exceedance = exceedance

    # compute persistence (rolling sum over a window exceeds a number of forecast positives)
    events = exceedance.rolling({'leadtime': persistence[1]}, min_periods=1).sum() >= persistence[0]
    events = events.isel(leadtime=slice(None, None, -1))

    if resample is not None:
        # convert 'leadtime' from integer hours to timedelta
        events['leadtime'] = pd.to_timedelta(events.leadtime, 'h')
        # resample
        events = events.resample({'leadtime': resample}).any().astype(int)
        # reconvert 'leadtime' back to intege hours
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

    # number of observed and predicted events
    n_obs, n_pred = [count_events(da) for da in [obs, pred]]

    # buffer the predicted events
    buff = buffer_events(pred, center=center, w=w)
    compute_hits.buffer = buff

    # compute the true positive timeseries
    tp = buff.where(obs == 1) # apply observed mask on the buffered prediction
    tp = (tp == 1).astype(int) # ones in the masked array are true positives
    compute_hits.true_positives = tp

    # compute performance metrics
    TP = count_events(tp)
    TP = xr.ufuncs.minimum(TP, n_obs)
    FN = n_obs - TP
    FP = xr.ufuncs.maximum(0, n_pred - TP) #max(0, n_pred - TP)
    # if verbose:
    #     print(f'TP:\t{TP}\nFN:\t{FN}\nFP:\t{FP}')
    #     print('recall:\t\t{0:.3f}\nprecision:\t{1:.3f}\nf1:\t\t{2:.3f}'.format(TP / (TP + FN),
    #                                                                             TP / (TP + FP),
    #                                                                             2 * TP / (2 * TP + FP + FN)))

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
            
            
            
