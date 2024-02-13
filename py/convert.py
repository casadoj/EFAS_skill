import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Dict, Literal



def dict2da(dictionary: Dict, dim: str) -> xr.DataArray:
    """It converts a dictionary of xarray.Datarray into a single xarray.DataArray combining the keys in the dictionary in a new dimension
    
    Inputs:
    -------
    dictionary: Dict
        A dictionary of xarray.DataArray
    dim: str
        Name of the new dimension in which the keys of 'dictionary' will be combined
    
    Output:
    -------
    array: xr.DataArray.
    """
    
    assert isinstance(dictionary, dict), 'ERROR. The input data must be a Python dictionary.'
        
    data = list(dictionary.values())
    coord = xr.DataArray(list(dictionary), dims=dim)

    return xr.concat(data, dim=coord)



def df2da(df: pd.DataFrame, dims: List, plot: bool = False, **kwargs) -> xr.DataArray:
    """It converts a pandas.DataFrame into a xarray.DataArray
    
    Inputs:
    -------
    df: pd.DataFrame
    dims: List
        Names of the dimensions for the DataArray. The first dimension corresponds to the columns in the DataFrame, and the second dimension to the index
    plot: bool
        Whether to plot or not a heat map of the data
    
    Output:
    -------
    da: xr.DataArray
    """
    
    da = xr.DataArray(df.transpose(), dims=dims, coords={dims[0]: df.columns.tolist(), dims[1]: df.index.tolist()})
    
    if plot:
        kwargs['xticklabels'] = df.index
        kwargs['yticklabels'] = df.columns
        plot_da(da, **kwargs)
        
    return da



def reshape_DataArray(da: xr.DataArray, trim: bool = False, chunks: Dict = None) -> xr.DataArray:
    """It converts a DataArray with 'forecast' and 'leadtime' dimensions into another DataArray with a 'datetime' and 'leadtime' dimensions.
    
    Inputs:
    -------
    da: xr.DataArray
        Original DataArray
    trim: bool
        Remove timesteps (at the beginning and end) for which all forecasts are not available
    chunks: Dict
        A dictionary that specifies the size of Dask chunks in which the DataArray will be segmented
        
    Output:
    -------
    da_new: xr.DataArray
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