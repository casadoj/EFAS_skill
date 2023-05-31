import numpy as np
import pandas as pd
import xarray as xr



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