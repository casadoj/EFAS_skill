import numpy as np
import pandas as pd



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
    
    
    
def filter_correlation_matrix(correlation_matrix, rho=.9):
    """This function is used to filter a correlation matrix based on a certain threshold. It takes in 3 parameters:
    
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





