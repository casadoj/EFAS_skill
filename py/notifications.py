import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


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


def plot_events_timeseries(discharge, events1, events2=None, thresholds=None, save=None, **kwargs):
    """It creates a plot with the discharge time series and the identified flood events
    
    Inputs:
    -------
    discharge:  pandas.Series (timesteps,). Discharge timeseries
    events1:    pandas.Series (timestpes,). Boolean series that defines the beginning of a flood event
    events2:    pandas.Series (timestpes,). Boolean series that defines the beginning of a flood event. If None, it is not plotted
    thresholds: list. If provided, it must contain 4 values with the discharge at four increasing return periods, e.g., 1.5, 2, 5 and 20 years
    save:       string. If not None, it must be a string with the file name (including extension) where the plot will be saved
    
    Ouput:
    ------
    The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
    """
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (16, 3)))
    
    # plot discharge timeseries
    ax.plot(discharge, lw=.7, zorder=0)
    
    # plot points of preliminary events
    if events2 is not None:
        ax.scatter(discharge[events2].index, discharge[events2], s=kwargs.get('size', 2), color='k')
        ax.text(.005, .85, 'no. preliminary events: {0}'.format(events2.sum()), transform=ax.transAxes, fontsize=9)
        
    # plot points of the events
    ax.scatter(discharge[events1].index, discharge[events1], s=kwargs.get('size', 2), color='r')
    ax.text(.005, .925, 'no. events: {0}'.format(events1.sum()), transform=ax.transAxes, fontsize=9, color='r')
    
    # find minimum and maximum discharge
    qmax = discharge.max()
    magnitude = len(str(int(qmax)))
    ymin = - 10**(magnitude -2)
    ymax = np.ceil(qmax / 10**(magnitude - 1)) * 10**(magnitude - 1) + 10**(magnitude - 2)

    # return periods
    if thresholds is not None:
        ax.fill_between(discharge.index, *thresholds[0:2], color='green', edgecolor=None, alpha=.1, zorder=0, label='1.5-year')
        ax.fill_between(discharge.index, *thresholds[1:3], color='yellow', edgecolor=None, alpha=.1, zorder=0, label='2-year')
        ax.fill_between(discharge.index, *thresholds[2:4], color='red', edgecolor=None, alpha=.1, zorder=0, label='5-year')
        ax.fill_between(discharge.index, thresholds[-1], ymax, color='mediumpurple', edgecolor=None, alpha=.1, zorder=0, label='20-year')

    # settings: limits, labels, title, legend...
    ax.set(xlim=kwargs.get('xlim', (discharge.index[0], discharge.index[-1])),
           ylim=(ymin, ymax),
           ylabel='discharge (m³/s)');
    if 'title' in kwargs:
        fig.text(.5, .9, kwargs['title'], horizontalalignment='center')
    # fig.legend(loc=8, ncol=2, bbox_to_anchor=[0.9, .55, .2, .2]);
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        

        
def plot_events_map(x, y, n_events, save=None, rivers=None, ax=None, **kwargs):
    """It plots a map of Europe with the reporting points and their number of flood events
    
    Inputs:
    -------
    x:        pandas.Series (stations,). Coordinate X of the stations
    y:        pandas.Series (stations,). Coordinate Y of the stations
    n_events: pandas.Series (stations,). Number of flood events identified in each station
    save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
    Ouput:
    ------
    The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
    """
    
    # define projection
    if ax is None:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000,
                                              globe=ccrs.Globe(ellipse='GRS80'))
        ax = plt.axes(projection=proj)
    
    # plot coatslines and country borders
    ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
    ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
    # plot rivers
    if rivers is not None:
        rivers.to_crs(crs='epsg:3035').plot(lw=.5, color='gray', ax=ax, zorder=0)
    
    # plot all the stations
    ax.scatter(x, y, s=kwargs.get('size', 1) / 10, c='dimgray', alpha=kwargs.get('alpha', .5), label='stations w/o events')
    
    # plot stations with flood events
    stns = n_events[n_events > 0].index
    im = ax.scatter(x[stns], y[stns], s=kwargs.get('size', 1), c=n_events[stns], cmap='coolwarm', alpha=kwargs.get('alpha', .5), vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(n_events.max(), 2)))
    plot_events_map.colorbar = im
    
    # settings
    if ax is None:
        plt.colorbar(im, location='bottom', shrink=.4, label='no. events')
        plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
        # ax.set_extent([-13, 45, 30, 70])
        ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    else:
        plot_events_map.legend = ax.get_legend_handles_labels()
    ax.axis('off')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
    """It plots a map of Europe with the reporting points and their number of flood events
    
    Inputs:
    -------
    x:        pandas.Series (stations,). Coordinate X of the stations
    y:        pandas.Series (stations,). Coordinate Y of the stations
    z:        pandas.Series (stations,). Number of flood events identified in each station
    save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
    Ouput:
    ------
    The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
    """
    
    # define projection
    if ax is None:
        fig = plt.figure(figsize=kwargs.get('figsize', None))
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
        ax = plt.axes(projection=proj)
    
    # plot coatslines and country borders
    ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
    ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
    # plot rivers
    if rivers is not None:
        rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax, zorder=0)
    
    # plot all the stations
    if mask is not None:
        ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 10, c='dimgray', alpha=kwargs.get('alpha', .5), label='stations w/o events')
    
    # plot stations with flood events
    if mask is not None:
        x = x[~mask]
        y = y[~mask]
        z = z[~mask]
    sct = ax.scatter(x, y, c=z, s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
                    alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
    plot_map_stations.colorbar = sct
    
    # settings
    if ax is None:
        plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
        plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
        # ax.set_extent([-13, 45, 30, 70])
        ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    else:
        plot_map_stations.legend = ax.get_legend_handles_labels()
    ax.axis('off')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_map_f1(x, y, z, rivers=None, ax=None, cbar=False, save=None, **kwargs):
    """It plots a map of Europe with the reporting points and their number of flood events
    
    Inputs:
    -------
    x:        pandas.Series (stations,). Coordinate X of the stations
    y:        pandas.Series (stations,). Coordinate Y of the stations
    z:        pandas.Series (stations,). Number of flood events identified in each station
    save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
    Ouput:
    ------
    The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
    """
    
    # define projection
    if ax is None:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000,
                                              globe=ccrs.Globe(ellipse='GRS80'))
        ax = plt.axes(projection=proj)
    
    # plot coatslines and country borders
    ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
    ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
    # plot rivers
    if rivers is not None:
        rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax, zorder=0)
    
    # plot all the stations
    ax.scatter(x, y, s=kwargs.get('size', 1) / 10, c='dimgray', alpha=kwargs.get('alpha', .5), label='stations w/o events')
    
    # plot stations with flood events
    stns = z[~z.isnull()].index
    sct = ax.scatter(x[stns], y[stns], c=z[stns], s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
                    alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
    if cbar:
        plt.colorbar(sct, label='f1 (-)', shrink=.66)#*kwargs.get('cbar_kwgs', []));
    plot_events_map.colorbar = sct
    
    # settings
    if ax is None:
        plt.colorbar(im, location='bottom', shrink=.4, label='no. events')
        plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
        # ax.set_extent([-13, 45, 30, 70])
        ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    else:
        plot_events_map.legend = ax.get_legend_handles_labels()
    ax.axis('off')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
    
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
    
    

def compute_exceedance_2(model_files, thresholds, verbose=True):
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
    for model, files in model_files.items():
        dct = {}
        for file in files:
            
            if verbose:
                print(f'{model}\t{file}', end='\r')

            # open dataaray with dicharge data
            dis = xr.open_dataarray(file)
            # limit the forecast to 10 days
            if len(dis.time) > 40:
                dis = dis.isel(time=slice(0, 40))
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

    
    
def deterministic_criteria(da, leadtime=4):
    """Given a boolean DataArray with the set of forecast of a deterministic model, it computes if the the formal notification criteria for deterministic models is fulfilled
    
    * Lead time >= 48 h (skip fisrt 4 timesteps since the temporal resolution is 6 h)
    * Any timestep exceeds Q5
    
    Input:
    ------
    da:        DataArray (forecast, leadtime). Set of forecast containing boolean data of exceedance/non-exceedance of the 5 year return period of discharge
    leadtime:  int. Amount of timesteps to skip at the beginning of each forecast.
    
    Output:
    -------
    da:        DataArray (forecast,). Boolean DataArray that defines whether a forecast fulfills the notification criteria or not
    """
      
    return da.isel(leadtime=slice(leadtime, None)).any('leadtime')


def probabilistic_criteria(da, leadtime=4, probability=.3, persistence=3):
    """Given a boolean DataArray with the set of forecast of a probabilistic model, it computes if the the formal notification criteria for probabilistic models is fulfilled
    
    * Lead time >= 48 h (skip fisrt 4 timesteps since the temporal resolution is 6 h)
    * The 5 year return period is exceeded with a given probability during a number of consecutive forecasts.
    
    Input:
    ------
    da:          DataArray (forecast, member, leadtime). Set of forecast containing boolean data of exceedance/non-exceedance of the 5 year return period of discharge
    leadtime:    int. Amount of timesteps to skip at the beginning of each forecast
    probability: float. Probability threshold required to send a notification
    presistence: int. Number of consecutive forecast that must exceed the probability threshold so that a notification is issued
    
    Output:
    -------
    da:        DataArray (forecast,). Boolean DataArray that defines whether a forecast fulfills the notification criteria or not
    """
    
    aux = (da.isel(leadtime=slice(leadtime, None)).mean('member') > probability).any('leadtime')
    aux = aux.rolling(forecast=persistence).sum() == persistence
    
    return aux


def compute_notifications(exceedance, leadtime=4, probability=.3, persistence=3):
    """Given a dictionary with the forecasts of the 2 deterministic modeles (EUD, DWD) and the 2 probabilistic models (EUE, COS), it computes all the notification criteria and creates a boolean DataArray with the issuance of formal notifications.
    
    Input:
    ------
    exceedance:  dict. Contains a DataArray for each model with the exceedance/not exceedance of the 5 year return period discharge.
    leadtime:    int. Amount of timesteps to skip at the beginning of each forecast
    probability: float. Probability threshold required to send a notification
    presistence: int. Number of consecutive forecast that must exceed the probability threshold so that a notification is issued
        
    Output:
    -------
    da:        DataArray (forecast,). Boolean DataArray that defines whether a formal notification should be issued or not
    """
    
    # DETERMINISTIC FORECASTS
    
    # calculate forecast in each model that comply with the notification criteria
    EUD = deterministic_criteria(exceedance['EUD'], leadtime)
    DWD = deterministic_criteria(exceedance['DWD'], leadtime)
    # combine models
    deterministic = xr.concat((EUD, DWD), 'model').any('model')
    
    # PROBABILISTIC FORECASTS
    
    # calculate forecast in each model that comply with the notification criteria
    EUE = probabilistic_criteria(exceedance['EUE'], leadtime, probability, persistence)
    COS = probabilistic_criteria(exceedance['COS'], leadtime, probability, persistence)
    # combine models
    probabilistic = xr.concat((EUE, COS), 'model').any('model')
    
    # COMBINE FORECASTS
    notifications = xr.concat((deterministic, probabilistic), 'type').all('type')
    
    return notifications


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



def compare_discharge(discharge, stations, threshold=None, **kwargs):
    """It compares visually the discharge timeseries in several stations. Comparison is done two-fold: in terms of Spearman correlation and a lineplot.
    One graph is devoted to each of these two facets. The plot related to correltion differs if we compare only two stations (scatter plot) or more (heat map).
    
    Inputs:
    -------
    discharge: pd.DataFrame (timesteps, stations). The timeseries to be compared
    stations:  list. IDs of the stations to be compared.
    threshold: pd.Series (stations,). Discharge threshold above which a flood is considered. If None (default), it is not plotted
    
    Output:
    -------
    A plot with two graphs.
       * On the left, either a scatter plot (2 stations) or a heat map (more than 2 stations) that shows the Spearman correlation between discharge timeseries
       * On the right, the line plot of the discharge timeseries
    
    """

    alpha = kwargs.get('alpha', .25)
    
    df = discharge[stations]
    if threshold is not None:
        thr = threshold[stations]

    qmax = df.max().max()
    order = len(str(int(qmax))) - 2
    qmax = np.ceil(qmax / 10**order) * 10**order + 10**order
    qmin = - 10**order

    fig = plt.figure(figsize=(10.5, 3.5), tight_layout=True)
    gs = gridspec.GridSpec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    #ax1.plot([qmin, qmax], [qmin, qmax], '-k', lw=.7, zorder=0)
    if len(stations) == 2:
        ax1.scatter(df[stations[0]], df[stations[1]], color=f'gray', s=1, alpha=alpha)
        ax1.set(xlabel=stations[0], xlim=(qmin, qmax), ylabel=stations[1], ylim=(qmin, qmax))
        if threshold is not None:
            ax1.vlines(thr[stations[0]], qmin, qmax, ls='--', lw=.5, color='C0')
            ax1.hlines(thr[stations[1]], qmin, qmax, ls='--', lw=.5, color='C1')
        spearman = df.corr(method='spearman').iloc[0, 1]
        ax1.text(0.01, .99, f'spearman = {spearman:.2f}', transform=ax1.transAxes, fontsize=9, verticalalignment='top')
    elif len(stations) > 2:
        # compute correlation matrix
        corr = df.corr(method='spearman')
        # remove the upper diagonal
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if j > i:
                    corr.iloc[i,j] = np.nan
        sns.heatmap(corr, cmap='Greys', annot=True, cbar=False, square=True, ax=ax1)
        ax1.set(xlabel=None, ylabel=None, title='Spearman correlation')
    
    if len(stations) == 2:
        ax2 = fig.add_subplot(gs[0, 1:], sharey=ax1)
    else:
        ax2 = fig.add_subplot(gs[0, 1:])
    df.plot(lw=1, ax=ax2)
    if threshold is not None:
        colors = [f'C{i}' for i, stn in enumerate(stations)]
        ax2.hlines(thr, df.index[0], df.index[-1], ls='--', lw=.5, color=colors)
    ax2.set(xlabel=None);
    
    if 'title' in kwargs:
        fig.text(.5, .995, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontweight='bold')
        
        
        
def exceedances_timeline(discharge, stations, thresholds=['rl5'], yticks=False, ax=None, save=None, **kwargs):
    """It creates a timeline with the exceedances of the discharge thresholds
    
    Input:
    ------
    discharge:  pd.DataFrame (timesteps, stations). Raw timeseries
    stations:   pd.DataFrame (stations, x). Table of attributes of the stations
    thresholds: list. Name of columns in 'stations' that will be used as thresholds: 'rl1.5', 'rl2', 'rl5', 'rl20' ...
    yticks:     boolean. Whether to add the station IDs as labels or not
    
    Output:
    -------
    A plot that shows for each station the timesteps at which discharge exceeded the thresholds
    """

    discharge = discharge.loc[:, stations.index]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
    n_stn = stations.shape[0]
    ax.set(xlim=(discharge.index[0], discharge.index[-1]), ylim=(0, n_stn + 1), ylabel='stations')
    if yticks:
        ax.set_yticks(range(1, n_stn + 1), labels=stations.index)
    else:
        ax.set_yticks([])
    if 'grid' in kwargs:
        ax.grid(visible=kwargs['grid'], which='major', axis='x')
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    
    # plot background lines
    for i, stn in enumerate(stations.index):
        ax.hlines(i + 1, discharge.index[0], discharge.index[-1], lw=.3, ls=':', color='gray')
    
    # plot exceedances
    colors = {'rl1.5': 'lightgreen', 'rl2': 'yellow', 'rl5': 'red', 'rl20': 'mediumpurple', 'rl500': 'k'}
    for rl in thresholds:
        # compute exceedances of the threshold
        exceed = discharge >= stations[rl]
        for i, stn in enumerate(exceed.columns):
            if exceed[stn].any():
                ts = exceed.index[exceed[stn]]
                ax.scatter(ts, [i + 1] * len(ts), marker='_', lw=1.2, color=colors[rl])
                
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
                
                
                
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



def heatmap_forecast(da, events=None, leadtime_grid=4, forecast_grid=10, ax=None, **kwargs):
    """It plots a heatmap with the forecast exceedance probability of a discharge threshold. Given a list of observed events when the discharge threshold was exceeded, it shows the corresponding point on top of the heatmap
    
    Inputs:
    -------
    da:             xarray.DataArray. It contains probability values and it must have two dimensions ('forecast', 'leadtime'). If the forecast is deterministic, it would be a matrix of 0 and 1; if it is probabilistic, it would contain values between 0 and 1
    events:         list. A list of date and times when a discharge exceedance was actually observed
    leadtime_grid:  integer. It defines the frequency of gridlines in the X axis. By default is 4, i.e., 1 day
    forecast_grid:  integer. It defindes the frequency of gridlines in the Y axis. By default is 10, i.e., 5 days
    ax:             matplotlib.axes. If provided, the plot will be added to that axis
    
    Output:
    -------
    Heatmap with points scattered on top of it (if events provided)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (16, 4)))
        
    # extract specific forecasts if defined in kwargs
    if 'ylim' in kwargs:
        da = da.sel(forecast=slice(*kwargs['ylim']))
        
    # plot heatmap of the data array
    sns.heatmap(da, mask=da == 0, cmap=kwargs.get('cmap', 'Blues'),
                vmin=kwargs.get('vmin', 0), vmax=kwargs.get('vmax', 1), alpha=kwargs.get('alpha', 1),
                cbar=kwargs.get('cbar', True), cbar_kws={'shrink': .66, 'label': kwargs.get('label', '')},
                ax=ax);
    
    last_leadtime = da.isnull().all('forecast').data.argmax()
    if last_leadtime > 0:
        ax.fill_between(np.arange(last_leadtime, len(da.leadtime) + 1), 0, len(da.forecast), color='lightgray', alpha=.33)
    else:
        last_leadtime = len(da.leadtime)
    
    # scatter plot of events
    if events is not None:
        ll, ff = np.meshgrid(da.leadtime.data[:last_leadtime], da.forecast.data)
        for event in events:
            ys, xs = np.where(ff + ll == event)
            ax.scatter(xs + .5, ys + .5, c='red', marker='.', s=15, lw=1, zorder=5)
            
    # set X axis
    xlabels = (da.leadtime / 86400e9).astype(int).data
    ax.vlines(range(0, last_leadtime + 1, leadtime_grid), 0, len(da.forecast), color='dimgray', lw=.5, ls=':')
    ax.set_xlabel('leadtime (d)')
    ax.set_xticks(range(0, len(da.leadtime), leadtime_grid))
    ax.set_xticklabels(xlabels[::leadtime_grid], rotation=0)
        
    # set Y axis
    ylabels = [pd.to_datetime(str(f)).strftime('%Y-%m-%d %H') for f in da.forecast.data]
    ax.hlines(range(0, len(da.forecast) + 1, forecast_grid), 0, last_leadtime, color='dimgray', lw=.5, ls=':')
    ax.set(ylabel='forecast', yticks=range(0, len(da.forecast) + 1, forecast_grid), yticklabels=ylabels[::forecast_grid])
    
    # set title
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    ax.tick_params(length=0)
    
    
    
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



def plot_skill(skill, n_events=None, xdim='id', ydim='probability', rowdim='model', save=None, **kwargs):
    """It creates a plot composed of several heatmaps with the skill assessment regarding several metrics (columns).
    
    Inputs:
    -------
    skill:    xarray.Dataset (xdim, ydim, rowdim). A data set with as many variables as skill metrics and 3 dimensions: 'id' is the code of the stations, 'model' are types of combining different model outputs, 'probability' are exceedance probability thresholds
    n_events: pandas.Series (xdim,). A series with the observed number of flood events in each station.
    xdim:     str. Name of the dimension that will correspond to the X axis in the plots
    ydim:     str. Name of the dimension that will correspond to the Y axis in the plots
    rowdim:   str. Name of the dimension that will be split across the graph rows
    """
    
    fig = plt.figure(constrained_layout=True, figsize=kwargs.get('figsize', (24, 9)))
    ncols = len(skill)
    widths = [1] * ncols
    if n_events is None:
        nrows = len(skill[rowdim]) + 1
        heights = [len(skill[ydim])] * len(skill[rowdim]) + [len(skill[rowdim])]
    else:
        nrows = len(skill[rowdim]) + 2
        heights = [len(skill[ydim])] * len(skill[rowdim]) + [len(skill[rowdim])] + [1]
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, width_ratios=widths, height_ratios=heights)

    for j, (metric, da) in enumerate(skill.items()):

        # probability with highest mean score
        best_model = da.mean(xdim).idxmax(ydim)
        best_model_i = da.mean(xdim).argmax(ydim)

        # score values for each station with the previous probability
        best_metric_model = xr.DataArray(dims=[rowdim, xdim], coords={rowdim: da[rowdim], xdim: da[xdim]})
        for row in best_model[rowdim]:
            p = best_model.sel({rowdim: row})
            best_metric_model.loc[{rowdim: row}] = da.sel({rowdim: row, ydim: p})

        for i, row in enumerate(da[rowdim].data):
            ax = fig.add_subplot(gs[i, j])

            if (i == 1) & (j == ncols -1):
                cbar = True
            else:
                cbar = False
            sns.heatmap(da.sel({rowdim: row}).transpose(), ax=ax, cmap=kwargs.get('cmap', 'Blues'), vmin=0, vmax=1,
                        cbar=cbar, cbar_kws={'label': '(-)', 'shrink': .9})
            ax.add_patch(plt.Rectangle((0, best_model_i.sel({rowdim: row})), len(skill[xdim]), 1,
                                       fc="none", edgecolor='red'))
            txt = '{0:.3f}'.format(best_metric_model.sel({rowdim: row}).mean(xdim).data)
            ax.text(len(skill[xdim]) + .5, best_model_i.sel({rowdim: row}) + .5, txt, 
                    verticalalignment='center', color='red')    
            ax.set_xticks([])
            yticks = np.arange(len(da[ydim].data)) + .5
            step = kwargs.get('ytick_step', 2)
            ax.set_yticks(yticks[1::step])  
            ax.set_yticklabels(da[ydim].data[1::step], rotation=0)
            if j == 0:
                ax.set_ylabel(f'{ydim} (-)')
                # ax.text(-8, len(da[ydim]) / 2, row, fontsize=13, rotation=90, verticalalignment='center')
                ax.text(-.15, .5, row, fontsize=13, rotation=90, verticalalignment='center', transform=ax.transAxes)
            # else:
            #     ax.set_yticks([])
            if i == 0:
                ax.set_title(metric)
            ax.tick_params(length=0)

            ax = fig.add_subplot(gs[3, j])
            sns.heatmap(best_metric_model, cmap=kwargs.get('cmap', 'Greys'), vmin=0, vmax=1, cbar=False, ax=ax)
            for i, row in enumerate(best_model[rowdim]):
                txt = '{0}'.format(best_model.sel({rowdim: row}).data)
                ax.text(len(da[xdim]) + .5, i + .5, txt,
                        verticalalignment='center', color='red')
            ax.set_xticks([])
            if j == 0:
                ax.set_yticks(np.arange(len(da[rowdim])) + .5)
                ax.set_yticklabels(da[rowdim].data, rotation=0)
            else:
                ax.set_yticks([])
            if n_events is None:
                xticks = np.arange(len(skill[xdim])) + .5
                step = kwargs.get('xtick_step', 2)
                ax.set_xticks(xticks[1::step])
                ax.set_xticklabels(skill[xdim].data[1::step], rotation=90)
                ax.set_xlabel(kwargs.get('xlabel', xdim))
            ax.tick_params(length=0);
            
            if n_events is not None:
                ax = fig.add_subplot(gs[4, j])
                if j == ncols - 1:
                    cbar = True
                else:
                    cbar = False
                sns.heatmap(n_events[np.newaxis,:], mask=n_events[np.newaxis,:] == 0, cmap='Greys',
                            vmin=0, cbar=cbar, cbar_kws={'label': '(-)'}, ax=ax)
                xticks = np.arange(n_events.shape[-1]) + .5
                step = kwargs.get('xtick_step', 2)
                ax.set_xticks(xticks[1::step])
                ax.set_xticklabels(n_events.index[1::step], rotation=90)
                ax.set_xlabel(kwargs.get('xlabel', xdim))
                ax.set_title('no. event')
                ax.set_yticks([])
                ax.tick_params(length=0)
                
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
            
            
            
def graphic_explanation(obs, pred, id, model, window, probability, forecast=False, verbose=True, **kwargs):
    """It creates a graph with six plots that explain how hits, misses and false alarms are computed.
    
    Inputs:
    -------
    obs:         xarray.DataArray (id, forecast, leadtime). Boolean matrix of observed events
    pred:        xarray.DataArray (id, model, forecast, leadtime, probability). Boolean matrix of predicted events
    id:          int. Station ID
    model:       string. Criteria: 'current', 'model_mean', 'weighted_mean'
    window:      int. Moving window centered on the observed event to take into account predictions in the vicinity of the actual event
    probability: float. Threshold of total probability (0-1)
    forecast:    tuple (star, end). Datetimes with the start and end timesteps to be plotted
    verbose:     boolean
    """

    if forecast is False:
        sel0 = {'id': id}
        sel1 = {'id': id, 'model': model, 'probability': probability}
    else:
        sel0 = {'id': id, 'forecast': forecast}
        sel1 = {'id': id, 'model': model, 'forecast': forecast, 'probability': probability}
    sel2 = {'id': id, 'model': model, 'probability': probability}
    
    if verbose:
        print(f'ID: {id}\nmodel: {model}\nprobability: {probability:.2f}\nwindow: {window}\n')

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=kwargs.get('figsize', (15, 6)))

    for axes in ax.flatten():
        axes.tick_params(length=0)
    
    sns.heatmap(obs.sel(sel0), cmap='Greys', cbar=False, ax=ax[0,0])
    ax[0,0].set(title='observed events', ylabel='forecast')
    
    mp = int(window / 2) + 1
    obs_w = obs.rolling({'leadtime': window}, center=True, min_periods=mp).sum() > 0
    sns.heatmap(obs_w.sel(sel0), cmap='Greys_r', cbar=False, ax=ax[1,0])
    ax[1,0].set(title='buffered observed events', xlabel='leadtime', yticks=[], ylabel='forecast')
       
    pred_w = pred.rolling({'leadtime': window}, center=True, min_periods=mp).sum() > 0
    sns.heatmap(pred.sel(sel1), cmap='Blues', cbar=False, ax=ax[1,1])
    ax[1,1].set(title='predicted events', xlabel='leadtime')

    sns.heatmap(pred_w.sel(sel1), cmap='Blues', cbar=False, ax=ax[0,1])
    ax[0,1].set(title='buffered predicted events')

    aux1 = pred_w.where(obs == 1)
    sns.heatmap(aux1.sel(sel1), cmap='Blues', cbar=False, ax=ax[0,2])
    ax[0,2].set(title='true positives + false negatives')
    tp = (aux1 == 1).sel(sel2).sum().data
    ax[0,2].text(0, .95, f'TP = {tp}', color='navy', transform=ax[0,2].transAxes)
    fn = (aux1 == 0).sel(sel2).sum().data
    ax[0,2].text(0, .85, f'FN = {fn}', color='navy', transform=ax[0,2].transAxes)

    aux2 = pred.where(obs_w == 0)
    sns.heatmap(aux2.sel(sel1), cmap='Blues', cbar=False, ax=ax[1,2])
    ax[1,2].set(title='false positives', xlabel='leadtime', xticks=[], yticks=[])
    fp = (aux2 == 1).sel(sel2).sum().data
    ax[1,2].text(0, .95, f'FP = {fp}', color='navy', transform=ax[1,2].transAxes)

    if verbose:
        print('f1 = {0:.3f}'.format(2 * tp / (2 * tp + fn + fp)))
        print('precision = {0:.3f}'.format(tp / (tp + fp)))
        print('recall = {0:.3f}'.format(tp / (tp + fn)))
        
        
        
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



def plot_da(da, ax=None, **kwargs):
    """It creates a heatmap plot of a 2D DataArray
    
    Input:
    ------
    da:   xarray.DataArray (n,m)
    ax:   matplotlib.axes
    """
    
    # extract kwargs
    figsize = kwargs.get('figsize', (16, 2))
    xtick_step = kwargs.get('xtick_step', 60)
    ytick_step = kwargs.get('ytick_step', 3)
    cmap = kwargs.get('cmap', 'magma')
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    cbar = kwargs.get('cbar', True)
    cbar_kws = kwargs.get('cbar_kws', None)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if len(da.shape) == 2:
        da_plot = da
        dimy, dimx = da.dims
    elif len(da.shape) == 1:
        da_plot = np.array(da)[np.newaxis,:]
        dimx = da.dims[0]
    sns.heatmap(da_plot, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, cbar=cbar, cbar_kws=cbar_kws)
    
    # if 'yticklabels' in kwargs:
    if 'dimy' in locals():
        yticklabels = kwargs.get('yticklabels', da[dimy].data)
        yticks = np.arange(0, len(yticklabels), ytick_step) + .5
        ax.set_yticks(yticks)
        if yticklabels.dtype == '<M8[ns]':
            yticklabels = [datetime.strftime(idx, '%Y-%m-%d') for idx in pd.to_datetime(yticklabels[::ytick_step])]
        else:
            yticklabels = yticklabels[::ytick_step]
        ax.set_yticklabels(yticklabels, rotation=0)
    
    # if 'xticklabels' in kwargs:
    if 'dimx' in locals():
        xticklabels = kwargs.get('xticklabels', da[dimx].data)
        xticks = np.arange(0, len(xticklabels), xtick_step) + .5
        ax.set_xticks(xticks)
        if xticklabels.dtype == '<M8[ns]':
            xticklabels = [datetime.strftime(idx, '%Y-%m-%d') for idx in pd.to_datetime(xticklabels[::xtick_step])]
        else:
            xticklabels = xticklabels[::xtick_step]
        ax.set_xticklabels(xticklabels, rotation=90)
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    
    ax.tick_params(length=0);

    
    
def reshape_da(da, coords, loop_dim='leadtime'):
    """It converts a DataArray with 'forecast' and 'leadtime' dimensions into another DataArray with a 'datetime' and 'leadtime' dimensions.
    
    Inputs:
    -------
    da:       xarray.DataArray. Original DataArray
    coords:   dict. Coordinates of the new DataArray to be created
    loop_dim: string. Dimension in the original DataArray over which the reshaping will be done.
    """
    
    # loop_dim = list(set(da.dims).difference(list(coords)))[0]
    
    da_new = xr.DataArray(coords=coords, dims=list(coords))
    
    new_shape = list(da.shape[:-1])
    new_shape[-1] *= 2 # the temporal resolution of the model is double as the frequency of forecasts

    for j, k in enumerate(np.arange(0, len(da[loop_dim]), 2)):
        aux = da.isel({loop_dim: slice(k, k + 2)}).data.reshape(new_shape)
        da_new[:, :, j, k:k + aux.shape[2]] = aux
        
    return da_new