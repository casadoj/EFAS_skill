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
from computations import *



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
        

        
# def plot_events_map(x, y, n_events, save=None, rivers=None, ax=None, **kwargs):
#     """It plots a map of Europe with the reporting points and their number of flood events
    
#     Inputs:
#     -------
#     x:        pandas.Series (stations,). Coordinate X of the stations
#     y:        pandas.Series (stations,). Coordinate Y of the stations
#     n_events: pandas.Series (stations,). Number of flood events identified in each station
#     save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
#     Ouput:
#     ------
#     The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
#     """
    
#     # define projection
#     if ax is None:
#         proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000,
#                                               globe=ccrs.Globe(ellipse='GRS80'))
#         ax = plt.axes(projection=proj)
    
#     # plot coatslines and country borders
#     ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
#     ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
#     # plot rivers
#     if rivers is not None:
#         rivers.to_crs(crs='epsg:3035').plot(lw=.5, color='gray', ax=ax, zorder=0)
    
#     # plot all the stations
#     ax.scatter(x, y, s=kwargs.get('size', 1) / 10, c='dimgray', alpha=kwargs.get('alpha', .5), label='stations w/o events')
    
#     # plot stations with flood events
#     stns = n_events[n_events > 0].index
#     im = ax.scatter(x[stns], y[stns], s=kwargs.get('size', 1), c=n_events[stns], cmap='coolwarm', alpha=kwargs.get('alpha', .5), vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(n_events.max(), 2)))
#     plot_events_map.colorbar = im
    
#     # settings
#     if ax is None:
#         plt.colorbar(im, location='bottom', shrink=.4, label='no. events')
#         plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
#         # ax.set_extent([-13, 45, 30, 70])
#         ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
#     else:
#         plot_events_map.legend = ax.get_legend_handles_labels()
#     ax.axis('off')
    
#     if 'title' in kwargs:
#         ax.set_title(kwargs['title'])
        
#     if save is not None:
#         plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
    """It plots a map of Europe with the reporting points and their number of flood events
    
    Inputs:
    -------
    x:        pandas.Series (stations,). Coordinate X of the stations
    y:        pandas.Series (stations,). Coordinate Y of the stations
    z:        pandas.Series (stations,). Number of flood events identified in each station
    mask:     pandas.Series (stations,). A boolean series of stations to be plotted differently, i.e., not included in the colour scale based on 'z'
    rivers:   geopandas. Shapefile of rivers
    ax:       matplotlib.axis. Axis in which the plot will be embedded. If None, a new figure will be created
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
        # plot masked stations
        ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 4, c='dimgray', alpha=kwargs.get('alpha', .5),
                   label='stations w/o events', zorder=0)
        x = x[~mask]
        y = y[~mask]
        z = z[~mask]

    # plot non-masked stations
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
    
    # reorganize Dataset according to the ordering of dimensions
    try:
        skill = skill.transpose(ydim, xdim, rowdim)
    except:
        print("ERROR. The plot dimensions ({0}) don't fit the Dataset dimensions".format(*[xdim, ydim, rowdim],
                                                                                         *list(skill.dims)))
        return
    
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
            sns.heatmap(da.sel({rowdim: row}), ax=ax, cmap=kwargs.get('cmap', 'Blues'), vmin=0, vmax=1,
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
                ax.text(-.15, .5, row, fontsize=13, rotation=90, verticalalignment='center', transform=ax.transAxes)
            if i == 0:
                ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.tick_params(length=0)

            ax_summary = fig.add_subplot(gs[len(skill[rowdim]), j])
            sns.heatmap(best_metric_model, cmap=kwargs.get('cmap', 'Blues'), vmin=0, vmax=1, cbar=False, ax=ax_summary)
            for i, row in enumerate(best_model[rowdim]):
                txt = '{0}'.format(best_model.sel({rowdim: row}).data)
                ax_summary.text(len(da[xdim]) + .5, i + .5, txt,
                                verticalalignment='center', color='red')
            ax_summary.set_xticks([])
            if j == 0:
                ax_summary.set_yticks(np.arange(len(da[rowdim])) + .5)
                ax_summary.set_yticklabels(da[rowdim].data, rotation=0)
            else:
                ax_summary.set_yticks([])
            if n_events is None:
                xticks = np.arange(len(skill[xdim])) + .5
                step = kwargs.get('xtick_step', 2)
                ax_summary.set_xticks(xticks[1::step])
                ax_summary.set_xticklabels(skill[xdim].data[1::step], rotation=90)
                ax_summary.set_xlabel(kwargs.get('xlabel', xdim))
            ax_summary.tick_params(length=0);
            
            if n_events is not None:
                if isinstance(n_events, xr.DataArray):
                    n_events = n_eventso.to_pandas()
                ax_events = fig.add_subplot(gs[len(skill[rowdim]) + 1, j])
                if j == ncols - 1:
                    cbar = True
                else:
                    cbar = False
                sns.heatmap(n_events[np.newaxis,:], mask=n_events[np.newaxis,:] == 0, cmap='Greys',
                            vmin=0, cbar=cbar, cbar_kws={'label': '(-)'}, ax=ax_events)
                xticks = np.arange(n_events.shape[-1]) + .5
                step = kwargs.get('xtick_step', 2)
                ax_events.set_xticks(xticks[1::step])
                ax_events.set_xticklabels(n_events.index[1::step], rotation=90)
                ax_events.set_xlabel(kwargs.get('xlabel', xdim))
                ax_events.set_title('no. event')
                ax_events.set_yticks([])
                ax_events.tick_params(length=0)
                
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
        
        
        
def plot_DataArray(da, ax=None, **kwargs):
    """It creates a heatmap plot of a 2D DataArray
    
    Input:
    ------
    da:   xarray.DataArray (n,m)
    ax:   matplotlib.axes
    """
    
    # extract kwargs
    figsize = kwargs.get('figsize', (16, 2))
    xtick_step = kwargs.get('xtick_step', 1)
    ytick_step = kwargs.get('ytick_step', 1)
    cmap = kwargs.get('cmap', 'magma')
    norm = kwargs.get('norm', None)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    cbar = kwargs.get('cbar', True)
    cbar_kws = kwargs.get('cbar_kws', None)
    xrotation = kwargs.get('xrotation', 90)
    yrotation = kwargs.get('yrotation', 0)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if len(da.shape) == 2:
        da_plot = da
        dimy, dimx = da.dims
    elif len(da.shape) == 1:
        da_plot = np.array(da)[np.newaxis,:]
        dimx = da.dims[0]
    hm = sns.heatmap(da_plot, cmap=cmap, norm=norm, ax=ax, vmin=vmin, vmax=vmax, cbar=cbar, cbar_kws=cbar_kws)
    if cbar is False:
        plot_DataArray.colorbar = hm
    
    # if 'yticklabels' in kwargs:
    if 'dimy' in locals():
        yticklabels = kwargs.get('yticklabels', da[dimy].data)
        if isinstance(yticklabels, list):
            yticklabels = np.array(yticklabels)
        yticks = np.arange(0, len(yticklabels), ytick_step) + .5
        if yticklabels.dtype == '<M8[ns]':
            yticklabels = [datetime.strftime(idx, '%Y-%m-%d') for idx in pd.to_datetime(yticklabels[::ytick_step])]
        else:
            yticklabels = yticklabels[::ytick_step]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=yrotation)
    else:
        ax.set_yticks([])
    
    # if 'xticklabels' in kwargs:
    if 'dimx' in locals():
        xticklabels = kwargs.get('xticklabels', da[dimx].data)
        if isinstance(xticklabels, list):
            xticklabels = np.array(xticklabels)
        xticks = np.arange(0, len(xticklabels), xtick_step) + .5
        if xticklabels.dtype == '<M8[ns]':
            xticklabels = [datetime.strftime(idx, '%Y-%m-%d') for idx in pd.to_datetime(xticklabels[::xtick_step])]
        else:
            xticklabels = xticklabels[::xtick_step]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=xrotation)
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    
    ax.tick_params(length=0);
    
    
    
def plot_skill_eventwise(skill, xdim='probability', ydim='combination', save=None, **kwargs):
    """Plot heatmaps of recall, precision and f1-score for the eventwise skill analysis
    
    Inputs:
    -------
    skill:  xr.Dataset. It must contain a DataArray for each metric, whose dimensions are 'model' and 'probability'
    save:   str. Filename (including directory and extension) where the image might be saved.
    """
    
    # extract keyword arguments
    cmap = kwargs.get('cmap', 'Blues')
    norm = kwargs.get('norm', None)
    figsize = kwargs.get('figsize', (9, 5.5))
    
    # find the value of 'xdim' that maximizes f1
    best_idx = skill['f1'].argmax(xdim)
    best_p = skill['f1'].idxmax(xdim)
    
    fig, axes = plt.subplots(nrows=len(skill), figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    for ax, (metric, da) in zip(axes, skill.items()):
        # plot heatmap
        plot_DataArray(da, ax=ax, cmap=cmap, norm=norm, title=metric, cbar=False, yrotation=0)
        # add rectangles and text with the best performing model
        for y, model in enumerate(skill[ydim].data):
            x = best_idx.sel({ydim: model}).data
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fc="none", edgecolor='red'))
            txt = '{0:.2f}'.format(skill[metric].sel({xdim: best_p.sel({ydim: model}), ydim: model}).data)
            ax.text(x + .5, y + .5, txt, horizontalalignment='center', verticalalignment='center', color='w', fontsize=9)
    ax.set(xlabel='probability (-)')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:], shrink=.5, label='metric (-)');

    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        
        
        
def lineplot_skill(ds, metric='f1', xdim='probability', rowdim='persistence', coldim=None, linedim='approach', save=None, **kwargs):
    """It creates a lineplot with the results of the eventwise skill analysis. A series of plots will be created. If 'coldim' is None,  the columns represent the metrics (variables of the Dataset 'ds'). If 'coldim' is not None, columns represent the dimension specified and the different metrics (variables in the Dataset 'ds') are represented by line colours.
    
    Inputs:
    -------
    ds:       xr.Dataset. It contains the arrays of skill for several metrics. At least, it should have the variables  for the chosen target metric (see attribute "metric"), 'recall' and 'precision'.
    metric:   string. Name of the skill metric for which the criteria will be optimize. This name should be one of the variables in the Dataset 'ds'. By default, f1
    xdim:     string. It defines the dimension in 'ds' that will correspond to the X axis in the plots
    rowdim:   string. It defines the dimension in 'ds' that will correspond to the rows in which the graph will be divided
    coldim:   string. It defines the dimension in 'ds' that will correspond to the cols in which the graph will be divided. If None (default), each column will represent a different skill metric (variables in 'ds')
    linedim:  string. It defines the dimension in 'ds' that will correspond to the different lines in the plots
    save:     string. Directory and filename (including extension) where the graph will be saved
    
    Output:
    -------
    """

    # extract kwargs
    alpha = kwargs.get('alpha', .8)
    lw = kwargs.get('linewidth', .8)
    
    xmin, xmax = ds[metric][xdim].min(), ds[metric][xdim].max()
    xlim = kwargs.get('xlim', (xmin, xmax))
    ylim = kwargs.get('ylim', (-.025, 1.025))
    r = kwargs.get('round', 3)
    
    if coldim is None:
        
        # extract best 'X' value and score for each combination of 'rowdim' and 'linedim'
        # best_x = ds[metric].round(r).idxmax(xdim)
        best_x = find_best_criterion(ds, dim=xdim, metric=metric)[xdim]
        
        ncols = len(ds)
        nrows = len(ds[rowdim])
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)
        colors = kwargs.get('color', {'current': 'steelblue', '1_deterministic_+_1_probabilistic': 'steelblue',
                                      'model_mean': 'lightsteelblue', 'member_weighted': 'C1', 'brier_weighted': 'navajowhite'})
        
        for j, (col, da) in enumerate(ds.items()):
            for i, row in enumerate(da[rowdim].data):
                ax = axes[i, j]
                for c, line in enumerate(da[linedim].data):
                    loc = {rowdim: row, linedim: line}
                    label = line.replace('_', ' ')
                    ax.plot(da[xdim], da.sel(loc), lw=lw, c=colors[line], alpha=alpha, label=label, zorder=5 - c)#f'C{c}'
                    x = best_x.sel(loc).data
                    y = da.sel(loc).sel({xdim: x}).data
                    ax.vlines(x, ylim[0], y, lw=.5, color=colors[line], alpha=alpha, ls='--', zorder=0)
                    ax.hlines(y, xlim[0], x, lw=.5, color=colors[line], alpha=alpha, ls='--', zorder=0)
                    # ax.scatter(x, y, marker='+', color=colors[line])
                if i == 0:
                    ax.set_title(col)#, fontsize=11)
                elif i == nrows - 1:
                    ax.set_xlabel(f'{xdim}')
                if j == 0:
                    ax.set_ylabel(kwargs.get('ylabel', 'skill (-)'))
                    ax.text(-.3, 1, f'{rowdim}\n{row}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')#, fontsize=11
                if 'aspect' in kwargs:
                    ax.set_aspect(kwargs['aspect'])
            
        hndls, lbls = ax.get_legend_handles_labels()
        # fig.legend(hndls, lbls, bbox_to_anchor=kwargs.get('loc_legend', [0.3, -.04, .5, .1]), ncol=len(ds[linedim]));
        fig.legend(hndls, lbls, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [1., .8, .2, .1]));
                
    else:
        # extract best 'X' value and score for each combination of 'rowdim' and 'coldim'
        agg_metric = ds[metric].round(r).max(linedim)
        best_x = agg_metric.idxmax(xdim)
        best_metric = agg_metric.max(xdim)
        
        ncols, nrows = len(ds[coldim]), len(ds[rowdim])
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)
        colors = kwargs.get('color', {metric: 'k', 'recall': 'steelblue', 'precision': 'firebrick'})

        for i, row in enumerate(ds[rowdim].data):
            for j, col in enumerate(ds[coldim].data):
                ax = axes[i,j]
                loc = {coldim:col, rowdim:row}
                
                # plot metrics
                ds_sel = ds.sel(loc)
                for lw, (metric, da) in zip([lw / 2, lw / 2, lw], ds_sel[['recall', 'precision', metric]].items()):
                    try:
                        ax.plot(ds_sel[xdim], da, ls='-', lw=lw, c=colors[metric], alpha=alpha, label=metric)
                    except:
                        ax.plot(ds_sel[xdim], da.transpose(), ls='-', lw=lw, c=colors[metric], alpha=alpha, label=metric)
                
                # best-performing x value
                x = best_x.sel(loc)
                y = best_metric.sel(loc)
                ax.vlines(x, ylim[0], y, 'k', ':', lw=lw * .8, alpha=alpha)
                ax.hlines(y, xlim[0], x, 'k', ':', lw=lw * .8, alpha=alpha)
                
                # config
                if i == 0:
                    ax.set_title(f'{coldim}: {col}')
                elif i == nrows - 1:
                    ax.set_xlabel(xdim)
                if j == 0:
                    ax.set_ylabel('skill (-)')
                    ax.text(-.3, 1, f'{rowdim}\n{row}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')#, fontsize=12
    
        hndls, lbls = ax.get_legend_handles_labels()
        hndls = hndls[::len(ds[linedim])]
        lbls = lbls[::len(ds[linedim])]
        fig.legend(hndls, lbls, bbox_to_anchor=kwargs.get('loc_legend', [0.075, -.04, .5, .1]), ncol=len(ds));
    
    ax.set(xlim=(xmin, xmax), ylim=ylim)
    xticks = kwargs.get('xticks', ds[xdim][1::kwargs.get('xtick_step', 4)])
    ax.set_xticks(xticks)
              
    # save dictionary of the best criteria
    best_criteria = ds[metric].argmax(list(ds.dims))
    best_criteria = {dim: ds.isel(best_criteria)[dim].data for dim in best_criteria}
    lineplot_skill.best_criteria = best_criteria

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def create_cmap(cmap, bounds, name='', specify_color=None):
    """Given the name of a colour map and the boundaries, it creates a discrete colour ramp for future plots
    
    Inputs:
    ------
    cmap:          string. Matplotlib's name of a colourmap. E.g. 'coolwarm', 'Blues'...
    bounds:        list. Values that define the limits of the discrete colour ramp
    name:          string. Optional. Name given to the colour ramp
    specify_color: tuple (position, color). It defines a specific color for a specific position in the colour scale. Position must be an integer, and color must be either a colour name or a tuple of 4 floats (red, gren, blue, transparency)
    
    Outputs:
    --------
    cmap:   List of colours
    norm:   List of boundaries
    """
    
    cmap = plt.get_cmap(cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if specify_color is not None:
        cmaplist[specify_color[0]] = specify_color[1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm



def map_hits(stations, cols=['TP', 'FN', 'FP'], mask=None, rivers=None, save=None, **kwargs):
    """It creates a graph that plots both a map and a histogram for each of the variables in 'cols'. These plots show the performance of reporting points individually.
    
    Inputs:
    -------
    stations:   pd.DataFrame (n_station, m). It must contain at least the columns X and Y (to be able to plot the map) and the columns specified in 'cols'
    cols:       list. List of columns to be plotted. For each column a map and a histogram will be drawn
    mask:       pd.Series (n_stations,). A boolean series with the selection of stations to skip. This is meant to skip stations without observed flood events in the plots of true positives (TP) and false negatives (FN)
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
    """
    
    cols_map = {'TP': 'hits', 'FN': 'misses', 'FP': 'false alarms'}
    
    # set up the plots
    ncols = len(cols)
    fig = plt.figure(figsize=kwargs.get('figsize', (ncols * 5, 6)), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=ncols, height_ratios=kwargs.get('height_ratios', [5, 1]))

    # find maximum value of the Y axis in the histograms
    r = kwargs.get('round', 100)
    ymax = max(stations[col].value_counts().max() for col in cols)
    ymax = np.ceil(ymax / r) * r
    
    # find projection
    if 'proj' not in kwargs:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
    else:
        proj = kwargs['proj']

    # plot maps
    for i, (col) in enumerate(cols):
        # pandas.Series of the column
        z = stations[col]
        
        # map
        ax_map = fig.add_subplot(gs[0, i], projection=proj)
        cmax = z.max()
        if 'TP' in col:
            cmap, norm = create_cmap('Blues', np.arange(0, cmax + 2, 1), col, specify_color=(0, (.98, .65, .25, 1)))#(0, (.95, .5, .5, 1)))
        else:
            cmap, norm = create_cmap('Oranges', np.arange(0, cmax + 2, 1), col, specify_color=(0, (.27, .50, .70, 1)))
        if ('TP' in col or 'FN' in col) and (mask is not None):
            plot_map_stations(stations.X, stations.Y, z, ax=ax_map, mask=~mask,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=cols_map[col])
            z = z[mask]
        else:
            plot_map_stations(stations.X, stations.Y, z, ax=ax_map,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=cols_map[col])
        ticks = np.arange(cmax + 1).astype(int)
        if len(ticks) > 6:
            ticks = ticks[::2]
        cbar = plt.colorbar(plot_map_stations.colorbar, ax=ax_map, shrink=.5, label=None, ticks=ticks + .5)
        cbar.ax.set_yticklabels(ticks)
        cbar.ax.tick_params(size=0)
        ax_map.text(.5, -.1, f'total {cols_map[col]}: {z.sum():.0f}', horizontalalignment='center', transform=ax_map.transAxes)
        
        # plot rivers
        if rivers is not None:
            rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax_map, zorder=0)
        
        # histogram
        ax_hist = fig.add_subplot(gs[1, i])
        counts = z.value_counts()
        counts.sort_index(inplace=True)
        color = [cmap(i) for i in np.linspace(0, cmap.N, norm.N).astype(int)]
        plt.bar(counts.index, counts, width=1, alpha=.66, color=color)
        ax_hist.spines[['right', 'top']].set_visible(False)
        if i == 0:
            ylabel = 'no. points'
        else:
            ylabel = None
        ax_hist.set(ylim=(0, ymax), ylabel=ylabel, xticks=np.arange(0, cmax + 1))
        
        # ancillary texts
        n_points = z.shape[0]
        n_zeros = counts.loc[0]
        ax_hist.text(.5, 1.15, f'total points: {n_points}', horizontalalignment='center', transform=ax_hist.transAxes)
        ax_hist.text(0, n_zeros + ymax / 40, n_zeros, horizontalalignment='center')
    
    if 'title' in kwargs:
        fig.text(.5, 1.1, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontsize=13)
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300);

        
        
def map_skill(stations, cols=['recall', 'precision', 'f1'], bins=50, cmap='coolwarm', norm=None, rivers=None, save=None, **kwargs):
    """It creates a graph that plots both a map and a histogram for each of the variables in 'cols'. These plots show the performance of reporting points individually.
    
    Inputs:
    -------
    stations:   pd.DataFrame (n_station, m). It must contain at least the columns X and Y (to be able to plot the map) and the columns specified in 'cols'
    cols:       list. List of columns to be plotted. For each column a map and a histogram will be drawn
    bins:       int. Number of bins in which the histograms will be divided
    cmap:       string. Matplotlib colormap to be used in the plots
    norm:       matplotlib.colors.BoundaryNorm. Used to create a discrete colour scale out of 'cmap'
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
    """

    # set up the plots
    fig = plt.figure(figsize=kwargs.get('figsize', (15, 6)), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=kwargs.get('height_ratios', [5, 1]))
    axes = np.empty((2, 3)).astype('object')
    
    # find maximum value of the Y axis in the histograms
    alpha = kwargs.get('alpha', .66)
    r = kwargs.get('round', 100)
    ymax = max((stations[cols] == x).sum().max() for x in [0, 1])
    ymax = np.ceil(ymax / r) * r
    
    # find projection
    if 'proj' not in kwargs:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
    else:
        proj = kwargs['proj']
    
    # generate plots for each column
    for i, (col) in enumerate(cols):
        ax_map = fig.add_subplot(gs[0, i], projection=proj)
        mask = stations[col].isnull()
        plot_map_stations(stations.X, stations.Y, stations[col], ax=ax_map, mask=mask,
                          cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=alpha, title=col)
        axes[0, i] = ax_map
        
        # plot rivers
        if rivers is not None:
            rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax_map, zorder=0)

        # histogram
        ax_hist = fig.add_subplot(gs[1, i])
        counts = stations[col].value_counts(bins=bins, sort=False)
        counts.index = [idx.left.round(2) for idx in counts.index]
        ax_hist.bar(counts.index, counts, align='edge', width=1/bins, color=cmap(np.linspace(0, 1, bins)), alpha=alpha)
        ax_hist.spines[['right', 'top']].set_visible(False)
        if i == 0:
            ylabel = 'no. points'
        else:
            ylabel = None
        ax_hist.set(xlim=(0, 1), ylim=(0, ymax), ylabel=ylabel)
        n_points = stations.shape[0] - mask.sum()
        n_zeros = (stations[col] == 0).sum()
        n_ones = (stations[col] == 1).sum()
        ax_hist.text(.5, 1.4, f'total points: {n_points}', horizontalalignment='center', transform=ax_hist.transAxes)
        ax_hist.text(0 + .02, n_zeros + 10, n_zeros, horizontalalignment='left')
        ax_hist.text(1 - .02, n_ones + 10, n_ones, horizontalalignment='right')
        axes[1, i] = ax_hist

    cbar = fig.colorbar(plot_map_stations.colorbar, ax=axes[:,2], shrink=.333, label=f'score (-)');
    cbar.ax.tick_params(size=0)
    
    if 'title' in kwargs:
        fig.text(.5, 1.1, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontsize=13)
        
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300);
        
        
        
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
        
        
        
def lineplot_hits(ds, xdim='probability', coldim='persistence', rowdim=None, linedim='approach', beta=1, yscale='linear', xtick_step=4, save=None, **kwargs):
    """It creates several lineplots of hits (TP), misses (FN) and false alarms (FP).
    
    Depending on the dimensions of the analysis, a plot will be created for each combination of 'rowdim' and 'coldim'.
    
    Inputs:
    -------
    ds:       xr.Dataset. It contains three DataArrays with names 'TP' (hits), 'FN' (misses) and 'FP' (false alarms). Its dimensions must be those defined in `xdim`, `coldim`, `rowdim` and `linedim
    xdim:       string. Dimension in `ds` that will be represented in the X axis of the plots
    coldim:     string. Dimension in `ds` that will be used to create several columns of plots
    rowdim:     string. Dimension in `ds` that will be used to create several rows of plots. If None, there will be only one single line of plots
    yscale:     string. Type of scaling used in the Y axis, e.g., 'linear' or 'log'
    xtick_step: int. Frequency of the labels in the X axis
    save:       str. Directory where a JPG file of the plot will be saved
    """
    
    xmin, xmax = ds[xdim].min(), ds[xdim].max()
    
    # compute f-beta score and select best-performing criteria
    #fscore = (1 +  beta**2) * ds['TP'] / ((1 +  beta**2) * ds['TP'] + ds['FP'] + beta**2 * ds['FN'])
    f_beta = (1 +  beta**2) * ds.TP / ((1 +  beta**2) * ds.TP + beta**2 * ds.FN + ds.FP)
    if isinstance(beta, int):
        score = f'f{beta}'
    else:
        score = f'f{beta:.1f}'
    agg_metric = f_beta.max(linedim)
    best_x = agg_metric.idxmax(xdim)
    best_metric = agg_metric.max(xdim)
        
    if rowdim is None:
        ncols = len(ds[coldim])
        fig, axes = plt.subplots(ncols=ncols, figsize=(3 * ncols, 2.75), sharex=True, sharey=True)

        for j, col in enumerate(ds[coldim].data):
            ax = axes[j]
            ax.set_yscale(yscale)
            loc = {coldim: col}

            # true positives (FP)
            tp = ds['TP'].sel(loc)
            # total predicted events: TP + FP
            pred = tp + ds['FP'].sel(loc)
            # total observed events: TP + FN
            obs = tp + ds['FN'].sel(loc)
            
            ax.plot(tp[xdim], (tp / obs).data.transpose(), lw=.5, alpha=.66, c=f'steelblue', label=f'hits', zorder=4)
            ax.plot(pred[xdim], (pred / obs).data.transpose(), lw=.5, ls='-', alpha=.66, c=f'firebrick', label='predicted', zorder=3)
            ax.axhline(1, lw=.5, ls='-', color='k', label='observed', zorder=5)
            
            # best-performing x value
            x = best_x.sel(loc).data
            text = '{0} = {1:.2f}'.format(score, best_metric.sel(loc).data)
            ax.text((x - xmin) / xmax, .975, text, horizontalalignment='right', verticalalignment='top', rotation=90, transform=ax.transAxes)
            ax.axvline(x, color='k', ls='--', lw=.8)
            
            # config
            ax.set_title(f'{coldim} {col}')
            ax.set_xlabel(f'{xdim}')
            ax.set(xlim=(xmin, xmax), ylim=kwargs.get('ylim', (0, None)))
            xticks = kwargs.get('xticks', ds[xdim][1::kwargs.get('xtick_step', 4)])
            ax.set_xticks(xticks)

            if j == 0:
                ax.set_ylabel(r'$\frac{x}{obs}$', rotation=0, horizontalalignment='right')
    
    else:
        ncols, nrows = len(ds[coldim]), len(ds[rowdim])
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)

        for i, row in enumerate(ds[rowdim].data):
            for j, col in enumerate(ds[coldim].data):
                ax = axes[i,j]
                ax.set(xlim=(xmin, xmax), ylim=kwargs.get('ylim', (0, None)))
                ax.set_yscale(yscale)
                loc = {rowdim: row, coldim: col}
                
                # true positives (FP)
                tp = ds['TP'].sel(loc)
                # total observed events: TP + FN
                obs = tp + ds['FN'].sel(loc)
                # total predicted events: TP + FP
                pred = tp + ds['FP'].sel(loc)
                
                # ax.plot(obs[xdim], obs.data.transpose(), lw=.5, ls='-', color='k', label='observed', zorder=5)
                ax.axhline(1, lw=.5, ls='-', color='k', label='observed', zorder=5)
                ax.plot(tp[xdim], (tp / obs).data.transpose(), lw=.5, alpha=.66, c=f'steelblue', label='hits', zorder=4)
                ax.plot(pred[xdim], (pred / obs).data.transpose(), lw=.5, ls='-', alpha=.66, c=f'firebrick', label='predicted', zorder=3)
                
                # best-performing x value
                x = best_x.sel(loc).data
                text = '{0} = {1:.2f}'.format(score, best_metric.sel(loc).data)
                ax.text((x - xmin) / xmax, .975, text, horizontalalignment='right', verticalalignment='top', rotation=90, transform=ax.transAxes)
                ax.axvline(x, color='k', ls='--', lw=.8)
                
                if i == 0:
                    ax.set_title(f'{coldim} {col}')
                elif i == nrows - 1:
                    ax.set_xlabel(f'{xdim}')
                    xticks = kwargs.get('xticks', ds[xdim][1::kwargs.get('xtick_step', 4)])
                    ax.set_xticks(xticks)
                if j == 0:
                    ax.set_ylabel(r'$\frac{x}{obs}$', rotation=0, horizontalalignment='right')
                    ax.text(-.35, 1, f'{rowdim}\n{row}', fontsize=11, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
        
    # legend
    hndls, lbls = ax.get_legend_handles_labels()
    hndls = hndls[::len(ds[linedim])]
    lbls = lbls[::len(ds[linedim])]
    fig.legend(hndls, lbls, bbox_to_anchor=kwargs.get('loc_legend', [0.075, -.04, .5, .1]), ncol=len(ds));

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_skill_by_variable(skill, optimal_criteria, variable, reference=None, metric='f1', current_criteria=None, optimized_criteria=None,
                           shades=True, save=None, **kwargs):
    """It generates a graph with as many lineplots as approaches in the 'skill' dataset. The lineplots reprensent both the evolution of skill and probability with regard to a specified variable
    
    Inputs:
    -------
    skill:              xr.Dataset (area, persistence, approach, probability). It contains as variables recall, precision and the specified metric
    optimal_criteria:   dict. For each approach in skill, it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable:           string. Name of the variable in 'hits' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    reference:          int of float. Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric:             string. Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria:   dict. It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    optimized_criteria: xr.Dataset (variable, approach, persistence). It contains as variables probability, recall, precision and the specified metric  
    shades:             boolean. If True, a shaded shape shows the difference bewteen recall and precision
    save:               string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Ouput:
    ------
    The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """

    colors = kwargs.get('color', ['k', 'steelblue', 'orange'])
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .15)
    loc_text = kwargs.get('loc_text', 1)
      
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=kwargs.get('figsize', (18, 4)))
    
    if current_criteria is not None:
        skill_current = skill.sel(current_criteria).to_pandas().drop(['probability', 'approach', 'persistence'], axis=1)
        y1_current = skill_current[['recall', 'precision']].min(axis=1)
        y2_current = skill_current[['recall', 'precision']].max(axis=1)

    for ax1, approach in zip(axes, skill.approach.data):

        # SKILL
        # -----
        
        # skill for the current operational criteria
        if current_criteria is not None:
            if shades:
                ax1.fill_between(skill_current.index, y1_current, y2_current, alpha=alpha * .5, color=colors[0], zorder=0, label=f'P-R (current)')
                ax1.plot(skill_current.index, skill_current[metric], c=colors[0], lw=lw, label=f'{metric} (current)', zorder=6)

        # skill for the optimal criteria (that optimized for the reference value of the variable)
        skill_optimal = skill.sel(optimal_criteria[approach]).to_pandas().drop(['probability', 'approach', 'persistence'], axis=1)
        if shades:
            y1_optimal = skill_optimal[['recall', 'precision']].min(axis=1)
            y2_optimal = skill_optimal[['recall', 'precision']].max(axis=1)
            ax1.fill_between(skill_optimal.index, y1_optimal, y2_optimal, alpha=alpha, color=colors[1], zorder=7, label=f'P-R (optimal)')
        ax1.plot(skill_optimal.index, skill_optimal[metric], c=colors[1], lw=lw, label=f'{metric} (optimal)', zorder=3)
        persistence = optimal_criteria[approach]['persistence']
        
        # skill optimized for each value of the target variable
        if optimized_criteria is not None:
            probability = optimized_criteria.sel(approach=approach, persistence=persistence).data
            skill_var = pd.DataFrame(index=optimized_criteria[variable].data, columns=list(skill))
            for v, p in zip(skill_var.index, probability):
                if np.isnan(p): # due to persistence
                    continue
                skill_var.loc[v] = skill.sel({variable: v, 'probability': p, 'approach': approach, 'persistence': persistence}).to_pandas()
            if shades:
                y1_var = skill_var[['recall', 'precision']].min(axis=1)
                y2_var = skill_var[['recall', 'precision']].max(axis=1)
                ax1.fill_between(skill_var.index, y1_var, y2_var, alpha=alpha, color=colors[2], zorder=2, label=f'P-R ({variable} optimized)')
            ax1.plot(skill_var.index, skill_var[metric], c=colors[2], lw=lw, label=f'{metric} ({variable} optimized)', zorder=8)

        # reference line
        if reference is not None:
            ax1.axvline(x=reference, ls='-', lw=.5, color='k')

        # settings
        ax1.set_title(approach.replace('_', ' '))
        if loc_text == 1:
            x, y, ha, va = .975, .975, 'right', 'top'
        elif loc_text == 2:
            x, y, ha, va = .025, .975, 'left', 'top'
        elif loc_text == 3:
            x, y, ha, va = .025, .025, 'left', 'bottom'
        elif loc_text == 4:
            x, y, ha, va = .975, .025, 'right', 'bottom'
        else:
            print('WARNING. Parameter "loc_text" must be either 1, 2, 3 or 4')
            x, y, ha, va = .975, .975, 'right', 'top'
        criteria_disp = [x is not None for x in [current_criteria, optimal_criteria, optimized_criteria]]
        if criteria_disp == [True, True, False]:
            color = colors[1]
        elif criteria_disp == [True, False, True]:
            color = colors[2]
        elif criteria_disp == [False, True, True]:
            color = 'k'
        elif all(criteria_disp):
            color = colors[1]
        ax1.text(x, y, f'persistence = {persistence}', color=color, horizontalalignment=ha, verticalalignment=va, transform=ax1.transAxes,
                 backgroundcolor='w')
        ax1.set(xlabel=kwargs.get('xlabel', variable),
                xlim=kwargs.get('xlim', (skill[variable].min(), skill[variable].max())),
                xscale=kwargs.get('xscale', 'linear'),
                ylim=(-.025, 1.025))
        if ax1 == axes[0]:
            ax1.set_ylabel('skill')
        if 'xticks' in kwargs:
            step = int(kwargs['xticks'])
            xticks = skill[variable].data[step::step]
            ax1.set_xticks(xticks)
        
        # PROBABILITY
        # -----------

        ax2 = ax1.twinx()
        
        # probability of the current operational criteria
        if current_criteria is not None:
            prob = current_criteria['probability']
            # pers = int(str(current_criteria['persistence']).split('/')[0])
            ax2.axhline(prob, color=colors[0], lw=lw, ls=':', zorder=3, label='prob. (current)')

        # probability of the optimal criteria
        if optimal_criteria is not None:
            p = int(str(persistence).split('/')[0])
            ax2.axhline(optimal_criteria[approach]['probability'],
                       color=colors[1], lw=lw, ls=':', zorder=4, label='prob. (optimal)')

        # probability optimized for each value of the target variable
        if optimized_criteria is not None:
            ax2.plot(optimized_criteria[variable].data, probability,
                     c=colors[2], lw=lw, ls=':', zorder=5, label=f'prob. ({variable} optimized)')

        # settings
        ax2.set(ylim=(-.025, 1.025))
        if ax1 == axes[-1]:
            ax2.set_ylabel('probability')
        else:
            ax2.set_yticklabels([])
            
    # legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles, labels = [], []
    for i in range(len(labels2)):
        handles += handles1[i * 2:i * 2 + 2] + [handles2[i]]
        labels += labels1[i * 2:i * 2 + 2] + [labels2[i]]
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.945, .8, .2, .1]))
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_hits_by_variable(hits, optimal_criteria, variable, reference=None, current_criteria=None, optimized_criteria=None, save=None, **kwargs):
    """It generates a graph with as many lineplots as approaches in the 'hits' dataset. The lineplots reprensent both the evolution of true positives (hits) and false positives (false alarms) and probability with regard to a specified variable
    
    Inputs:
    -------
    hits:               xr.Dataset (area, persistence, approach, probability). It contains as variables TP (true positives), FN (false negatives) and FP (false positives)
    optimal_criteria:   dict. For each approach in 'hits', it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable:           string. Name of the variable in 'hits' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    reference:          int of float. Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    current_criteria:   dict. It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    optimized_criteria: xr.DataArray (variable, approach, persistence). It contains the optimized probability threshold for each combination of the 'variable', approach and persistence
    save:               string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Ouput:
    ------
    The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    colors = kwargs.get('colors', ['k', 'steelblue', 'orange'])
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .15)
    loc_text = kwargs.get('loc_text', 1)
    
    fig, axes = plt.subplots(ncols=4, figsize=kwargs.get('figsize', (18, 4)), sharex=True, sharey=True)
    
    if current_criteria is not None:
        obs_current = hits.sel(current_criteria)['TP'] + hits.sel(current_criteria)['FN']
        pred_current = hits.sel(current_criteria)['TP'] + hits.sel(current_criteria)['FP']
    
    for ax1, (approach, criteria) in zip(axes, optimal_criteria.items()):
        
        # HITS, FALSE POSITIVES
        # ---------------------
        
        ds = hits.sel(approach=approach)
        
        # hits/false alarms for the current operational criteria
        if current_criteria is not None:
            pred = pred_current / obs_current
            tp = hits.sel(current_criteria)['TP'] / obs_current
            ax1.fill_between(hits[variable], tp, pred, color=colors[0], alpha=alpha * .5, zorder=0, label='FP (current)')
            ax1.plot(hits[variable], tp, c=colors[0], label=f'TP (current)', lw=lw, ls='-', zorder=3)
        
        # hits/false alarms for the optimized criteria
        obs = hits.sel(criteria)['TP'] + hits.sel(criteria)['FN']
        pred = hits.sel(criteria)['TP'] + hits.sel(criteria)['FP']
        ax1.fill_between(hits[variable], hits.sel(criteria)['TP'] / obs, pred / obs, color=colors[1], alpha=alpha, zorder=1, label='FP (optimal)')
        ax1.plot(hits[variable], hits.sel(criteria)['TP'] / obs, c=colors[1], label=f'TP (optimal)', lw=lw, zorder=5)
        persistence = optimal_criteria[approach]['persistence']
        
        # hits/false alarms optimized for each value of the target variable
        if optimized_criteria is not None:
            probability = optimized_criteria.sel(approach=approach, persistence=persistence)
            hits_var = pd.DataFrame(index=probability[variable].data, columns=list(hits))
            for v, p in zip(hits_var.index, probability.data):
                if np.isnan(p): # due to persistence
                    continue
                hits_var.loc[v] = hits.sel({variable: v, 'probability': p, 'approach': approach, 'persistence': persistence}).to_pandas()
            obs = hits_var['TP'] + hits_var['FN']
            pred = hits_var['TP'] + hits_var['FP']
            y1 = (hits_var['TP'] / obs).astype(float)
            y2 = (pred / obs).astype(float)
            ax1.fill_between(hits_var.index, y1, y2, color=colors[2], alpha=alpha, zorder=2, label=f'FP ({variable} optimized)')
            ax1.plot(hits_var.index, y1, c=colors[2], lw=lw, label=f'TP ({variable} optimized)', zorder=5)
                
        # reference lines
        ax1.axhline(1, c='k', lw=.5, zorder=0)
        if reference is not None:
            ax1.axvline(reference, c='k', lw=.5, zorder=0)

        # settings
        ax1.set(xscale=kwargs.get('xscale', 'linear'),
               xlabel=kwargs.get('xlabel', variable),
               ylim=kwargs.get('ylim', (-.05, 2.05)),
               xlim=kwargs.get('xlim' ,(hits[variable].min(), hits[variable].max())))
        ax1.set_title(approach.replace('_', ' '))
        if loc_text == 1:
            x, y, ha, va = .975, .975, 'right', 'top'
        elif loc_text == 2:
            x, y, ha, va = .025, .975, 'left', 'top'
        elif loc_text == 3:
            x, y, ha, va = .025, .025, 'left', 'bottom'
        elif loc_text == 4:
            x, y, ha, va = .975, .025, 'right', 'bottom'
        ax1.text(x, y, f'persistence = {persistence}', color=colors[1], horizontalalignment=ha, verticalalignment=va, transform=ax1.transAxes,
                 backgroundcolor='w')
        if ax1 == axes[0]:
            ax1.text(-.15, .5, r'$\frac{x}{obs}$', horizontalalignment='right', transform=ax1.transAxes)
        if 'xticks' in kwargs:
            step = int(kwargs['xticks'])
            xticks = hits[variable].data[step::step]
            ax1.set_xticks(xticks)
        
        # PROBABILITY
        # -----------
        
        ax2 = ax1.twinx()
        
        # probability of the current operational criteria
        if current_criteria is not None:
            prob = current_criteria['probability']
            ax2.axhline(prob, color=colors[0], lw=lw, ls=':', zorder=3, label='prob. (current)')
        
        # probability of the optimal criteria
        if optimal_criteria is not None:
            ax2.axhline(optimal_criteria[approach]['probability'],
                       color=colors[1], lw=lw, ls=':', zorder=4, label='prob. (optimal)')
            
        # probability optimized for each value of the target variable
        if optimized_criteria is not None:
            ax2.plot(optimized_criteria[variable].data, probability,
                     c=colors[2], lw=lw, ls=':', zorder=5, label=f'prob. ({variable} optimized)')
            
        # settings
        ax2.set(ylim=(-.025, 1.025))
        if ax1 == axes[-1]:
            ax2.set_ylabel('probability')
        else:
            ax2.set_yticklabels([])
            
    # legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles, labels = [], []
    for i in range(len(labels2)):
        handles += handles1[i * 2:i * 2 + 2] + [handles2[i]]
        labels += labels1[i * 2:i * 2 + 2] + [labels2[i]]
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.9, .8, .2, .1]))
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')