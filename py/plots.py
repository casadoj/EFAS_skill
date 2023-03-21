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
        ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 10, c='dimgray', alpha=kwargs.get('alpha', .5),
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
        
        
        
def lineplot_skill(ds, xdim='probability', rowdim='persistence', linedim='approach', obs=None, bestvar=None, yscale='log', verbose=True, save=None, **kwargs):
    """It creates a lineplot with the results of the eventwise skill analysis. A series of plots will be created, where columns are the metrics (variables).
    
    Inputs:
    -------
    ds:       xr.Dataset. Contains the results. The variables will correspond to the columns in the graph. The use of the different dimensions is controlled by the following attributes
    xdim:     string. It defines the
    dimension in 'ds' that will correspond to the X axis in the plots
    rowdim:   string. It defines the dimension in 'ds' that will correspond to the rows in which the graph will be divided
    linedim:  string. It defines the dimension in 'ds' that will correspond to the different lines in the plots
    obs:      int. Number of observed events
    bestvar:  string. If used, it is the variable in 'ds' used  to select the best performing model
    yscale:   string. Type of scaling of the Y axis
    save:     string. Directory and filename (including extension) where the graph will be saved
    
    Output:
    -------
    """

    nrows, ncols = len(ds[rowdim]), len(list(ds))
    figsize = kwargs.get('figsize', (ncols * 3, nrows * 3))
    if yscale == 'log':
        r = 10**kwargs.get('round', 4)
    elif yscale == 'linear':
        r = kwargs.get('round', 1000)
    ymax = [ds[var].max().data for var in list(ds)]
    ymax = np.ceil(np.max(ymax) / r) * r
    ylim = (0, ymax)
    xlim = kwargs.get('xlim', (ds[xdim].min(), ds[xdim].max()))
    alpha = kwargs.get('alpha', .66)
    lw = kwargs.get('linewidth', .8)
    
    # find the value of 'xdim' that maximizes f1
    if bestvar is not None:
        best = ds[bestvar].idxmax(xdim)
        
        if verbose:
            best_criteria = ds['f1'].argmax(list(ds.dims))
            best_criteria = {dim: ds.isel(best_criteria)[dim].data for dim in best_criteria}
            lineplot_skill.best_criteria = best_criteria
            print('Best criteria:')
            print('--------------\n')
            for dim in ds.dims:
                print('{0}:\t{1}'.format(dim, best_criteria[dim]))
            print()
            for var in list(ds):
                print('{0:>10} = {1:.3f}'.format(var, ds[var].sel(best_criteria).data))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    for j, (col, da) in enumerate(ds.items()):
        for i, row in enumerate(da[rowdim].data):
            ax = axes[i, j]
            ax.set_yscale(yscale)
            if obs is not None:
                ax.hlines(obs, xlim[0], xlim[1], color='k', ls='-', lw=lw)
            for c, line in enumerate(da[linedim].data):
                ax.plot(da[xdim], da.sel({rowdim: row, linedim: line}), lw=lw, c=f'C{c}', alpha=alpha, label=line)
                if bestvar is not None:
                    x = best.sel({rowdim: row, linedim: line}).data
                    y = da.sel({xdim: x, rowdim: row, linedim: line}).data
                    ax.vlines(x, ylim[0], y, lw=.5, color=f'C{c}', alpha=alpha, ls='--', zorder=0)
                    ax.hlines(y, xlim[0], x, lw=.5, color=f'C{c}', alpha=alpha, ls='--', zorder=0)
                    ax.scatter(x, y, marker='+', color=f'C{c}')
            if i == 0:
                ax.set_title(col, fontsize=11)
            elif i == nrows - 1:
                ax.set_xlabel('probability (-)')
            if j == 0:
                ax.set_ylabel(f'{rowdim} {row}', fontsize=11)
            if 'aspect' in kwargs:
                ax.set_aspect(kwargs['aspect'])

    ax.set(xlim=xlim, ylim=ylim)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=len(ds[linedim]), bbox_to_anchor=[.1, .0075 * len(ds[rowdim]), .8, .1])
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        
        
        
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



def map_hits(stations, cols=['TP', 'FN', 'FP'], mask=None, save=None, **kwargs):
    """It creates a graph that plots both a map and a histogram for each of the variables in 'cols'. These plots show the performance of reporting points individually.
    
    Inputs:
    -------
    stations:   pd.DataFrame (n_station, m). It must contain at least the columns X and Y (to be able to plot the map) and the columns specified in 'cols'
    cols:       list. List of columns to be plotted. For each column a map and a histogram will be drawn
    mask:       pd.Series (n_stations,). A boolean series with the selection of stations to skip. This is meant to skip stations without observed flood events in the plots of true positives (TP) and false negatives (FN)
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
    """
    
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
        if col == 'TP':
            cmap, norm = create_cmap('Blues', np.arange(0, cmax + 1, 1), col, specify_color=(0, (.95, .5, .5, 1)))
        else:
            cmap, norm = create_cmap('Reds', np.arange(0, cmax + 1, 1), col, specify_color=(0, (.27, .50, .70, 1)))
        if (col in ['TP', 'FN']) and (mask is not None):
            plot_map_stations(stations.X, stations.Y, z, ax=ax_map, mask=~mask,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=col)
            z = z[mask]
        else:
            plot_map_stations(stations.X, stations.Y, z, ax=ax_map,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=col)
        plt.colorbar(plot_map_stations.colorbar, ax=ax_map, shrink=.333, label=None);
        
        # histogram
        ax_hist = fig.add_subplot(gs[1, i])
        counts = z.value_counts()
        counts.sort_index(inplace=True)
        color = [cmap(i) for i in np.linspace(0, cmap.N, norm.N).astype(int)]
        plt.bar(counts.index, counts, width=1, alpha=.66, color=color)
        ax_hist.spines[['right', 'top']].set_visible(False)
        if i == 0:
            ylabel = 'count (-)'
        else:
            ylabel = None
        ax_hist.set(ylim=(0, ymax), ylabel=ylabel)
        
        # ancillary texts
        n_points = z.shape[0]
        n_zeros = counts.loc[0]
        ax_hist.text(.5, 1.4, f'no. total points: {n_points}', horizontalalignment='center', transform=ax_hist.transAxes)
        ax_hist.text(.5, 1.2, f'no. total {col}: {z.sum()}', horizontalalignment='center', transform=ax_hist.transAxes)
        ax_hist.text(0, n_zeros + 20, n_zeros, horizontalalignment='center')

    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300);
        
        
        
def map_skill(stations, cols=['recall', 'precision', 'f1'], bins=50, cmap='coolwarm', norm=None, save=None, **kwargs):
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
                          cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=col)
        axes[0, i] = ax_map

        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(stations[col], bins=bins, width=1/bins, alpha=.66)
        ax_hist.spines[['right', 'top']].set_visible(False)
        if i == 0:
            ylabel = 'count (-)'
        else:
            ylabel = None
        ax_hist.set(xlim=(0, 1), ylim=(0, ymax), ylabel=ylabel)
        n_points = stations.shape[0] - mask.sum()
        n_zeros = (stations[col] == 0).sum()
        n_ones = (stations[col] == 1).sum()
        ax_hist.text(.5, 1.4, f'no. total points: {n_points}', horizontalalignment='center', transform=ax_hist.transAxes)
        ax_hist.text(0 + .02, n_zeros + 10, n_zeros, horizontalalignment='left')
        ax_hist.text(1 - .02, n_ones + 10, n_ones, horizontalalignment='right')
        axes[1, i] = ax_hist

    fig.colorbar(plot_map_stations.colorbar, ax=axes[:,2], shrink=.333, label=f'score (-)');
    
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