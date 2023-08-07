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
from optimize import find_best_criterion
  
        
        
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



def map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
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
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
    if ax is None:
        fig = plt.figure(figsize=kwargs.get('figsize', None))
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
        ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 8, c='dimgray', alpha=kwargs.get('alpha', .5),
                   label='stations w/o events', zorder=0)
        x = x[~mask]
        y = y[~mask]
        z = z[~mask]

    # plot non-masked stations
    sct = ax.scatter(x, y, c=z, s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
                    alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
    map_stations.colorbar = sct
    
    # settings
    if ax is None:
        plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
        plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
        # ax.set_extent([-13, 45, 30, 70])
        ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    else:
        map_stations.legend = ax.get_legend_handles_labels()
    ax.axis('off')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
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
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
    if ax is None:
        fig = plt.figure(figsize=kwargs.get('figsize', None))
        ax = plt.axes(projection=proj)
    
    # add polygon of land
    ax.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor=None, facecolor='gray'), alpha=.5, zorder=0)
    # # plot coatslines and country borders
    # ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
    # ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
    # plot rivers
    if rivers is not None:
        rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax, zorder=0)
    
    # plot all the stations
    if mask is not None:
        # plot masked stations
        ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 8, c='dimgray', alpha=kwargs.get('alpha', .5),
                   label='stations w/o events', zorder=0)
        x = x[~mask]
        y = y[~mask]
        z = z[~mask]

    # plot non-masked stations
    sct = ax.scatter(x, y, c=z, s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
                    alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
    map_stations.colorbar = sct
    
    # settings
    if ax is None:
        plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
        plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
        # ax.set_extent([-13, 45, 30, 70])
        ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    else:
        map_stations.legend = ax.get_legend_handles_labels()
    ax.axis('off')
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def map_hits(stations, cols=['TP', 'FN', 'FP'], mask=None, rivers=None, save=None, **kwargs):
    """It creates a graph that plots both a map and a histogram for each of the variables in 'cols'. These plots show the performance of reporting points individually.
    
    Inputs:
    -------
    stations:   pd.DataFrame (n_station, m). It must contain at least the columns X and Y (to be able to plot the map) and the columns specified in 'cols'
    cols:       list. List of columns to be plotted. For each column a map and a histogram will be drawn
    mask:       pd.Series (n_stations,). A boolean series with the selection of stations to skip. This is meant to skip stations without observed flood events in the plots of true positives (TP) and false negatives (FN)
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
    """
    
    cols_map = {'TP': 'hits', 'FN': 'misses', 'FP': 'false alarms', 'n_events_obs': 'eventos'}
    
    # set up the plots
    ncols = len(cols)
    fig = plt.figure(figsize=kwargs.get('figsize', (ncols * 5, 5.5)), constrained_layout=True)
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
            map_stations(stations.X, stations.Y, z, ax=ax_map, mask=~mask,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=cols_map[col])
            z = z[mask]
        else:
            map_stations(stations.X, stations.Y, z, ax=ax_map,
                              cmap=cmap, norm=norm, size=kwargs.get('s', 4), alpha=.66, title=cols_map[col])
        ticks = np.arange(cmax + 1).astype(int)
        if len(ticks) > 6:
            ticks = ticks[::2]
        cbar = plt.colorbar(map_stations.colorbar, ax=ax_map, shrink=.5, label=None, ticks=ticks + .5)
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
        ax_hist.text(.5, 1.15, f'total no. points: {n_points}', horizontalalignment='center', transform=ax_hist.transAxes)
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
    fig = plt.figure(figsize=kwargs.get('figsize', (15, 5.5)), constrained_layout=True)
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
        map_stations(stations.X, stations.Y, stations[col], ax=ax_map, mask=mask,
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

    cbar = fig.colorbar(map_stations.colorbar, ax=axes[:,2], shrink=.333, label=f'score (-)');
    cbar.ax.tick_params(size=0)
    
    if 'title' in kwargs:
        fig.text(.5, 1.1, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontsize=13)
        
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300);
        
        
        
def map_events(stations, col, rivers=None, save=None, **kwargs):
    """It plots a map and a histogram of the number of events.
    
    Inputs:
    -------
    stations:   pd.DataFrame (n_station, m). It must contain at least the columns X and Y (to be able to plot the map) and the column specified in 'col'
    col:        string. Name of the columns of 'stations' that contains the number of events
    rivers:   geopandas. Shapefile of rivers
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
    """
    
    # extract kwargs
    s = kwargs.get('s', 2)
    alpha = kwargs.get('alpha', .5)
    yscale = kwargs.get('yscale', 'linear')
    if ('cmap' not in kwargs) or ('norm' not in kwargs):
        xmax = stations[col].max()
        cmap, norm = create_cmap('Oranges', np.arange(xmax + 2), 'no. events', [0, (0.41176, 0.41176, 0.41176, 1)])
    else:
        cmap = kwargs['cmap']
        norm = kwargs['norm']
    
    # set up the plots
    fig = plt.figure(figsize=kwargs.get('figsize', (7, 7)), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, height_ratios=kwargs.get('height_ratios', [7, 1]))

    # map
    # define projection
    if 'proj' not in kwargs:
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
    ax_map = fig.add_subplot(gs[0], projection=proj)
    map_stations(stations.X, stations.Y, stations[col], rivers=rivers, cmap=cmap, norm=norm, size=s, alpha=alpha, ax=ax_map)

    # histogram
    ax_hist = fig.add_subplot(gs[1])
    counts = stations[col].value_counts()
    counts.sort_index(inplace=True)
    color = [cmap(i) for i in np.linspace(0, cmap.N, norm.N).astype(int)]
    plt.bar(counts.index, counts, width=1, alpha=.66, color=color)
    ax_hist.set(xlabel='no. observed events', xlim=(norm.boundaries.min() - .5, norm.boundaries.max() - .5),
                xticks=norm.boundaries[:-1])
    ax_hist.spines[['right', 'top']].set_visible(False)
    if yscale != 'linear':
        ax_hist.set_yscale(yscale)
    
    if 'title' in kwargs:
        fig.text(.5, 1, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontsize=13)
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')