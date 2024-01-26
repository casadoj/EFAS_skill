import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from optimize import find_best_criterion
from pathlib import Path
from typing import Union, List, Tuple, Dict
  
        
        
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



# def map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
#     """It plots a map of Europe with the reporting points and their number of flood events
    
#     Inputs:
#     -------
#     x:        pandas.Series (stations,). Coordinate X of the stations
#     y:        pandas.Series (stations,). Coordinate Y of the stations
#     z:        pandas.Series (stations,). Number of flood events identified in each station
#     mask:     pandas.Series (stations,). A boolean series of stations to be plotted differently, i.e., not included in the colour scale based on 'z'
#     rivers:   geopandas. Shapefile of rivers
#     ax:       matplotlib.axis. Axis in which the plot will be embedded. If None, a new figure will be created
#     save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
#     Ouput:
#     ------
#     The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
#     """
    
#     # define projection
#     proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
#     if ax is None:
#         fig = plt.figure(figsize=kwargs.get('figsize', None))
#         ax = plt.axes(projection=proj)
    
#     # plot coatslines and country borders
#     ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
#     ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
#     # plot rivers
#     if rivers is not None:
#         rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax, zorder=0)
    
#     # plot all the stations
#     if mask is not None:
#         # plot masked stations
#         ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 8, c='dimgray', alpha=kwargs.get('alpha', .5),
#                    label='stations w/o events', zorder=0)
#         x = x[~mask]
#         y = y[~mask]
#         z = z[~mask]

#     # plot non-masked stations
#     sct = ax.scatter(x, y, c=z, s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
#                     alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
#     map_stations.colorbar = sct
    
#     # settings
#     if ax is None:
#         plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
#         plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
#         # ax.set_extent([-13, 45, 30, 70])
#         ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
#     else:
#         map_stations.legend = ax.get_legend_handles_labels()
#     ax.axis('off')
    
#     if 'title' in kwargs:
#         ax.set_title(kwargs['title'])
        
#     if save is not None:
#         plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
# def map_stations(x, y, z, mask=None, rivers=None, ax=None, save=None, **kwargs):
#     """It plots a map of Europe with the reporting points and their number of flood events
    
#     Inputs:
#     -------
#     x:        pandas.Series (stations,). Coordinate X of the stations
#     y:        pandas.Series (stations,). Coordinate Y of the stations
#     z:        pandas.Series (stations,). Number of flood events identified in each station
#     mask:     pandas.Series (stations,). A boolean series of stations to be plotted differently, i.e., not included in the colour scale based on 'z'
#     rivers:   geopandas. Shapefile of rivers
#     ax:       matplotlib.axis. Axis in which the plot will be embedded. If None, a new figure will be created
#     save:     string. A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
#     Ouput:
#     ------
#     The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
#     """
    
#     # define projection
#     proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10, central_latitude=52, false_easting=4321000, false_northing=3210000, globe=ccrs.Globe(ellipse='GRS80'))
#     if ax is None:
#         fig = plt.figure(figsize=kwargs.get('figsize', None))
#         ax = plt.axes(projection=proj)
    
#     # add polygon of land
#     ax.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor=None, facecolor='gray'), alpha=.5, zorder=0)
#     # # plot coatslines and country borders
#     # ax.add_feature(cf.COASTLINE, lw=.7, zorder=0)
#     # ax.add_feature(cf.BORDERS, lw=.7, ls='--', color='k', zorder=0)
    
#     # plot rivers
#     if rivers is not None:
#         rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='gray', ax=ax, zorder=0)
    
#     # plot all the stations
#     if mask is not None:
#         # plot masked stations
#         ax.scatter(x[mask], y[mask], s=kwargs.get('size', 1) / 8, c='dimgray', alpha=kwargs.get('alpha', .5),
#                    label='stations w/o events', zorder=0)
#         x = x[~mask]
#         y = y[~mask]
#         z = z[~mask]

#     # plot non-masked stations
#     sct = ax.scatter(x, y, c=z, s=kwargs.get('size', 1), cmap=kwargs.get('cmap', 'viridis'), norm=kwargs.get('norm', None),
#                     alpha=kwargs.get('alpha', .5))#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
#     map_stations.colorbar = sct
    
#     # settings
#     if ax is None:
#         plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
#         plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
#         # ax.set_extent([-13, 45, 30, 70])
#         ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
#     else:
#         map_stations.legend = ax.get_legend_handles_labels()
#     ax.axis('off')
    
#     if 'title' in kwargs:
#         ax.set_title(kwargs['title'])
        
#     if save is not None:
#         plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
    
def map_stations(x: pd.Series, y: pd.Series, z: pd.Series, theta: pd.Series = None, mask: pd.Series = None, rivers: gpd.GeoDataFrame = None, ax=None, save: Union[str, Path] = None, **kwargs):
    """It plots a map of Europe with the reporting points and their number of flood events
    
    Inputs:
    -------
    x:        pandas.Series (stations,)
        Coordinate X of the stations
    y:        pandas.Series (stations,)
        Coordinate Y of the stations
    z:        pandas.Series (stations,)
        Values of the variable to be plotted
    theta:    pandas.Series (stations,)
        If provided, the marker of the scatter plot will be arrows. Theta represents the angle (in radians) of the arrow
    mask:     pandas.Series (stations,)
        A boolean series of stations to be plotted differently, i.e., not included in the colour scale based on 'z'
    rivers:   geopandas
        Shapefile of rivers
    ax:       matplotlib.axis
        Axis in which the plot will be embedded. If None, a new figure will be created
    save:     string
        A string with the file name (including extension) where the plot will be saved. If None, the plot is not saved
    
    Kwargs:
    -------
    alpha: float
        Transparency of the points
    extent: List
        Extension of the map [xmin, xmax, ymin, ymax]
    cmap:
        Colour map
    figsize: Tuple
        size of the figure
    headaxislength: float
        If arrows are plotted, the length of the arrow head axis
    headwidth: float
        If arrows are pltoted, the width of the arrow head
    norm:
        Normalization of the colour map
    scale: int
        Size of the arrows (in case 'theta' is not None)
    size: float
        Size of the points
    width: float
        If arrows are plotted, the widht of the arrow line
    
    Ouput:
    ------
    The plot is printed in the screen, and if 'save' is provided, it saves the figure as a PNG file
    """
    
    alpha = kwargs.get('alpha', 1)
    extent = kwargs.get('extent', [-10, 44, 28, 69])#[-13, 45, 30, 70])
    cmap = kwargs.get('cmap', 'viridis')
    figsize = kwargs.get('figsize', None)
    hw = kwargs.get('headwidht', 4)
    hal = kwargs.get('headaxislength', hw * 1.15)
    hl = 1.5 * hw
    norm = kwargs.get('norm', None)
    scale = kwargs.get('scale', 60)
    s = kwargs.get('size', 1)
    w = kwargs.get('width', .00175)
    
    # define projection
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10,
                                          central_latitude=52,
                                          false_easting=4321000,
                                          false_northing=3210000,
                                          globe=ccrs.Globe(ellipse='GRS80'))
    if ax is None:
        # cbar = True
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        
    # add polygon of land
    ax.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor=None, facecolor='whitesmoke'), alpha=.5, zorder=0)
    # plot coatslines and country borders
    ax.add_feature(cf.COASTLINE, lw=.5, color='darkgray', zorder=0)
    ax.add_feature(cf.BORDERS, lw=.5, ls=':', color='darkgray', zorder=0)
    ax.set_extent(extent)
    
    # plot rivers
    if rivers is not None:
        rivers.to_crs(crs='epsg:3035').plot(lw=kwargs.get('lw', .5), color='lightsteelblue', ax=ax, zorder=0)
    
    # plot all the stations
    if mask is not None:
        # plot masked stations
        ax.scatter(x[mask], y[mask], marker='.', s=s / 5, c='dimgray', alpha=alpha, label='stations w/o events', zorder=0)
        x = x[~mask]
        y = y[~mask]
        z = z[~mask]
        if theta is not None:
            theta = theta[~mask]

    # plot non-masked stations
    if theta is None:
        sct = ax.scatter(x, y, c=z, s=s, cmap=cmap, norm=norm, alpha=alpha)#, vmin=kwargs.get('vmin', 1), vmax=kwargs.get('vmax', max(z.max(), 2)))
    elif isinstance(theta, pd.Series):
        sct = ax.quiver(x, y, np.cos(theta), np.sin(theta),
                        z, cmap=cmap, norm=norm, alpha=alpha,
                        scale=scale,
                        width=w,
                        headwidth=hw,
                        headlength=hl,
                        headaxislength=hal)
    map_stations.colorbar = sct
    
    # settings
    # if cbar: # when ax is None
    #     plt.colorbar(sct, location='bottom', shrink=.4, label='no. events')
    #     plt.gcf().set_size_inches(kwargs.get('figsize', (8, 8)))
    #     ax.legend(bbox_to_anchor=[.2, -.2, .5, .1]);
    # else:
    #     map_stations.legend = ax.get_legend_handles_labels()
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
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10,
                                              central_latitude=52, 
                                              false_easting=4321000, 
                                              false_northing=3210000,
                                              globe=ccrs.Globe(ellipse='GRS80'))
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
        
        
        
def map_events(x: pd.Series, y: pd.Series, events: pd.Series, rivers: gpd.GeoDataFrame = None, save: Union[str, Path] = None, **kwargs):
    """It plots a map and a histogram of the number of events.
    
    Inputs:
    -------
    x: pandas.Series (stations,)
        Coordinate X of the stations
    y: pandas.Series (stations,)
        Coordinate Y of the stations
    z: pandas.Series (stations,)
        Number of flood events identified in each station
    rivers:   geopandas
        Shapefile of rivers
    save:       string
        Directory where to save the plot as a JPG file. If None (default), the plot won't be saved
        
    Kwargs:
    -------
    alpha: float
        Transparency of the points
    extent: List
        Extension of the map [xmin, xmax, ymin, ymax]
    cmap:
        Colour map
    figsize: Tuple
        size of the figure
    norm:
        Normalization of the colour map
    size: float
        Size of the points
    """
    
    # extract kwargs
    alpha = kwargs.get('alpha', 1)
    extent = kwargs.get('extent', [-10, 44, 28, 69])#[-13, 45, 30, 70])
    if ('cmap' not in kwargs) or ('norm' not in kwargs):
        xmax = events.max()
        Or = ListedColormap(mpl.cm.get_cmap('Oranges', 128)(np.linspace(.15, 1., 128)), 'blues')
        cmap, norm = create_cmap(Or, np.arange(xmax + 2), 'no. events', [0, 'dimgray'])#(0.41176, 0.41176, 0.41176, 1)])
    else:
        cmap = kwargs['cmap']
        norm = kwargs['norm']
    s = kwargs.get('size', 2)
    yscale = kwargs.get('yscale', 'linear')
    
    # set up the plots
    fig = plt.figure(figsize=kwargs.get('figsize', (7, 7)), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, height_ratios=kwargs.get('height_ratios', [7, 1]))

    # map
    if 'proj' not in kwargs: # define projection
        proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10,
                                              central_latitude=52,
                                              false_easting=4321000,
                                              false_northing=3210000,
                                              globe=ccrs.Globe(ellipse='GRS80'))
    ax_map = fig.add_subplot(gs[0], projection=proj)
    map_stations(x, y, events, #mask=events == 0,
                 rivers=rivers, cmap=cmap, norm=norm, size=s, alpha=alpha, ax=ax_map, extent=extent)

    # histogram
    ax_hist = fig.add_subplot(gs[1])
    counts = events.value_counts()
    counts.sort_index(inplace=True)
    color = [cmap(i) for i in np.linspace(0, cmap.N, norm.N).astype(int)]
    plt.bar(counts.index, counts, width=1, alpha=.66, color=color)
    ax_hist.set(xlabel='no. observed events',
                xlim=(norm.boundaries.min() - .5, norm.boundaries.max() - .5),
                xticks=norm.boundaries[:-1],
                ylabel='no. points')
    ax_hist.set_axisbelow(True)
    ax_hist.grid(axis='y', zorder=0, ls=':', lw=0.5)
    ax_hist.spines[['right', 'top']].set_visible(False)
    if yscale != 'linear':
        ax_hist.set_yscale(yscale)
        yticks = [int(y) for y in ax_hist.get_yticks() if (y >= 1) & (y <= 1000)]
        ax_hist.set_yticks(yticks, labels=yticks)
    
    if 'title' in kwargs:
        fig.text(.5, 1, kwargs['title'], horizontalalignment='center', verticalalignment='top', fontsize=13)
        
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def gauge_legend(theta_min: float = 20, ax=None, **kwargs):
    """It creates a legend that is a gauge charg.
    
    Input:
    ------
    theta_min: float
        Minimum angle (in degrees) in the gauge chart. This angle will represent the best value of the variable
    ax:       matplotlib.axis
        Axis in which the plot will be embedded. If None, a new figure will be created
        
    Keywords:
    ---------
    lw: float
        Line width
    r:  float
        Radius of the gauge chart
    width: float
        Width of the arrow
    headwidth: float
        Width of the head of the arrow
    headaxislength: float
        Lenght of the axis of the arrow head
    scale: float
        Size of the arrow
    """
    
    theta_max = 180 - theta_min
    c = 'dimgray'
    lw = kwargs.get('lw', .2)
    r = kwargs.get('r', 1) # radius of the circle
    w = kwargs.get('width', .005)
    hw = kwargs.get('headwidth', 6)
    hal = kwargs.get('headaxislength', hw*1.2)
    scale = kwargs.get('scale', 2.2)
    
    if ax is None:
        fig, ax = plt.subplots()

    # plot the arch
    theta = np.linspace(np.deg2rad(theta_min), np.deg2rad(theta_max), 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, c, lw=lw*2)
    
    # plot the arch limits
    ax.plot([0, x[0]], [0, y[0]], c, lw=lw)
    ax.plot([0, x[-1]], [0, y[-1]], c, lw=lw)

    # ticks and example arrow
    labels = {theta_min: 1,
              55: .5,
              90: 0,
              125: -.5,
              theta_max: -1}
    for i, (angle, label) in enumerate(labels.items()):
        mult = np.array([.95, 1, 1.05, 1.3])
        x_line = mult * r * np.cos(np.deg2rad(angle))
        y_line = mult * r * np.sin(np.deg2rad(angle))
        ax.plot(x_line[[0, 2]], y_line[[0, 2]], color='k', linewidth=1)
        ax.text(x_line[3], y_line[3], label, ha='center', va='center')
        if i == 1:
            ax.quiver(0, 0, x_line[1], y_line[1], scale=scale, width=w, headwidth=hw, headlength=hw*1.5, headaxislength=hal)
    
    # setting
    ax.set_aspect('equal')
    ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)
    ax.set(xticks=[],
           yticks=[])
    if 'title' in kwargs:
        ax.text(0, -.3, kwargs['title'], ha='center', va='center')