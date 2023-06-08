import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns



def plot_events_timeseries(discharge, events1=None, events2=None, thresholds=None, save=None, **kwargs):
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
    
    if events1 is not None:
        start = max(discharge.index.min(), events1.index.min())
        end = min(discharge.index.max(), events1.index.max())
        discharge = discharge.loc[start:end]
        events1 = events1.loc[start:end]
        if events2 is not None:
            events2 = events2.loc[start:end]
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (16, 3)))
    
    # plot discharge timeseries
    ax.plot(discharge, lw=.7, zorder=0)
    
    # plot points of preliminary events
    if events2 is not None:
        ax.scatter(discharge[events2].index, discharge[events2], s=kwargs.get('size', 2), color='k')
        ax.text(.005, .85, 'no. preliminary events: {0}'.format(events2.sum()), transform=ax.transAxes, fontsize=9)
        
    # plot points of the events
    if events1 is not None:
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