import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
        
        

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