import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from optimize import find_best_criterion
from compute import exceedance2events, buffer_events
from plot.maps import create_cmap
from collections.abc import Iterable
from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional, Literal



def plot_correlation_matrix(
    corr: pd.DataFrame,
    rho: float = .9,
    save: Union[str, Path] = None,
    **kwargs
) -> None:
    """
    It creates a heat map that shows the correlation matrix and highlights the cases in which the correlation coefficient exceeds a certain value
    
    Parameters:
    -------
    corr: pd.DataFrame
        Correlation matrix
    rho: float
        The maximum value allowed for the correlation coefficient between two reporting points
    save: str or Path
        Directory where to save the plot as a JPG file. If None (default), the plot won't be saved 
    """
    
    assert 0 < rho < 1, 'ERROR. "rho" must be a float between 0 and 1'
    
    if ('cmap' not in kwargs) or ('norm' not in kwargs):
        cmap, norm = create_cmap('Blues', np.arange(0, 1.01, .05), 'correlation coefficient')
    else:
        cmap = kwargs['cmap']
        norm = kwargs['norm']

    # compute exceedance of the correlation threshold
    highly_correlated = corr > rho
    highly_correlated = highly_correlated.astype(int)
    highly_correlated[highly_correlated == 0] = np.nan

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7, 7)))
    sns.heatmap(corr, vmin=0, vmax=1, ax=ax, cmap=cmap, norm=norm, cbar_kws={'label': 'Spearman correlation (-)', 'shrink': .5});
    sns.heatmap(highly_correlated, ax=ax, cmap='Oranges', vmin=0.5, vmax=1.5, alpha=1, cbar=None)
    ax.set_aspect('equal')
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight', dpi=300);
        
           
        
def graphic_explanation(
    obs: xr.DataArray, 
    pred: xr.DataArray, 
    id: int, 
    model: Literal['current', 'model_mean', 'weighted_mean'], 
    window: int, 
    probability: float, 
    forecast=False, 
    verbose=True, 
    **kwargs
) -> None:
    """
    It creates a graph with six plots that explain how hits, misses and false alarms are computed.
    
    Parameters:
    -------
    obs:         xarray.DataArray
        Boolean matrix of observed events
    pred:        xarray.DataArray
        Boolean matrix of predicted events
    id:          int
        Station ID
    model:       string
        Criteria: 'current', 'model_mean', 'weighted_mean'
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
        
        
        
def plot_DataArray(
    da: xr.DataArray, 
    ax: Optional[mpl.axes.Axes] = None, 
    **kwargs
) -> None:
    """
    It creates a heatmap plot of a 2D DataArray
    
    Input:
    ------
    da:   xr.DataArray
    ax:   matplotlib.axes.Axes, optional
    
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    # extract kwargs
    figsize = kwargs.get('figsize', (16, 2))
    xtick_step = kwargs.get('xtick_step', 1)
    ytick_step = kwargs.get('ytick_step', 1)
    cmap = kwargs.get('cmap', 'magma')
    alpha = kwargs.get('alpha', 1)
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
    hm = sns.heatmap(da_plot, cmap=cmap, norm=norm, ax=ax, vmin=vmin, vmax=vmax, cbar=cbar, cbar_kws=cbar_kws, alpha=alpha)
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
        
        
        
def lineplot_hits(
    ds: xr.Dataset,
    xdim: str = 'probability', 
    coldim: str = 'persistence', 
    rowdim: Optional[str] = None, 
    linedim: str = 'approach',
    beta: float = 1, 
    yscale: Literal['linear', 'log', 'semilog'] = 'linear', 
    xtick_step: int = 4,
    save: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    It creates several lineplots of hits (TP), misses (FN) and false alarms (FP).
    
    Depending on the dimensions of the analysis, a plot will be created for each combination of 'rowdim' and 'coldim'.
    
    Parameters:
    -----------
    ds: xr.Dataset
        It contains three DataArrays with names 'TP' (hits), 'FN' (misses) and 'FP' (false alarms). Its dimensions must be those defined in `xdim`, `coldim`, `rowdim` and `linedim
    xdim: str
        Dimension in `ds` that will be represented in the X axis of the plots
    coldim: str
        Dimension in `ds` that will be used to create several columns of plots
    rowdim: str, optional
        Dimension in `ds` that will be used to create several rows of plots. If None, there will be only one single line of plots
    yscale:     string. Type of scaling used in the Y axis, e.g., 'linear' or 'log'
    xtick_step: int
        Frequency of the labels in the X axis
    save: str or Path
        Directory where a JPG file of the plot will be saved
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
        
        

def lineplot_skill(
    ds: xr.Dataset,
    metric: str = 'f1',
    xdim: str = 'probability',
    rowdim: str = 'persistence',
    coldim: Optional[str] = None,
    linedim: str = 'approach',
    save: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    It creates a lineplot with the results of the eventwise skill analysis. A series of plots will be created. If 'coldim' is None,  the columns represent the metrics (variables of the Dataset 'ds'). If 'coldim' is not None, columns represent the dimension specified and the different metrics (variables in the Dataset 'ds') are represented by line colours.
    
    Parameters:
    -------
    ds: xr.Dataset
        It contains the arrays of skill for several metrics. At least, it should have the variables  for the chosen target metric (see attribute "metric"), 'recall' and 'precision'.
    metric: str
        Name of the skill metric for which the criteria will be optimize. This name should be one of the variables in the Dataset 'ds'. By default, f1
    xdim: str
        It defines the dimension in 'ds' that will correspond to the X axis in the plots
    rowdim: str
        It defines the dimension in 'ds' that will correspond to the rows in which the graph will be divided
    coldim: str, optional
        It defines the dimension in 'ds' that will correspond to the cols in which the graph will be divided. If None (default), each column will represent a different skill metric (variables in 'ds')
    linedim: str
        It defines the dimension in 'ds' that will correspond to the different lines in the plots
    save: str or Path
        Directory and filename (including extension) where the graph will be saved
    
    Output:
    -------
    None
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
        best_x = find_best_criterion(ds, dim=xdim, metric=metric, tolerance=.01, min_spread=False)[xdim]
        
        ncols = len(ds)
        nrows = len(ds[rowdim])
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)
        colors = kwargs.get('color', {'current': 'steelblue', '1_deterministic_+_1_probabilistic': 'steelblue', 'model_mean': 'lightsteelblue', 'member_weighted': 'C1', 'brier_weighted': 'navajowhite'})
        
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
        
        
        
def plot_hits_by_variable(
    hits: xr.Dataset, 
    optimal_criteria: Dict, 
    variable: str, 
    coldim: str = 'approach', 
    reference: Optional[Union[int, float]] = None, 
    current_criteria: Optional[Dict] = None, 
    optimized_criteria: Optional[xr.DataArray] = None, 
    save: Optional[Union[Path, str]] = None, 
    **kwargs
) -> None:
    """It generates a graph with as many lineplots as approaches in the 'hits' dataset. The lineplots reprensent both the evolution of true positives (hits) and false positives (false alarms) and probability with regard to a specified variable
    
    Parameters:
    -----------
    hits: xr.Dataset
        It contains as variables TP (true positives), FN (false negatives) and FP (false positives)
    optimal_criteria: Dict
        For each approach in 'hits', it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable: str
        Name of the variable in 'hits' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    coldim: str
        Name of the dimension that defines each of the plots in the graph
    reference: int or float, optional
        Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    current_criteria: Dict, optional
        It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    optimized_criteria: xr.DataArray (variable, approach, persistence)
        It contains the optimized probability threshold for each combination of the 'variable', approach and persistence
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    colors = kwargs.get('colors', ['k', 'steelblue', 'orange'])
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .15)
    loc_text = kwargs.get('loc_text', 1)
    
    ncols = len(hits[coldim])
    fig, axes = plt.subplots(ncols=ncols, figsize=kwargs.get('figsize', (4.5 * ncols, 4)), sharex=True, sharey=True)
    
    if current_criteria is not None:
        obs_current = hits.sel(current_criteria)['TP'] + hits.sel(current_criteria)['FN']
        pred_current = hits.sel(current_criteria)['TP'] + hits.sel(current_criteria)['FP']
    
    # for ax1, (key, criteria) in zip(axes, optimal_criteria.items()):
    for ax1, key in zip(axes, hits[coldim].data):
        
        if key not in optimal_criteria:
            ax1.set(title=key.replace('_', ' '),
                    xlabel=kwargs.get('xlabel', variable))
            if ax1 == axes[0]:
                ax1.text(-.15, .5, r'$\frac{x}{obs}$', horizontalalignment='right', transform=ax1.transAxes)
            continue
            
        # HITS, FALSE POSITIVES
        # ---------------------
        
        sel = {coldim: key}
        ds = hits.sel(sel)
        
        # hits/false alarms for the current operational criteria
        if current_criteria is not None:
            pred = pred_current / obs_current
            tp = hits.sel(current_criteria)['TP'] / obs_current
            ax1.fill_between(hits[variable], tp, pred, color=colors[0], alpha=alpha * .5, zorder=0, label='FP (current)')
            ax1.plot(hits[variable], tp, c=colors[0], label=f'TP (current)', lw=lw, ls='-', zorder=3)
        
        # hits/false alarms for the optimized criteria
        criteria = optimal_criteria[key]
        obs = hits.sel(criteria)['TP'] + hits.sel(criteria)['FN']
        pred = hits.sel(criteria)['TP'] + hits.sel(criteria)['FP']
        ax1.fill_between(hits[variable], hits.sel(criteria)['TP'] / obs, pred / obs, color=colors[1], alpha=alpha, zorder=1, label='FP (optimal)')
        ax1.plot(hits[variable], hits.sel(criteria)['TP'] / obs, c=colors[1], label=f'TP (optimal)', lw=lw, zorder=5)
        persistence = optimal_criteria[key]['persistence']
        
        # hits/false alarms optimized for each value of the target variable
        if optimized_criteria is not None:
            probability = optimized_criteria.sel(sel).sel(persistence=persistence)
            hits_var = pd.DataFrame(index=probability[variable].data, columns=list(hits))
            for v, p in zip(hits_var.index, probability.data):
                if np.isnan(p): # due to persistence
                    continue
                hits_var.loc[v] = hits.sel({variable: v, 'probability': p, coldim: key, 'persistence': persistence}).to_pandas()
            obs = (hits_var['TP'] + hits_var['FN']).replace(0, np.nan)
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
        ax1.set_title(key.replace('_', ' '))
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
            prob = optimal_criteria[key]['probability']           
            xmin = hits[variable].data.min()
            xmax = hits['TP'].sel({coldim: key, 'persistence': persistence, 'probability': prob}).to_pandas().last_valid_index()
            ax2.hlines(prob, xmin, xmax, color=colors[1], lw=lw, ls=':', zorder=4, label='prob. (optimal)')
            
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
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.9, .8, .2, .1]), frameon=False)
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_skill_by_variable(
    skill: xr.Dataset, 
    optimal_criteria: Dict, 
    variable: str, 
    coldim: str = 'approach', 
    reference: Optional[Union[int, float]] = None, 
    metric: str = 'f1', 
    current_criteria: Optional[Dict] = None, 
    optimized_criteria: Optional[xr.Dataset] = None,
    shades: bool = True,
    save: Optional[Union[Path, str]] = None,
    **kwargs
) -> None:
    """
    It generates a graph with as many lineplots as approaches in the 'skill' dataset. The lineplots reprensent both the evolution of skill and probability with regard to a specified variable
    
    Parameters:
    -----------
    skill: xr.Dataset
        It contains as variables recall, precision and the specified metric
    optimal_criteria: Dict
        For each approach in skill, it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable: str
        Name of the variable in 'skill' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    coldim: str
        Name of the dimension that defines each of the plots in the graph
    reference: int or float, optional
        Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric: str
        Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria: dict, optional
        It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    optimized_criteria: xr.Dataset, optional
        It contains as variables probability, recall, precision and the specified metric  
    shades: bool
        If True, a shaded shape shows the difference bewteen recall and precision
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """

    colors = kwargs.get('color', ['k', 'steelblue', 'orange'])
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .15)
    loc_text = kwargs.get('loc_text', 1)
      
    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True, figsize=kwargs.get('figsize', (4.5 * ncols, 4)))
    
    if current_criteria is not None:
        skill_current = skill.sel(current_criteria).to_pandas().drop(['probability', coldim, 'persistence'], axis=1)
        y1_current = skill_current[['recall', 'precision']].min(axis=1)
        y2_current = skill_current[['recall', 'precision']].max(axis=1)

    for ax1, key in zip(axes, skill[coldim].data):
        
        if key not in optimal_criteria:
            ax1.set(title=key.replace('_', ' '),
                    xlabel=kwargs.get('xlabel', variable))
            if ax1 == axes[0]:
                ax1.set_ylabel('skill')
            continue
        
        # SKILL
        # -----
        
        # skill for the current operational criteria
        if current_criteria is not None:
            if shades:
                ax1.fill_between(skill_current.index, y1_current, y2_current, alpha=alpha * .5, color=colors[0], zorder=0, label=f'P-R (current)')
            ax1.plot(skill_current.index, skill_current[metric], c=colors[0], lw=lw, label=f'{metric} (current)', zorder=6)

        # skill for the optimal criteria (that optimized for the reference value of the variable)
        skill_optimal = skill.sel(optimal_criteria[key]).to_pandas().drop(['probability', coldim, 'persistence'], axis=1)
        if shades:
            y1_optimal = skill_optimal[['recall', 'precision']].min(axis=1)
            y2_optimal = skill_optimal[['recall', 'precision']].max(axis=1)
            ax1.fill_between(skill_optimal.index, y1_optimal, y2_optimal, alpha=alpha, color=colors[1], zorder=1, label=f'P-R (fixed)')
        ax1.plot(skill_optimal.index, skill_optimal[metric], c=colors[1], lw=lw, label=f'{metric} (fixed)', zorder=7)
        persistence = optimal_criteria[key]['persistence']
        
        # skill optimized for each value of the target variable
        if optimized_criteria is not None:
            probability = optimized_criteria.sel({coldim: key}).sel(persistence=persistence).data
            skill_var = pd.DataFrame(index=optimized_criteria[variable].data, columns=list(skill))
            for v, p in zip(skill_var.index, probability):
                if np.isnan(p): # due to persistence
                    continue
                skill_var.loc[v] = skill.sel({variable: v, 'probability': p, coldim: key, 'persistence': persistence}).to_pandas()
            if shades:
                y1_var = skill_var[['recall', 'precision']].min(axis=1)
                y2_var = skill_var[['recall', 'precision']].max(axis=1)
                ax1.fill_between(skill_var.index, y1_var, y2_var, alpha=alpha, color=colors[2], zorder=2, label=f'P-R ({variable}-specific)')
            ax1.plot(skill_var.index, skill_var[metric], c=colors[2], lw=lw, label=f'{metric} ({variable}-specific)', zorder=8)

        # reference line
        if reference is not None:
            ax1.axvline(x=reference, ls=':', lw=.5, color='k')

        # settings
        ax1.set_title(key.replace('_', ' '))
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
        else:
            color = 'k'
        ax1.text(x, y, f'persistence = {persistence}', color=color, horizontalalignment=ha, verticalalignment=va, transform=ax1.transAxes,
                 backgroundcolor='w')
        ax1.set(xlabel=kwargs.get('xlabel', variable),
                xlim=kwargs.get('xlim', (skill[variable].min(), skill[variable].max())),
                xscale=kwargs.get('xscale', 'linear'),
                ylim=(-.025, 1.025))
        # if ax1 == axes[0]:
        #     ax1.set_ylabel('skill')
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
            ax2.axhline(prob, color=colors[0], lw=lw, ls='--', zorder=3, label='prob. (current)')

        # probability of the optimal criteria
        if optimal_criteria is not None:
            prob = optimal_criteria[key]['probability']
            xmin = skill[variable].data.min()
            xmax = skill[metric].sel({coldim: key, 'persistence': persistence, 'probability': prob}).to_pandas().last_valid_index()
            ax2.hlines(prob, xmin, xmax, color=colors[1], lw=lw, ls='--', zorder=4, label='prob. (fixed)')

        # probability optimized for each value of the target variable
        if optimized_criteria is not None:
            ax2.plot(optimized_criteria[variable].data, probability,
                     c=colors[2], lw=lw, ls='--', zorder=5, label=f'prob. ({variable}-specific)')

        # settings
        ax2.set(ylim=(-.025, 1.025))
        # if ax1 == axes[-1]:
        #     ax2.set_ylabel('probability')
        # else:
        ax2.set_yticklabels([])
            
    # legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles, labels = [], []
    for i in range(len(labels2)):
        handles += handles1[i * 2:i * 2 + 2] + [handles2[i]]
        labels += labels1[i * 2:i * 2 + 2] + [labels2[i]]
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.945, .8, .2, .1]), frameon=False)
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_skill_training(
    train: xr.Dataset, 
    test: xr.Dataset, 
    complete: Optional[xr.Dataset] = None, 
    xdim: str = 'approach', 
    save: Optional[Union[str, Path]] = None, 
    **kwargs
) -> None:
    """
    Scatter (and box) plot of the performance achieved for every approach in the train, test and complete data sets.
    
    Parameters:
    -----------
    train: xr.Dataset
        The skill in the training set.The 'kfold' dimension is not mandatory, it would contain the skill in any of the folds of the cross-validation
    test: xr.Dataset
        The skill in the test set
    complete: xr.Dataset, optional
        The skill in the complete set (training + test)
    xdim: str
        Name of the dimension in 'train', 'test' and 'complete' to plot on the X axis
    save: str or pathlib.Path, optional
        Directory and filename (including extension) where the graph will be saved
        
    Returns:
    --------
    None
    """
    
    # kwargs
    figsize = kwargs.get('figsize', (4, 3.7))
    ylim = kwargs.get('ylim', (-.02, 1.02))
    
    # plot performance
    ncols = len(train)
    fig, axes = plt.subplots(ncols=ncols, figsize=(figsize[0] * ncols, figsize[1]), sharex=True, sharey=True)
    xticks = np.arange(1, len(train[xdim].data) + 1)
    for ax, score in zip(axes, list(test)):
        if 'kfold' in train.dims:
            ax.boxplot(train[score].transpose(), zorder=0)
        else:
            ax.scatter(xticks, train[score].data, c='k', zorder=1, label='train')
        ax.scatter(xticks, test[score].data, c='C1', zorder=3, label='test')
        if complete is not None:
            ax.scatter(xticks, complete[score].data, c='steelblue', zorder=2, label='all')
        ax.set_title(score)
        if ax == axes[0]:
            ax.set_ylabel('skill')
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    xlabels = []
    for label in list(train[xdim].data):
        if len(label) > 3:
            xlabels.append(''.join([x[0].upper() for x in label.split('_')]))
        else:
            xlabels.append(label)
    ax.set_xticklabels(xlabels)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=kwargs.get('loc_legend', [.8, .8, .2, .1]), frameon=False)
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        


def plot_prediction(da, obs, probability, persistence=(1, 1), min_leadtime='all', center=True, w=5, save=None, **kwargs):
    """It creates a plot with several heat maps that explain the process applied to compute hits, misses and false alarms.
    
    1. On top it shows a heat map of total probability (values from 0 to 1) of exceeding the discharge threshold. 
    
    2. The second plot is a binary heat map of exceedances over the probability threshold.
    
    3. The third plot shows the time steps for which the persistence criterion is met.
    
    4. The fourth plot shows a buffer over the 3rd plot.
    
    5. The fifth plot shows a binary heat map of exceedances over the discharge threshold in the observed data.
    
    Parameters:
    -------
    da:           xr.DataArray (leadtime, datetime). A matrix of total probability of exceeding the discharge threshold
    obs:          xr.DataArray (datetime,). Binary matrix of exceedances over the discharge threshold in the observed data
    probability:  float. A probability threshold that will be applied on the total probability of exceeding the discharge threshold
    persistence:  list (2,). Persistence criterion consisting on the number of positive predictions out of a number of forecasts. For instance, a persistence of (2, 3) requires 2 positive prediction out of 3 forecasts
    min_leadtime: string or int. The minimum value of lead time (in hours) from which notifications can be sent. If 'all', all lead times are taken into account
    center:       boolean. Whether the rolling window used to buffer the predicted events should be centered or not
    w:            int. Width of the rolling window used to buffer the predicted events
    save:         string. File where the plot should be saved. If 'None', the plot won't be saved
    """
    
    cmap = kwargs.get('cmap', 'magma_r')
    norm = kwargs.get('norm', None)
    
    fig = plt.figure(figsize=kwargs.get('figsize', (16, 6.6)), constrained_layout=True)
    height_ratios = [len(da.leadtime)] * 2 + [1] * 3
    gs = fig.add_gridspec(nrows=len(height_ratios), height_ratios=height_ratios)
    
    # total probability
    plot_DataArray(da, ytick_step=2, cbar=False, cbar_kws={'label': 'probability'}, title='total probability',
                   xticklabels=[], xlabel=None, ylabel='leadtime (h)', cmap=cmap, norm=norm, ax=fig.add_subplot(gs[0]))
    
    # exceedance over probability threshold
    plot_DataArray(da > probability, ytick_step=2, cbar=True, cbar_kws={'label': 'probability'}, title='exceedance', xlabel=None,
                   xticklabels=[], ylabel='leadtime (h)', cmap=cmap, norm=norm, ax=fig.add_subplot(gs[1]))
    
    # predicted events
    pred = exceedance2events(da, probability=probability, persistence=persistence, min_leadtime=min_leadtime)
    plot_prediction.pred = pred
    plot_DataArray(pred, title='predicted', xticklabels=[], cmap=cmap, norm=norm, cbar=False, ax=fig.add_subplot(gs[2]))
    
    # buffered, predicted events
    buff = buffer_events(pred, center=center, w=w)
    plot_prediction.buff = buff
    plot_DataArray(buff, title='buffered', xticklabels=[], cmap=cmap, norm=norm, cbar=False, ax=fig.add_subplot(gs[3]))

    # observed events
    plot_DataArray(obs, xlabel='datetime', xtick_step=4, title='observed', cmap=cmap, norm=norm, cbar=False, ax=fig.add_subplot(gs[4]))
    
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=13);
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_skill_by_probability(
    skill: xr.Dataset, 
    probability: Union[List, np.array], 
    persistence: str = '1/1', 
    coldim: str = 'approach', 
    ref_prob: Optional[Union[int, float]] = None, 
    metric: str = 'f1', 
    benchmark: xr.Dataset = None, 
    save: Union[Path, str] = None, 
    **kwargs
) -> None:
    """
    It generates a graph with as many line plots as approaches in the 'skill' dataset. The line plots reprensent the evolution of skill depending on the probability threshold
    
    Parameters:
    -----------
    skill: xr.Dataset
        It contains as variables recall, precision and the specified metric
    probability: list or np.array
        List of probability thresholds to be plotted
    persistence: str
        Fixed value of persistence
    coldim: str
        Name of the dimension that defines each of the plots in the graph
    ref_prob: int or float, optional
        Value of probability used as reference
    metric: str
        Name of the target metric. This metric should be a variable in both datasets 'skill' and 'benchmark'
    benchmark: xr.Dataset, optional
        Skill of the a benchmark set of criteria
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Other parameters:
    -----------------
    alpha: float
        Transparency of the lines
    cmap: Union[str, mpl.colors.Colormap]
        Colour map used to plot the different lines in 'linedim'
    figsize: tuple, optional
        Size of the figure
    label: str
        Name given to the benchmark in the legend
    loc_leged: Tuple[float, float, float, float]
        Location of the legend (xmin, ymin, width, height)
    lw: float
        Width of the lines
    offset: int
        Number of hours used to convert the initial lead time values into days
    xlabel: str, optional
        Label of the X axis
    xlim: Tuple[float, float]
        Limits of the X axis
    ylim: Tuple[float, float]
        Limites of the Y axis
    
    Ouput:
    ------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    # extract kwargs
    alpha = kwargs.get('alpha', .666)
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm_r'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(probability)))).colors
    label = kwargs.get('label', 'current')
    lw = kwargs.get('lw', 1.2)
    offset = kwargs.get('offset', -12) 
    xlabel = kwargs.get('xlabel', 'lead time ≥ (d)')
    
    # set up the figure
    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True, figsize=kwargs.get('figsize', (4.5 * ncols, 4)))
    axes[0].set(xlim=kwargs.get('xlim', (1, 10)),
                ylim=kwargs.get('ylim', (-.02, 1.02)))
    
    if benchmark is not None:
        df_bm = benchmark.to_pandas()
        df_bm.index = (df_bm.index + offset) / 24
        
    for ax, key in zip(axes, skill[coldim].data):
        if benchmark is not None:
            ax.plot(df_bm.index, df_bm[metric], c='k', lw=lw * 1.2, label=label, zorder=25)
        
        for p, c in zip(probability, colors):
            criteria = {coldim: key, 'probability': p, 'persistence': persistence}
            df = skill.sel(criteria).to_pandas()
            df.index = (df.index + offset) / 24
            if p == ref_prob:
                ax.plot(df.index, df[metric], c='k', ls='--', lw=lw, alpha=1, label=f'P ≥ {p:.2f}', zorder=24)
            else:
                ax.plot(df.index, df[metric], c=c, lw=lw, alpha=alpha, label=f'P ≥ {p:.2f}')
                       
        # return df
        ax.axvline(df.index[2], c='k', ls=':', lw=lw / 2)
        ax.axvline(df.index[5], c='k', ls=':', lw=lw / 2)
        ax.axvline(df.index[6], c='k', ls=':', lw=lw / 2)
        ax.set_xlabel(xlabel)
        if ax == axes[0]:
            ax.set_ylabel(f'{metric} (-)')
            ax.text(df.index[2], .9999, 'start warnings', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
            ax.text(df.index[5], .9999, 'end COS', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
            ax.text(df.index[6], .9999, 'end DWD', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        ax.set_title(key.replace('_', ' '))
        
    ax.set_xticks(df.index[::2])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=kwargs.get('loc_legend', [.79, .62, .2, .3]), frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
def plot_skill_by_persistence(
    skill: xr.Dataset, 
    xdim: str = 'probability', 
    coldim: str = 'approach', 
    linedim: str = 'persistence', 
    metric: str = 'f1', 
    benchmark: Optional[xr.Dataset] = None,
    save: Optional[Union[Path, str]] = None, 
    **kwargs
) -> None:
    """
    It generates a graph with as many line plots as approaches in the 'skill' dataset. The line plots reprensent the evolution of skill depending on persistence
    
    Parameters:
    -----------
    skill: xr.Dataset
        Skill of the several combinations of criteria tested. It contains as variables recall, precision and the metric specified in 'metric'
    xdim: str
       Name of the dimension in 'skill' that will be plotted in the X axis
    coldim: str
        Name of the dimension in 'skill' that defines the columns in the fiture
    linedime: str
        Name of the dimension in 'skill' that defines the lines in each of the plots
    metric: str
        Name of the target metric. This metric should be a variable in both datasets 'skill' and 'benchmark'
    benchmark: xr.Dataset, optional
        Skill of the a benchmark set of criteria
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Other parameters:
    -----------------
    alpha: float
        Transparency of the lines
    cmap: Union[str, mpl.colors.Colormap]
        Colour map used to plot the different lines in 'linedim'
    label: str
        Name given to the benchmark in the legend
    loc_leged: Tuple[float, float, float, float]
        Location of the legend (xmin, ymin, width, height)
    lw: float
        Width of the lines
    marker: str
        Symbol to use to plot the benchmark
    size: float
        Size of the marker that represents the benchmark
   
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    # extract kwargs
    marker = kwargs.get('marker', '+')
    s = kwargs.get('size', 50)
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .666)
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(skill[linedim])))).colors
    label = kwargs.get('label', 'current')
    
    # set up the figure
    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, figsize=(4.5 * ncols, 4), sharex=True, sharey=True)
    axes[0].set(xlim=(skill[xdim].min(), skill[xdim].max()),
           ylim=(-.02, 1.02))
    
    # plot skill
    for ax, col in zip(axes, skill[coldim].data):

        # benchmark skill
        if benchmark is not None:
            ax.axvline(benchmark[xdim], lw=lw / 2, c='k', ls=':', zorder=0)
            ax.axhline(benchmark[metric].data, lw=lw / 2, c='k', ls=':', zorder=0)
            ax.scatter(benchmark[xdim], benchmark[metric].data, marker=marker, lw=lw, c='k', s=s, zorder=20, label=label)
            
        # skill of the tested sets of criteria
        da = skill[metric].sel({coldim: col})
        for line, color in zip(da[linedim].data, colors):
            serie = da.sel({linedim: line}).to_pandas()
            ax.plot(serie.index, serie, c=color, lw=lw, alpha=alpha, label=line)
        
        # labels and title
        ax.set_xlabel(xdim)
        if ax == axes[0]:
            ax.set_ylabel(f'{metric} (-)')
        ax.set_title(col.replace('_', ' '))

    # legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=kwargs.get('loc_legend', [.78, .62, .2, .3]), frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_weights(
    Weights: xr.Dataset, 
    save: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    It creates a stacked bar plot to represent the distribution of weights among NWP models for each lead time. A plot is created for each variable in 'Weights'
    
    Parameters:
    -----------
    Weights: xr.Dataset
        It contains the DataArray of weights for each approach. It contains as many variables as approaches; every variable contains a matrix of 2 dimensions: leadtime and model.
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
        
    Other parameters:
    -----------------
    alpha: float
        Transparency of the bar plots
    cmap: Union[str, mpl.colors.Colormap]
        Colour map used to plot the different lines in 'linedim'
    lw: float
        Width of the lines representing specific lead times
    ls: str
        Style of the lines representing specific lead times
    offset: int
        Number of hours used to convert the initial lead time values into days
        
    Returns:
    --------
    None
        The plot shows the cumulative weights assigned to each model in every combination
    """
    
    # extract kwargs
    alpha = kwargs.get('alpha', .3)
    lw = kwargs.get('lw', .5)
    ls = kwargs.get('ls', ':')
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(Weights.model)))).colors
    offset = kwargs.get('offset', 12)
    
    # set up the plots
    ncols = len(Weights)
    fig, axes = plt.subplots(ncols=ncols, figsize=(4.5 * ncols, 4), sharex=True, sharey=True)

    for ax, (approach, da) in zip(axes, Weights.items()):
        
        # extract weights for an approach
        weights = da.to_pandas().transpose()
        weights.replace(np.nan, 0, inplace=True)
        weights.index += offset
        weights.index /= 24
        
        # barplot
        cumweight = pd.Series(0, index=weights.index)
        for model, color in zip(weights.columns, colors): #zip(['EUE', 'COS', 'EUD', 'DWD'], colors):
            ax.bar(weights.index, bottom=cumweight, height=weights[model], width=.5, align='edge', color=color, alpha=alpha, label=model)
            cumweight += weights[model]
        
        # auxiliary lines
        ax.axvline(weights.index[4], c='k', ls=ls, lw=lw)
        ax.axvline(weights.index[11], c='k', ls=ls, lw=lw)
        ax.axvline(weights.index[14], c='k', ls=ls, lw=lw)
        
        # configuraion
        ax.set_title(approach.replace('_', ' '))
        ax.set_xlabel('lead time (d)')
        if ax == axes[0]:
            ax.set_ylabel('cumulative weight (-)')
            ax.text(weights.index[4], 0.01, 'start notif.', rotation=90, ha='right', va='bottom', fontsize=10.5, color='k')
            ax.text(weights.index[11], 0.01, 'end COS', rotation=90, ha='right', va='bottom', fontsize=10.5, color='k')
            ax.text(weights.index[14], 0.01, 'end DWD', rotation=90, ha='right', va='bottom', fontsize=10.5, color='k')
        # ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    
    # plot limits
    ax.set(xlim=(weights.index.min(), weights.index.max()),
           ylim=(0, 1),
           xticks=weights.index[::4]);
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=[.8, .6, .2, .3], frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_brier_skill(
    BSS: xr.DataArray, 
    save: Optional[Union[str, Path]] = None, 
    **kwargs
) -> None:
    """
    A line plot of the evolution of the Brier skill score with lead time.
    
    Parameters:
    -----------
    BSS:    xr.DataArray (leadtime, model)
        Brier skill score
    save:   Union[str, Path]
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
        
    Other parameters:
    -----------------
    cmap: Union[str, mpl.colors.Colormap], optional
        Colour map used to plot the different lines in 'linedim'
    lw: float, optional
        Width of the lines
    offset: int, optional
        Number of hours used to convert the initial lead time values into days
    r: float, optional
        Rounding value
    
    Returns:
    --------
    None
        A line plot of the evolution of probabilistic skill (Brier skill score) with lead time, where every line represents a different model
    """
    
    r = kwargs.get('round', .2)
    lw = kwargs.get('lw', 1.4)
    cmap = kwargs.get('cmap', 'coolwarm_r')
    colors = ListedColormap(cmap(np.linspace(0, 1, len(BSS.model)))).colors
    offset = kwargs.get('offset', -12)
    
    df = BSS.to_pandas().transpose()
    df.index = (df.index + offset) / 24
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (4.5, 4.5)))

    for model, color in zip(df.columns, colors):
        ax.plot(df.index, df[model], lw=lw, c=color, label=model)
    
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
    else:
        ymax = np.round(np.ceil(df.abs().max().max() / r) * r, 2)
        ylim = (-ymax, ymax)
    ax.set(xlabel='lead time (d)', xlim=(df.index.min(), df.index.max()),
           ylabel='Brier skill score (-)', ylim=ylim)
    ax.set_xticks(df.index[::4])
    
    ax.axhline(0, c='k', lw=.5, zorder=0)
    ax.axvline(df.index[4], c='k', lw=.5, ls=':', zorder=0)
    ax.text(df.index[4], ylim[1] * .99, 'start notif.', rotation=90, va='top', ha='right', fontsize=10)
    ax.axvline(df.index[10], c='k', lw=.5, ls=':', zorder=0)
    ax.text(df.index[10], ylim[1] * .99, 'end COS', rotation=90, va='top', ha='right', fontsize=10)
    ax.axvline(df.index[13], c='k', lw=.5, ls=':', zorder=0)
    ax.text(df.index[13], ylim[1] * .99, 'end DWD', rotation=90, va='top', ha='right', fontsize=10)
    
    fig.legend(frameon=False, bbox_to_anchor=[1.1, .6, .1, .3]);
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')             
        
        
        
def PR_CSI(CSI: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a value of CSI (critical success index), it returns all possible pairs of values of precision and recall that correspond to that CSI
    
    Parameters:
    -----------
    CSI: float
        Value of the critical success index
        
    Returns:
    --------
    precision: np.ndarray
        Array of precision values
    recall: np.ndarray
        Array of recall values
    """
    
    assert 0 <= CSI <= 1, '"CSI" must be a value between 0 and 1'
    
    e = .005
    Pmin = 2 * CSI / (1 + CSI)
    P = np.linspace(Pmin, 1 + e, 100)
    R = CSI * P / (P - (1 - P) * CSI)
    
    return np.concatenate((R[::-1], P)), np.concatenate((P[::-1], R))



def PR_fscore(fscore: float, beta: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a value of the f-score, it returns all possible pairs of values of precision and recall that correspond to that f-score
    
    Parameters:
    -----------
    fscore: float
        Value of the fscore
    beta: float
        Coefficient that gives more weight to precision (beta < 0) or recall (beta > 0) in the computation of the f-score
    
    Returns:
    --------
    P: np.ndarray
        Array of precision values
    R: np.ndarray
        Array of recall values
    """
    
    assert 0 <= fscore <= 1, '"CSI" must be a value between 0 and 1'
    assert beta > 0, '"beta" must be a positive value.'
    
    e = .005
    Pmin = fscore / (1 + beta**2 * (1 - fscore))
    P = np.linspace(Pmin, 1 + e, 100)
    R = beta**2 * fscore * P / ((1 + beta**2) * P - fscore)
    
    return P, R



def roebber_diagram(
    metric: Literal['CSI', 'fscore'] = 'fscore',
    beta: float = 1,
    ax: Optional[mpl.axes.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, mpl.axes.Axes]:
    """
    It creates the figure of the Roebber diagram. This diagram shows in a single plot the precision and recall values (X and Y axis), the bias and the specified metric (background lines).
    
    Parameters:
    -----------
    metric: str
        Metric that will be shown in the background of the diagram. Either "CSI" (critical success index) or "fscore"
    beta: float
        If the metric is the f-score, coefficient that weights precision in the computation of the f-score
    ax: matplotlib.axes.Axes, optional
        Matplotlib axes where the diagram will be added. If not provided (default), a figure will be created
        
    Other parameters:
    -----------------
    figsize: Tuple[float, float], optional
        Size of every individual plot in the figure
    lw: float, optional
        Width of the lines
    lim: Tuple[float, float], optional
        Limits of the X and Y axis
    dashes: Tuple[int, int], optional
        Lenght of the lines and spaces in the dashed line that represents bias
    
    Returns:
    --------
    fig: plt.Figure
    ax: mpl.axes.Axes
    """
    
    assert beta > 0, '"beta" must be a positive value.'
    assert metric in ['CSI', 'fscore'], '"metric" must be one of these values: "CSI" or "fscore"'
    
    figsize = kwargs.get('figsize', (5, 5))
    lw = kwargs.get('lw', .5)
    lim = kwargs.get('lim', (-.02, 1.02))
    dashes = kwargs.get('dashes', (10, 10))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fs = ax.get_xticklabels()[0].get_fontsize()
    ax.plot([0, 1], [0, 1], lw=lw, c='k', ls='--', dashes=dashes)

    metric_values = np.arange(.2, 1., .2)
    P = np.linspace(0, 1, 101)

    for value in metric_values:
        if metric == 'CSI':
            P, R = PR_CSI(CSI=value)
        elif metric == 'fscore':
            P, R = PR_fscore(fscore=value, beta=beta)
        ax.plot(P, R, lw=lw, c='k', ls='-', zorder=0)
        i = int(len(P) * .8)
        ax.text(P[i], R[i], f'{value:.1}', ha='center', va='center', fontsize=fs,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.05'))#, backgroundcolor='w')
        for i in [0, -1]:
            ax.plot([0, P[i]], [0, R[i]], lw=lw, c='k', ls='--', dashes=dashes, zorder=0)
            bias = R[i] / P[i]
            if bias < 1:
                ax.text(lim[1] + .01, bias, f'{bias:.1f}', ha='left', va='center', fontsize=fs)
            if bias > 1:
                ax.text(1 / bias, lim[1] + .01, f'{bias:.1f}', ha='center', va='bottom', fontsize=fs)
        
    ax.set(xlim=(-.02, 1.02),
           xlabel='precision',
           ylim=(-.02, 1.02),
           ylabel='recall')
    ax.text(1.12, .5, 'bias', rotation=90, va='center')
    ax.text(.5, 1.1, 'bias', ha='center')
    
    if 'title' in kwargs:
        ax.text(.5, 1.125, kwargs['title'], ha='center', fontsize=12)
        
    return fig, ax



def plot_skill_by_area(
    skill: xr.Dataset, 
    optimal_criteria: Dict, 
    reference: Optional[Union[int, float]] = None, 
    metric: str = 'f1', 
    current_criteria: Optional[Dict] = None, 
    plot_prob: bool = False, 
    save: Optional[Union[Path, str]] = None, 
    **kwargs
) -> None:
    """
    It generates a graph with as many lineplots as approaches in the 'skill' dataset. The lineplots reprensent both the evolution of skill and probability with regard to a specified variable
    
    Parameters:
    -----------
    skill: xr.Dataset
        It contains as variables recall, precision and the specified metric
    optimal_criteria: Dict
        For each approach in skill, it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    reference: int or float, optional
        Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric: str
        Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria: dict, optional
        It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    plot_prob: bool
        Whether to add (True) or not (False) a fourth plot with the optimal probability threshold of each model
    save: str or pathlib.Path, optional
        Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Other parameters:
    -----------------
    alpha: float, optional
        Transparency of the lines
    colors: list[Union[str, mpl.Colors]], optional
        Colour map used to plot the different lines
    figsize: Tuple[float, float], optional
        Size of every individual plot in the figure
    lw: float, optional
        Width of the lines
    xlabel: str, optional
        Label of the X axis
    xlim: Tuple[float, float], optional
        Limits of the X axis
    xscale: Literal['linear', 'log', 'semilog'], optional
        Scaling of the X axis
    
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    alpha = kwargs.get('alpha', .666)
    colors = kwargs.get('colors', ['k', 'steelblue', 'orange'])
    figsize = kwargs.get('figsize', (4.5, 4))
    lw = kwargs.get('lw', 1.2)
    xlabel = kwargs.get('xlabel', None)
    xlim = kwargs.get('xlim', None)
    xscale = kwargs.get('xscale', 'linear')
    
    ncols = 4 if plot_prob else 3
    fig, axes = plt.subplots(ncols=ncols, figsize=(figsize[0] * ncols, figsize[1]), sharex=True, sharey=True)
    
    if current_criteria is not None:
        df = skill.sel(current_criteria).to_pandas()
        for ax, met in zip(axes, [metric, 'recall', 'precision']):
            ax.plot(df.index, df[met], 'k', lw=lw, label='current', zorder=0)
        if plot_prob:
            axes[-1].axhline(current_criteria['probability'], c='k', lw=lw, label='current', zorder=0)
            axes[-1].axvline(reference, c='k', ls='--', lw=1)
            axes[-1].set(xlabel=xlabel,
                        title='probability threshold')


    for i, (model, criteria) in enumerate(optimal_criteria.items()):
        if model == 'current':
            continue
        c = colors[model]
        if len(model.split('_')) > 1:
            label = ''.join([x[0] for x in model.split('_')]).upper()
        else:
            label = model
        zorder = 1 + i
        df = skill.sel(criteria).to_pandas()
        for j, (ax, met) in enumerate(zip(axes, [metric, 'recall', 'precision'])):
            ax.plot(df.index, df[met], lw=lw, c=c, alpha=alpha, label=label, zorder=i+1)
            if i == 0:
                ax.axvline(reference, c='k', ls=':', lw=.5)
                ax.set(xlabel=xlabel,
                       title=met)
            if (i == 0) & (j == 0):
                ax.text(reference, 0, 'current limit', rotation=90, ha='right', va='bottom', fontsize=14)
                if not plot_prob:
                    ax.set(ylabel='skill (-)')
        if plot_prob:
            axes[-1].axhline(criteria['probability'], lw=lw, c=c, alpha=alpha, label=label, zorder=i+1)


    ax.set(ylim=(-.02, 1.02),
           xlim=xlim,
           xscale=xscale)

    anchor = [.9, .5, .09, .43] if plot_prob else [.9, .5, .12, .43]
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, bbox_to_anchor=anchor);

    # export
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_area_distribution(
    area: pd.Series,
    mask: Optional[Union[List, pd.Series]] = None,
    reference: Optional[int] = 2000,
    save: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Plots a histogram of the number of reporting points depending on catchment area. If 'mask' is provided, the histogram of the whole set of reporting points is compared with that of a subset, e.g., the reporting points with observed flood events
    
    Parameters:
    -----------
    area: pd.Series
        Catchment area of the reporting points
    mask: list or pd.Series, optional
        Subset of stations. It is thought to be used as a selection of stations with observed events
    reference: int, optional
        Reference catchment area, e.g., the minimum area for which notifications are issued
    save: str or Path, optional
        If provided, file name where the plot will be saved
        
    Other parameters:
    -----------------
    alpha: float, optional
        Transparency
    figsize: Tuple, optional
        Size of the plot
    xlim: Tuple, optional
        Limits of the X axis
        
    Returns:
    --------
    None
        The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    alpha = kwargs.get('alpha', .5)
    figsize = kwargs.get('figsize', (6, 5.5))
    xlim = kwargs.get('xlim', (0, np.ceil(area.max() / 500) * 500))
    
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(area, ax=ax, binwidth=500, alpha=alpha, label='all')
    if mask is not None:
        sns.histplot(area[mask], ax=ax, binwidth=500, alpha=alpha, color='orange', label='w/ events')
    if reference is not None:
        ax.axvline(reference, color='k', ls=':', lw=.75)
    ax.set(xlabel='area (km²)',
           ylabel='no. reporting points',
           xlim=(xlim));
    ax.xaxis.set_major_locator(MultipleLocator(10000))
    ax.xaxis.set_minor_locator(MultipleLocator(2000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False)

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')