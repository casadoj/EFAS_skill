import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
from optimize import find_best_criterion
from compute import exceedance2events, buffer_events
from plot.maps import create_cmap



def plot_correlation_matrix(corr, rho=.9, save=None, **kwargs):
    """It creates a heat map that shows the correlation matrix and highlights the cases in which the correlation coefficient exceeds a certain value
    
    Inputs:
    -------
    corr:  pd.DataFrame (n, n). Correlation matrix
    rho:   float. The maximum value allowed for the correlation coefficient between two reporting points
    save:       string. Directory where to save the plot as a JPG file. If None (default), the plot won't be saved 
    """
    
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
        
        
        
def plot_hits_by_variable(hits, optimal_criteria, variable, coldim='approach', reference=None, current_criteria=None, optimized_criteria=None, save=None, **kwargs):
    """It generates a graph with as many lineplots as approaches in the 'hits' dataset. The lineplots reprensent both the evolution of true positives (hits) and false positives (false alarms) and probability with regard to a specified variable
    
    Inputs:
    -------
    hits:               xr.Dataset (area, persistence, approach, probability). It contains as variables TP (true positives), FN (false negatives) and FP (false positives)
    optimal_criteria:   dict. For each approach in 'hits', it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable:           string. Name of the variable in 'hits' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    coldim:             string. Name of the dimension that defines each of the plots in the graph
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
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.9, .8, .2, .1]))
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_skill_by_variable(skill, optimal_criteria, variable, coldim='approach', reference=None, metric='f1', current_criteria=None, optimized_criteria=None,
                           shades=True, save=None, **kwargs):
    """It generates a graph with as many lineplots as approaches in the 'skill' dataset. The lineplots reprensent both the evolution of skill and probability with regard to a specified variable
    
    Inputs:
    -------
    skill:              xr.Dataset (area, persistence, approach, probability). It contains as variables recall, precision and the specified metric
    optimal_criteria:   dict. For each approach in skill, it contains a dictionary with the best combination of criteria for that approach {'approach', 'probability', 'persistence'}
    variable:           string. Name of the variable in 'hits' that will be displayed in the X axis. for which 'optimized_criteria' was fitted
    coldim:             string. Name of the dimension that defines each of the plots in the graph
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
            ax1.fill_between(skill_optimal.index, y1_optimal, y2_optimal, alpha=alpha, color=colors[1], zorder=1, label=f'P-R (optimal)')
        ax1.plot(skill_optimal.index, skill_optimal[metric], c=colors[1], lw=lw, label=f'{metric} (optimal)', zorder=7)
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
                ax1.fill_between(skill_var.index, y1_var, y2_var, alpha=alpha, color=colors[2], zorder=2, label=f'P-R ({variable} optimized)')
            ax1.plot(skill_var.index, skill_var[metric], c=colors[2], lw=lw, label=f'{metric} ({variable} optimized)', zorder=8)

        # reference line
        if reference is not None:
            ax1.axvline(x=reference, ls='-', lw=.5, color='k')

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
            prob = optimal_criteria[key]['probability']
            xmin = skill[variable].data.min()
            xmax = skill[metric].sel({coldim: key, 'persistence': persistence, 'probability': prob}).to_pandas().last_valid_index()
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
    fig.legend(handles, labels, loc=1, bbox_to_anchor=kwargs.get('loc_legend', [.945, .8, .2, .1]))
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_skill_training(train, test, complete=None, xdim='approach', save=None, **kwargs):
    """Scatter (and box) plot of the performance achieved for every approach in the train, test and complete data sets.
    
    Inputs:
    -------
    train:    xr.Dataset (xdim, (kfold)). The skill in the training set.The 'kfold' dimension is not mandatory, it would contain the skill in any of the folds of the cross-validation
    test:     xr.Dataset (xdim,). The skill in the test set
    complete: xr.Dataset (xdim,). The skill in the complete set (training + test)
    xdim:     string. Name of the dimension in 'train', 'test' and 'complete' to plot on the X axis
    save:     string. Directory and filename (including extension) where the graph will be saved
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
    fig.legend(handles, labels, bbox_to_anchor=kwargs.get('loc_legend', [.8, .8, .2, .1]))
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        


def plot_prediction(da, obs, probability, persistence=(1, 1), min_leadtime='all', center=True, w=5, save=None, **kwargs):
    """It creates a plot with several heat maps that explain the process applied to compute hits, misses and false alarms.
    
    1. On top it shows a heat map of total probability (values from 0 to 1) of exceeding the discharge threshold. 
    
    2. The second plot is a binary heat map of exceedances over the probability threshold.
    
    3. The third plot shows the time steps for which the persistence criterion is met.
    
    4. The fourth plot shows a buffer over the 3rd plot.
    
    5. The fifth plot shows a binary heat map of exceedances over the discharge threshold in the observed data.
    
    Inputs:
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
        
        
        
def plot_skill_by_probability(skill, probability, coldim='approach', reference=None, metric='f1', current_criteria=None, save=None, **kwargs):
    """It generates a graph with as many line plots as approaches in the 'skill' dataset. The line plots reprensent the evolution of skill depending on the probability threshold
    
    Inputs:
    -------
    skill:              xr.Dataset (area, persistence, approach, probability). It contains as variables recall, precision and the specified metric
    probability:        list or np.array. List of probability thresholds to be plotted
    coldim:             string. Name of the dimension that defines each of the plots in the graph
    reference:          int of float. Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric:             string. Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria:   dict. It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    save:               string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Ouput:
    ------
    The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm_r'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(probability)))).colors
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .666)
    
    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True, figsize=kwargs.get('figsize', (4.5 * ncols, 4)))
    
    if current_criteria is not None:
        df_current = skill.sel(current_criteria).to_pandas()
        
    for ax, key in zip(axes, skill[coldim].data):
        if current_criteria is not None:
            ax.plot((df_current.index - 12) / 24, df_current[metric], c='k', lw=lw * 1.2, label='current', zorder=25)
        
        for p, c in zip(probability, colors):
            criteria = {coldim: key, 'probability': p, 'persistence': '1/1'}
            df = skill.sel(criteria).to_pandas()
            if p == .5:
                ax.plot((df.index - 12) / 24, df['f0.8'], c='k', ls='--', lw=lw, alpha=1, label=f'P ≥ {p:.2f}', zorder=24)
            # elif p == .4:
            #     ax.plot((df.index - 12) / 24, df['f0.8'], c='k', ls=':', lw=lw, alpha=1, label=f'P ≥ {p:.2f}', zorder=24)
            else:
                ax.plot((df.index - 12) / 24, df['f0.8'], c=c, lw=lw, alpha=alpha, label=f'P ≥ {p:.2f}')
                
        if reference is not None:
            ax.axvline((reference - 12) / 24, c='k', ls='-', lw=lw / 2)
        # ax.text(2, .999, 'start notifications', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        # ax.axvline(5.5, c='k', ls=':', lw=lw / 2)
        # ax.text(5.5, .999, 'end COSMO', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        # ax.axvline(7, c='k', ls=':', lw=lw / 2)
        # ax.text(7, .999, 'end DWD', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        ax.set_xlabel(kwargs.get('xlabel', 'lead time ≥ (d)'))
        if ax == axes[0]:
            ax.set_ylabel(f'{metric} (-)')
        ax.set_title(key.replace('_', ' '))
        
    ax.set(xlim=kwargs.get('xlim', (0, 9)), ylim=kwargs.get('ylim', (-.02, 1.02)))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=kwargs.get('loc_legend', [.14, -.1, .6, .08]), frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_skill_by_probability(skill, probability, persistence='1/1', coldim='approach', ref_p=None, ref_lt=None, metric='f1', benchmark=None, save=None, **kwargs):
    """It generates a graph with as many line plots as approaches in the 'skill' dataset. The line plots reprensent the evolution of skill depending on the probability threshold
    
    Inputs:
    -------
    skill:              xr.Dataset (area, persistence, approach, probability). It contains as variables recall, precision and the specified metric
    probability:        list or np.array. List of probability thresholds to be plotted
    persistence:        str. Fixed value of persistence
    coldim:             string. Name of the dimension that defines each of the plots in the graph
    reference:          int of float. Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric:             string. Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria:   dict. It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    save:               string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Ouput:
    ------
    The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm_r'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(probability)))).colors
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .666)
    
    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True, figsize=kwargs.get('figsize', (4.5 * ncols, 4)))
    
    if benchmark is not None:
        df_bm = benchmark.to_pandas()
        
    for ax, key in zip(axes, skill[coldim].data):
        if benchmark is not None:
            ax.plot((df_bm.index - 12) / 24, df_bm[metric], c='k', lw=lw * 1.2, label='current', zorder=25)
        
        for p, c in zip(probability, colors):
            criteria = {coldim: key, 'probability': p, 'persistence': persistence}
            df = skill.sel(criteria).to_pandas()
            if p == ref_p:
                ax.plot((df.index - 12) / 24, df['f0.8'], c='k', ls='--', lw=lw, alpha=1, label=f'P ≥ {p:.2f}', zorder=24)
            # elif p == .4:
            #     ax.plot((df.index - 12) / 24, df['f0.8'], c='k', ls=':', lw=lw, alpha=1, label=f'P ≥ {p:.2f}', zorder=24)
            else:
                ax.plot((df.index - 12) / 24, df['f0.8'], c=c, lw=lw, alpha=alpha, label=f'P ≥ {p:.2f}')
                
        if ref_lt is not None:
            ax.axvline((ref_lt - 12) / 24, c='k', ls='-', lw=lw / 2)
        # ax.text(2, .999, 'start notifications', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        # ax.axvline(5.5, c='k', ls=':', lw=lw / 2)
        # ax.text(5.5, .999, 'end COSMO', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        # ax.axvline(7, c='k', ls=':', lw=lw / 2)
        # ax.text(7, .999, 'end DWD', rotation=90, horizontalalignment='right', verticalalignment='top', fontsize=10)
        ax.set_xlabel(kwargs.get('xlabel', 'lead time ≥ (d)'))
        if ax == axes[0]:
            ax.set_ylabel(f'{metric} (-)')
        ax.set_title(key.replace('_', ' '))
        
    ax.set(xlim=kwargs.get('xlim', (0, 9)), ylim=kwargs.get('ylim', (-.02, 1.02)))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, bbox_to_anchor=kwargs.get('loc_legend', [.14, -.1, .6, .08]), frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');
        
        
        
def plot_weights(Weights, save=None, **kwargs):
    """It creates a stacked bar plot to represent the distribution of weights among NWP models for each lead time. A plot is created for each approach in 'Weights'
    
    Inputs:
    -------
    Weights:   xr.Dataset (leadtime, model). It contains the DataArray of weights for each approach
    save:      string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    """
    
    # extract kwargs
    alpha = kwargs.get('alpha', .3)
    lw = kwargs.get('lw', .5)
    ls = kwargs.get('ls', ':')
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(Weights.model)))).colors
    
    # set up the plots
    ncols = len(Weights)
    fig, axes = plt.subplots(ncols=ncols, figsize=(4.5 * ncols, 4), sharex=True, sharey=True)

    for ax, (approach, da) in zip(axes, Weights.items()):
        
        # extract weights for an approach
        weights = da.to_pandas().transpose()
        weights.replace(np.nan, 0, inplace=True)
        weights.index -= 12
        
        # barplot
        cumweight = pd.Series(0, index=weights.index)
        for model, color in zip(['EUE', 'COS', 'EUD', 'DWD'], colors):
            ax.bar(weights.index, bottom=cumweight, height=weights[model], width=12, align='edge', color=color, alpha=alpha, label=model)
            cumweight += weights[model]
        
        # auxiliary lines
        ax.axvline(2 * 24, c='k', ls=ls, lw=lw)
        ax.axvline(5.5 * 24, c='k', ls=ls, lw=lw)
        ax.axvline(7 * 24, c='k', ls=ls, lw=lw)
        if ax == axes[-1]:
            ax.text(2 * 24, 0.01, 'start notifications', rotation=90, horizontalalignment='right', verticalalignment='bottom', fontsize=11)
            ax.text(5.5 * 24, 0.01, 'end COS', rotation=90, horizontalalignment='right', verticalalignment='bottom', fontsize=11)
            ax.text(7 * 24, 0.01, 'end DWD', rotation=90, horizontalalignment='right', verticalalignment='bottom', fontsize=11)
        
        # configuraion
        ax.set_title(approach.replace('_', ' '))
        ax.set_xlabel('lead time (h)')
        if ax == axes[0]:
            ax.set_ylabel('cumulative weight (-)')
        # ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    
    # plot limits
    xticks = weights.index[::4]
    ax.set(xlim=(weights.index.min(), weights.index.max()), ylim=(0, 1), xticks=xticks);
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=[.8, .6, .2, .3], frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_brier_skill(BSS, save=None, **kwargs):
    """A line plot of the evolution of the Brier skill score with lead time.
    
    Inputs:
    -------
    BSS:    xr.DataArray (leadtime, model). Brier skill score
    save:   string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    """
    
    r = kwargs.get('round', .2)
    lw = kwargs.get('lw', 1.4)
    cmap = kwargs.get('cmap', 'coolwarm_r')
    colors = ListedColormap(cmap(np.linspace(0, 1, len(BSS.model)))).colors
    
    
    df = BSS.to_pandas().transpose()

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (4.5, 4.5)))

    ax.axhline(0, c='k', lw=.5, zorder=0)
    ax.axvline(df.index[4], c='k', lw=.5, ls=':', zorder=0)


    for model, color in zip(['EUE', 'COS', 'EUD', 'DWD'], colors):
        ax.plot(df.index, df[model], lw=lw, c=color, label=model)

    ymax = np.round(np.ceil(df.abs().max().max() / r) * r, 2)
    ax.set(xlabel='lead time (h)', xlim=(df.index.min(), df.index.max()),
           ylabel='Brier skill score (-)', ylim=(-ymax, ymax))
    ax.set_xticks(df.index[::4])
    fig.legend(frameon=False, bbox_to_anchor=[1.1, .6, .1, .3]);
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        
        
        
def plot_skill_by_persistence(skill, xdim='probability', coldim='approach', linedim='persistence', metric='f1',
                              benchmark=None, save=None, **kwargs):
    """It generates a graph with as many line plots as approaches in the 'skill' dataset. The line plots reprensent the evolution of skill depending on persistence
    
    Inputs:
    -------
    skill:              xr.Dataset (area, persistence, approach, probability). It contains as variables recall, precision and the specified metric
    probability:        list or np.array. List of probability thresholds to be plotted
    coldim:             string. Name of the dimension that defines each of the plots in the graph
    reference:          int of float. Fixed value of the 'variable' for which 'optimal_criteria' was fitted
    metric:             string. Name of the target metric. This metric should be a variable in both datasets 'skill' and 'optmized_criteria'
    current_criteria:   dict. It contains the current operation criteria used in EFAS {'approach', 'probability', 'persistence'}
    save:               string. Path where the graph will be saved. By default is 'None', and the graph is not saved.
    
    Ouput:
    ------
    The graph is plotted on the screen, and saved if a path is set in the attribute 'save'
    """
    
    lw = kwargs.get('lw', 1.2)
    alpha = kwargs.get('alpha', .666)
    cmap = plt.get_cmap(kwargs.get('cmap', 'coolwarm'))
    colors = ListedColormap(cmap(np.linspace(0, 1, len(skill[linedim])))).colors

    ncols = len(skill[coldim])
    fig, axes = plt.subplots(ncols=ncols, figsize=(4.5 * ncols, 4), sharex=True, sharey=True)

    for ax, col in zip(axes, skill[coldim].data):

        da = skill[metric].sel({coldim: col})
        
        if benchmark is not None:
            # ax.axvline(current_criteria[xdim], lw=lw / 2, c='k', zorder=0)
            ax.scatter(benchmark[xdim], benchmark[metric].data, marker='x', lw=lw, c='k', zorder=20, label='current')
        for line, color in zip(da[linedim].data, colors):
            serie = da.sel({linedim: line}).to_pandas()
            ax.plot(serie.index, serie, c=color, lw=lw, alpha=alpha, label=line)

        ax.set_xlabel(xdim)
        if ax == axes[0]:
            ax.set_ylabel(f'{metric} (-)')
        ax.set_title(col.replace('_', ' '))

    ax.set(xlim=(skill[xdim].min(), skill[xdim].max()), ylim=(-.02, 1.02))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=kwargs.get('loc_legend', [.78, .62, .2, .3]), frameon=False);

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight');