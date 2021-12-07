#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

# =============================================================================
# import
# =============================================================================


import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import pandas as pd
import os
import regionmask as rm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import pickle as pk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
from copy import deepcopy
import pickle as pk
import geopandas as gp
import mapclassify as mc
from scipy import stats as sts


# =============================================================================
# functions
# =============================================================================

#%%============================================================================

def nc_read(file,
            y1,
            var,
            obs=False,
            freq=False):
    
    """ Read in netcdfs based on variable and set time.
    
    Parameters
    ----------
    file : files in data directory
    
    Returns
    ------- 
    Xarray data array
    """
    
    ds = xr.open_dataset(file,decode_times=False)
    da = ds[var].squeeze()
    units, reference_date = da.time.attrs['units'].split('since')
    reference_date = reference_date.replace(reference_date[1:5],str(y1))[1:]
    new_date = pd.date_range(start=reference_date, periods=da.sizes['time'], freq='YS')
    da['time'] = new_date
    
    if 'height' in da.coords:
        da = da.drop('height')
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
        
    da = da.resample(time=freq,
                     closed='left',
                     label='left').mean('time') #this mean doesn't make sense for land cover maps
    # da['time'] = da['time.year']
    da['time'] = np.arange(1,11)
    
    return da

#%%============================================================================

def ar6_mask(da):
    
    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)
    ar6_regs = ar6_regs.where(landmask == 0)
    
    return ar6_regs

#%%============================================================================

def lreg(x, y):
    # Wrapper around scipy linregress to use in apply_ufunc
    slope, intercept, r_value, p_value, std_err = sts.linregress(x, y)
    return np.array([slope, p_value, r_value])

#%%============================================================================

def vectorize_lreg(da_y,
                   da_x=None):
    
    if da_x is not None:
        
        pass
    
    else:
        
        da_list = []
        for t in da_y.time.values:
            da_list.append(xr.where(da_y.sel(time=t).notnull(),t,da_y.sel(time=t)))
        da_x = xr.concat(da_list,dim='time')
        
    stats = xr.apply_ufunc(lreg, da_x, da_y,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[["parameter"]],
                           vectorize=True,
                           dask="parallelized",
                           output_dtypes=['float64'],
                           output_sizes={"parameter": 3})
    slope = stats.sel(parameter=0) 
    p_value = stats.sel(parameter=1)
    r_value = stats.sel(parameter=2)
    return slope,p_value,r_value

#%%============================================================================

def weighted_mean(continents,
                  da,
                  ar6_regs,
                  nt,
                  ns):
    
    nt = len(da.time.values)
    matrix = np.zeros(shape=(nt,ns))
    s = 0
    for c in continents.keys():
        for i in continents[c]:
            da_i = da.where(ar6_regs==i,
                            drop=True)                    
            for t in np.arange(nt):
                da_i_t = da_i.isel(time=t)
                weights = np.cos(np.deg2rad(da_i_t.lat))
                weights.name='weights'
                da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                matrix[t,s]= da_i_t_w
            s += 1
            
    return matrix


#%%============================================================================

def del_rows(matrix):
    
    # remove tsteps with nans (temporal x spatial shaped matrix)
    del_rows = []
    
    for i,row in enumerate(matrix):
        
        nans = np.isnan(row)
        
        if True in nans:
            
            del_rows.append(i)
            
    matrix = np.delete(matrix,
                       del_rows,
                       axis=0)
            
    return matrix

#%%============================================================================

def temp_center(ns,
                mod_ar6):
    
    for s in np.arange(ns):
        arr = mod_ar6[:,s]
        arr_mean = np.mean(arr)
        arr_center = arr - arr_mean
        mod_ar6[:,s] = arr_center
        
    return mod_ar6

#%%============================================================================

def ensembler(data_list,
              ax=False):
    if not ax:
        matrix = np.stack(data_list,axis=1)
        nx = np.shape(matrix)[1]
        ens_mean = np.mean(matrix,axis=1)
    
    if ax:
        matrix = np.stack(data_list,axis=ax)
        nx = np.shape(matrix)[ax]
        ens_mean = np.mean(matrix,axis=ax)
    
    return ens_mean,nx

#%%============================================================================

def da_ensembler(data):
    
    concat_dim = np.arange(len(data))
    aligned = xr.concat(data,dim=concat_dim)
    mean = aligned.mean(dim='concat_dim')
    
    return mean

#%%============================================================================

def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)
#%%============================================================================

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

#%%============================================================================

def pearson_correlation(x,y,dim):
    return xr.apply_ufunc(
        pearson_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

#%%============================================================================

def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)

#%%============================================================================

def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

#%%============================================================================

def ts_pickler(curDIR,
               ts,
               grid,
               t_ext,
               obs_mod):
    
    os.chdir(curDIR)
    if obs_mod == 'mod':
        pkl_file = open('mod_ts_{}-grid_{}.pkl'.format(grid,t_ext),'wb')
    elif obs_mod == 'obs':
        pkl_file = open('obs_ts_{}-grid_{}.pkl'.format(grid,t_ext),'wb')
    elif obs_mod == 'pic':
        pkl_file = open('pi_ts_{}-grid_{}.pkl'.format(grid,t_ext),'wb')
    pk.dump(ts,pkl_file)
    pkl_file.close()

#%%============================================================================

def pickler(curDIR,
            var_fin,
            analysis,
            grid,
            t_ext,
            exp_list):
    
    os.chdir(curDIR)
    if len(exp_list) == 2:
        pkl_file = open('var_fin_2-factor_{}-grid_{}_{}.pkl'.format(grid,analysis,t_ext),'wb')
    elif len(exp_list) == 1:
        exp = exp_list[0]
        pkl_file = open('var_fin_1-factor_{}_{}-grid_{}_{}.pkl'.format(exp,grid,analysis,t_ext),'wb')
    pk.dump(var_fin,pkl_file)
    pkl_file.close()
    
#%%============================================================================

def stats_subroutine(models,
                     ar6_land,
                     lulcc,
                     mod_ens,
                     maps):
    
    stats_ds = {}

    for mod in models:    
        
        stats_ds[mod] = xr.Dataset(
            coords=dict(
                lat = ('lat',ar6_land[mod].lat.values),
                lon = ('lon',ar6_land[mod].lon.values)
            )
        )
        mod_slope,mod_p,_ = vectorize_lreg(mod_ens[mod]['lu'])
        stats_ds[mod]['lu_slope'] = mod_slope
        stats_ds[mod]['lu_p'] = mod_p # maybe change to 0/1 for significance
        
        for lu in lulcc:
            
            # trends
            lc_slope,lc_p,_ = vectorize_lreg(maps[mod][lu])
            stats_ds[mod]['{}_slope'.format(lu)] = lc_slope
            stats_ds[mod]['{}_p'.format(lu)] = lc_p # maybe change to 0/1 for significance
            
            # correlation
            _,_,corr = vectorize_lreg(mod_ens[mod]['lu'],
                                    da_x=maps[mod][lu])
            stats_ds[mod]['lu-{}_corr'.format(lu)] = corr
            
    return stats_ds

#%%============================================================================

def glm_subroutine(models,
                   exps,
                   ar6_land,
                   mod_ens,
                   maps,
                   pi_data,
                   lulcc):
    
    glm_ds = {}

    for mod in models:    
        
        lat = ar6_land[mod].lat
        lon = ar6_land[mod].lon
        
        glm_ds[mod] = xr.Dataset(
            coords=dict(
                lat = ('lat',lat.values),
                lon = ('lon',lon.values)
            )
        )
        
        weights = np.cos(np.deg2rad(lat))
        weights.name='weights'
        
        for exp in exps:
            
            ts = mod_ens[mod][exp].weighted(weights).mean(('lon','lat'))
            glm_ds[mod][exp] = ts - ts.mean(dim='time')
            
            ts_land = mod_ens[mod][exp].where(ar6_land[mod]==1).weighted(weights).mean(('lon','lat'))
            glm_ds[mod]['{}_land'.format(exp)] = ts_land - ts_land.mean(dim='time')
            
        for lu in lulcc:
            
            ts_land = maps[mod][lu].where(ar6_land[mod]==1).weighted(weights).mean(('lon','lat'))
            glm_ds[mod]['{}_land'.format(lu)] = ts_land - ts_land.mean(dim='time')
            
        ts = pi_data[mod].weighted(weights).mean(('lon','lat'))
        glm_ds[mod]['pi'] = ts - ts.mean(dim='time')
        
        ts_land = pi_data[mod].where(ar6_land[mod]==1).weighted(weights).mean(('lon','lat'))
        glm_ds[mod]['pi_land'] = ts_land - ts_land.mean(dim='time')
        
    glm_ds['mmm'] = xr.Dataset(
        coords=dict(
            time = ('time',np.arange(1,11))
        )
    )
    glm_ds['mmm_land'] = xr.Dataset(
        coords=dict(
            time = ('time',np.arange(1,11))
        )
    )

    for exp in exps:
        
        concat_list = []
        concat_list_land = []
        
        for mod in models:
            
            concat_list.append(glm_ds[mod][exp])
            concat_list_land.append(glm_ds[mod]['{}_land'.format(exp)])
        
        glm_ds['mmm'][exp] = da_ensembler(concat_list)
        glm_ds['mmm_land']['{}_land'.format(exp)] = da_ensembler(concat_list_land)
        
    return glm_ds

#%%============================================================================

def data_lumper(dataset,
                models,
                maptype):
    
    if maptype == 'lu':
        
        data = np.empty(1)
        for mod in models:
            mod_data = dataset[mod]['lu_slope'].values.flatten()
            data = np.append(data,mod_data)
        
    elif maptype == 'lc':

        data = np.empty(1)
        for mod in models:
            for lc in ['treeFrac','cropFrac']:
                mod_data = dataset[mod]['{}_slope'.format(lc)].values.flatten()
                data = np.append(data,mod_data)
                
    elif maptype == 'corr':
               
        data = np.empty(1)
        for lc in ['treeFrac','cropFrac']:
            mod_data = dataset[lc]['lu-{}_corr'.format(lc)].values.flatten()
            data = np.append(data,mod_data)               
               
    data = data[~np.isnan(data)]
    return data

#%%============================================================================

def colormap_details(sequence_string,
                     data,
                     null_bnds):

    # identify colors for land cover transition trends
    cmap_brbg = plt.cm.get_cmap(sequence_string)
    cmap55 = cmap_brbg(0.01)
    cmap50 = cmap_brbg(0.05)   #blue
    cmap45 = cmap_brbg(0.1)
    cmap40 = cmap_brbg(0.15)
    cmap35 = cmap_brbg(0.2)
    cmap30 = cmap_brbg(0.25)
    cmap25 = cmap_brbg(0.3)
    cmap20 = cmap_brbg(0.325)
    cmap10 = cmap_brbg(0.4)
    cmap5 = cmap_brbg(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_brbg(0.525)
    cmap_10 = cmap_brbg(0.6)
    cmap_20 = cmap_brbg(0.625)
    cmap_25 = cmap_brbg(0.7)
    cmap_30 = cmap_brbg(0.75)
    cmap_35 = cmap_brbg(0.8)
    cmap_40 = cmap_brbg(0.85)
    cmap_45 = cmap_brbg(0.9)
    cmap_50 = cmap_brbg(0.95)  #red
    cmap_55 = cmap_brbg(0.99)

    colors = [cmap_45,
              cmap_35,
              cmap_30,
              cmap_25,
              cmap_10,
              cmap0,
              cmap10,
              cmap25,
              cmap30,
              cmap35,
              cmap45]

    cmap_list = mpl.colors.ListedColormap(colors,
                                          N=len(colors))
    
    cmap_list.set_over(cmap55)
    cmap_list.set_under(cmap_55)

    q_samples = []
    q_samples.append(np.abs(np.quantile(data,0.99)))
    q_samples.append(np.abs(np.quantile(data,0.01)))
        
    start = np.around(np.max(q_samples),decimals=4)
    inc = start/6
    # values = [np.around(-1*start,decimals=2),
    #           np.around(-1*start+inc,decimals=2),
    #           np.around(-1*start+inc*2,decimals=2),
    #           np.around(-1*start+inc*3,decimals=2),
    #           np.around(-1*start+inc*4,decimals=2),
    #           np.around(-1*start+inc*5,decimals=2),
    #           np.around(start-inc*5,decimals=2),
    #           np.around(start-inc*4,decimals=2),
    #           np.around(start-inc*3,decimals=2),
    #           np.around(start-inc*2,decimals=2),
    #           np.around(start-inc,decimals=2),
    #           np.around(start,decimals=2)]
    
    values = [np.around(-1*start,decimals=2),
              np.around(-1*start+inc,decimals=2),
              np.around(-1*start+inc*2,decimals=2),
              np.around(-1*start+inc*3,decimals=2),
              np.around(-1*start+inc*4,decimals=2),
              null_bnds[0],
              null_bnds[1],
              np.around(start-inc*4,decimals=2),
              np.around(start-inc*3,decimals=2),
              np.around(start-inc*2,decimals=2),
              np.around(start-inc,decimals=2),
              np.around(start,decimals=2)]

    tick_locs = [-1*start,
                 -1*start+inc,
                 -1*start+inc*2,
                 -1*start+inc*3,
                 -1*start+inc*4,
                 0,
                 start-inc*4,
                 start-inc*3,
                 start-inc*2,
                 start-inc,
                 start]

    tick_labels = [str(np.around(-1*start,decimals=2)),
                   str(np.around(-1*start+inc,decimals=2)),
                   str(np.around(-1*start+inc*2,decimals=2)),
                   str(np.around(-1*start+inc*3,decimals=2)),
                   str(np.around(-1*start+inc*4,decimals=2)),
                   str(0),
                   str(np.around(start-inc*4,decimals=2)),
                   str(np.around(start-inc*3,decimals=2)),
                   str(np.around(start-inc*2,decimals=2)),
                   str(np.around(start-inc,decimals=2)),
                   str(np.around(start,decimals=2))]

    norm = mpl.colors.BoundaryNorm(values,cmap_list.N)
    
    return cmap_list,tick_locs,tick_labels,norm,values

#%%============================================================================

def lineplot(glm_ds,
             models,
             exps,
             letters,
             outDIR):
    x=10
    y=10

    f,(ax1,ax2) = plt.subplots(nrows=2,
                               ncols=1,
                               figsize=(x,y))
            
    # cmap_whole = plt.cm.get_cmap('PRGn')
    cols={}
    cols['hist-noLu'] = plt.cm.get_cmap('PRGn')(0.35)
    cols['lu'] = plt.cm.get_cmap('PRGn')(0.75)
    cols['historical'] = '0.35'
    cols['pi'] = 'lightgray'
    mean_cols = {}
    mean_cols['hist-noLu'] = plt.cm.get_cmap('PRGn')(0.05)
    mean_cols['lu'] = plt.cm.get_cmap('PRGn')(0.95)
    mean_cols['historical'] = '0.05'
            
    for mod in models:
        
        for i in range(len(glm_ds[mod]['pi'].rls)):
            
            glm_ds[mod]['pi'].isel(rls=i).plot(ax=ax1,
                                            color=cols['pi'],
                                            zorder=1)
            glm_ds[mod]['pi_land'].isel(rls=i).plot(ax=ax2,
                                                    color=cols['pi'],
                                                    zorder=1)        
        
        for exp in exps:
            
            glm_ds[mod][exp].plot(ax=ax1,
                                color=cols[exp],
                                zorder=2)
            glm_ds[mod]['{}_land'.format(exp)].plot(ax=ax2,
                                                    color=cols[exp],
                                                    zorder=2)    

    for exp in exps:
        
        glm_ds['mmm'][exp].plot(ax=ax1,
                                color=mean_cols[exp],
                                lw=3,
                                zorder=3)
        glm_ds['mmm_land']['{}_land'.format(exp)].plot(ax=ax2,
                                                    color=mean_cols[exp],
                                                    lw=3,
                                                    zorder=3)
            
    n_t = len(glm_ds[mod][exp].time.values)

    i = 0
    for ax in (ax1,ax2):
        
        ax.tick_params(axis="x",
                    direction="in", 
                    left="off",
                    labelleft="on")
        ax.tick_params(axis="y",
                    direction="in")
        
        ax.set_xticks(np.arange(1,n_t+1))
                
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_ylabel('anomaly [°C]',
                    fontsize=12)
        
        ax.xaxis.set_ticklabels([])
        ax.set_xlabel(None)
        ax.set_title(letters[i],
                    loc='left',
                    fontweight='bold')
                    
        if i == 1:
            
            ax.set_xlabel('Years',
                        fontsize=12)
            ax.set_ylabel('land anomaly [°C]',
                        fontsize=12)
            
            ax.xaxis.set_ticklabels(np.arange(1965,2015,5))
            
        i += 1
        
    x0 = 0.1
    y0 = 0.95
    xlen = 0.55
    ylen = 0.5

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 2

    # legend
    legendcols = [mean_cols['lu'],
                  mean_cols['hist-noLu'],
                  mean_cols['historical'],
                  cols['pi']]

    handles = [Line2D([0],[0],linestyle='-',lw=2,color=legendcols[0]),\
               Line2D([0],[0],linestyle='-',lw=2,color=legendcols[1]),\
               Line2D([0],[0],linestyle='-',lw=2,color=legendcols[2]),\
               Line2D([0],[0],linestyle='-',lw=2,color=legendcols[3])]
        
    labels= ['lu',
             'hist-noLu',
             'historical',
             'pi']
    
    ax1.legend(handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            loc=3,   #bbox: (x, y, width, height)
            ncol=4, 
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            fontsize=12,
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad)
    
    f.savefig(outDIR+'/global_mean_timeseries.png')

#%%============================================================================

def trends_plot(stats_ds,
                models,
                lulcc,
                letters,
                null_bnds_lc,
                null_bnds_lu,
                outDIR):
    
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.75   # linewidth for coastlines

    # fonts
    title_font = 20
    cbtitle_font = 20
    tick_font = 18
    legend_font=12

    x=25
    y=15

    # placment lu trends cbar
    cb_lu_x0 = 0.1275
    cb_lu_y0 = 0.05
    cb_lu_xlen = 0.225
    cb_lu_ylen = 0.015

    # placment lc trends cbar
    cb_lc_x0 = 0.43
    cb_lc_y0 = 0.05
    cb_lc_xlen = 0.45
    cb_lc_ylen = 0.015

    # extent
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]

    cmap_list_lu,tick_locs_lu,tick_labels_lu,norm_lu,levels_lu = colormap_details('RdBu',
                                                                                  data_lumper(stats_ds,
                                                                                              models,
                                                                                              maptype='lu'),
                                                                                  null_bnds_lu)

    cmap_list_lc,tick_locs_lc,tick_labels_lc,norm_lc,levels_lc = colormap_details('BrBG_r',
                                                                                  data_lumper(stats_ds,
                                                                                              models,
                                                                                              maptype='lc'),
                                                                                  null_bnds_lc)

    f, axes = plt.subplots(nrows=len(models),
                           ncols=3,
                           figsize=(x,y),
                           subplot_kw={'projection':ccrs.PlateCarree()})

    cbax_lu = f.add_axes([cb_lu_x0, 
                        cb_lu_y0, 
                        cb_lu_xlen, 
                        cb_lu_ylen])
    cbax_lc = f.add_axes([cb_lc_x0, 
                        cb_lc_y0, 
                        cb_lc_xlen, 
                        cb_lc_ylen])

    i = 0
        
        

    for mod,row_axes in zip(models,axes):
        
        stats_ds[mod]['lu_slope'].plot(ax=row_axes[0],
                                       cmap=cmap_list_lu,
                                       cbar_ax=cbax_lu,
                                       levels=levels_lu,
                                       extend='both',
                                       center=0,
                                       add_labels=False)
        
        stats_ds[mod]['treeFrac_slope'].plot(ax=row_axes[1],
                                             cmap=cmap_list_lc,
                                             cbar_ax=cbax_lc,
                                             levels=levels_lc,
                                             extend='both',
                                             center=0,
                                             add_labels=False)
        
        stats_ds[mod]['cropFrac_slope'].plot(ax=row_axes[2],
                                             cmap=cmap_list_lc,
                                             cbar_ax=cbax_lc,
                                             levels=levels_lc,
                                             extend='both',
                                             center=0,
                                             add_labels=False)
        
        for ax,column in zip(row_axes,['LU response','treeFrac','cropFrac']):
            
            ax.set_extent(extent,
                          crs=ccrs.PlateCarree())
            ax.set_title(letters[i],
                         loc='left',
                         fontsize = title_font,
                         fontweight='bold')
            ax.coastlines(linewidth=cstlin_lw)
            
            if column == 'LU response':
                
                    if mod == 'CanESM5':
                        height = 0.3
                    else:
                        height= 0
                    ax.text(-0.1,
                            height,
                            mod,
                            fontsize=title_font,
                            fontweight='bold',
                            rotation='vertical',
                            transform=ax.transAxes)
            
            if i < 3:
                
                ax.set_title(column,
                            loc='center',
                            fontsize = title_font,
                            fontweight='bold')
                
            i += 1

    # lu response pattern colorbar
    cb_lu = mpl.colorbar.ColorbarBase(ax=cbax_lu, 
                                     cmap=cmap_list_lu,
                                     norm=norm_lu,
                                     spacing='uniform',
                                     orientation='horizontal',
                                     extend='both',
                                     ticks=tick_locs_lu,
                                     drawedges=False)
    cb_lu.set_label('LU trends (°C/5-years)',
                    size=title_font)
    cb_lu.ax.xaxis.set_label_position('top')
    cb_lu.ax.tick_params(labelcolor=col_cbticlbl,
                         labelsize=tick_font,
                         color=col_cbtic,
                         length=cb_ticlen,
                         width=cb_ticwid,
                         direction='out'); 
    cb_lu.ax.set_xticklabels(tick_labels_lu,
                             rotation=45)
    cb_lu.outline.set_edgecolor(col_cbedg)
    cb_lu.outline.set_linewidth(cb_edgthic)

    # lc pattern colorbar
    cb_lc = mpl.colorbar.ColorbarBase(ax=cbax_lc, 
                                     cmap=cmap_list_lc,
                                     norm=norm_lc,
                                     spacing='uniform',
                                     orientation='horizontal',
                                     extend='both',
                                     ticks=tick_locs_lc,
                                     drawedges=False)
    cb_lc.set_label('LC trends (% landcover/5-years)',
                    size=title_font)
    cb_lc.ax.xaxis.set_label_position('top')
    cb_lc.ax.tick_params(labelcolor=col_cbticlbl,
                        labelsize=tick_font,
                        color=col_cbtic,
                        length=cb_ticlen,
                        width=cb_ticwid,
                        direction='out'); 
    cb_lc.ax.set_xticklabels(tick_labels_lc,
                             rotation=45)
    cb_lc.outline.set_edgecolor(col_cbedg)
    cb_lc.outline.set_linewidth(cb_edgthic)

    # f.savefig(outDIR+'/lu_lc_trends_v2.png')

#%%============================================================================

def corr_plot(stats_ds,
              models,
              lulcc,
              letters,
              null_bnds_lc,
              outDIR):

    # fig size
    x=18
    y=15

    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.75   # linewidth for coastlines

    # fonts
    title_font = 20
    cbtitle_font = 20
    tick_font = 18
    legend_font=12

    # placment lu trends cbar
    cb_x0 = 0.275
    cb_y0 = 0.05
    cb_xlen = 0.5
    cb_ylen = 0.015

    # extent
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]

    # identify colors
    cmap_whole = plt.cm.get_cmap('RdBu_r')
    cmap55 = cmap_whole(0.01)
    cmap50 = cmap_whole(0.05)   #blue
    cmap45 = cmap_whole(0.1)
    cmap40 = cmap_whole(0.15)
    cmap35 = cmap_whole(0.2)
    cmap30 = cmap_whole(0.25)
    cmap25 = cmap_whole(0.3)
    cmap20 = cmap_whole(0.325)
    cmap10 = cmap_whole(0.4)
    cmap5 = cmap_whole(0.475)
    cmap0 = 'gray'
    cmap_5 = cmap_whole(0.525)
    cmap_10 = cmap_whole(0.6)
    cmap_20 = cmap_whole(0.625)
    cmap_25 = cmap_whole(0.7)
    cmap_30 = cmap_whole(0.75)
    cmap_35 = cmap_whole(0.8)
    cmap_40 = cmap_whole(0.85)
    cmap_45 = cmap_whole(0.9)
    cmap_50 = cmap_whole(0.95)  #red
    cmap_55 = cmap_whole(0.99)

    colors = [cmap_45,cmap_35,cmap_25,cmap_10,
                cmap0,
                cmap10,cmap25,cmap35,cmap45]

    # declare list of colors for discrete colormap of colorbar
    cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

    # colorbar args
    values = [-1,-.75,-.50,-.25,-0.01,0.01,.25,.5,.75,1]
    tick_locs = [-1,-.75,-.50,-.25,0,.25,.5,.75,1]
    tick_labels = ['-1','-0.75','-0.50','-0.25','0','0.25','0.50','0.75','1']
    norm = mpl.colors.BoundaryNorm(values,cmap_list.N)

    f, axes = plt.subplots(nrows=len(models),
                        ncols=2,
                        figsize=(x,y),
                        subplot_kw={'projection':ccrs.PlateCarree()})

    cbax = f.add_axes([cb_x0, 
                    cb_y0, 
                    cb_xlen, 
                    cb_ylen])

    i = 0

    for mod,row_axes in zip(models,axes):
        
        for ax,lc in zip(row_axes,['treeFrac','cropFrac']):
        
            plottable = stats_ds[mod]['lu-{}_corr'.format(lc)].where((stats_ds[mod]['{}_slope'.format(lc)]<null_bnds_lc[0])|\
                                                                     (stats_ds[mod]['{}_slope'.format(lc)]>null_bnds_lc[1]))
            
            plottable.plot(ax=ax,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap_list,
                           cbar_ax=cbax,
                           levels=values,
                           center=0,
                           add_labels=False)
        
        for ax,column in zip(row_axes,['LU,treeFrac','LU,cropFrac']):
            
            ax.set_extent(extent,
                          crs=ccrs.PlateCarree())
            ax.set_title(letters[i],
                         loc='left',
                         fontsize=title_font,
                         fontweight='bold')
            ax.coastlines(linewidth=cstlin_lw)
            
            if column == 'LU,treeFrac':
                
                    if mod == 'CanESM5':
                        height = 0.3
                    else:
                        height= 0
                    ax.text(-0.1,
                            height,
                            mod,
                            fontsize=title_font,
                            fontweight='bold',
                            rotation='vertical',
                            transform=ax.transAxes)
            
            if i < 2:
                
                ax.set_title(column,
                             loc='center',
                             fontsize = title_font,
                             fontweight='bold')
                
            i += 1
            
    cb = mpl.colorbar.ColorbarBase(ax=cbax, 
                                   cmap=cmap_list,
                                   norm=norm,
                                   spacing='uniform',
                                   orientation='horizontal',
                                   extend='neither',
                                   ticks=tick_locs,
                                   drawedges=False)
    cb.set_label('Correlation',
                 size=title_font)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelcolor=col_cbticlbl,
                      labelsize=tick_font,
                      color=col_cbtic,
                      length=cb_ticlen,
                      width=cb_ticwid,
                      direction='out'); 
    cb.ax.set_xticklabels(tick_labels,
                          rotation=45)
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)

    # f.savefig(outDIR+'/lu_lc_correlation_v2.png')

