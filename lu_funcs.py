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
# %%
# def data_lumper(dataset,
#                 models,
#                 maptype):
    
#     if maptype == 'lu':
        
#         data = np.empty(1)
#         for mod in models:
#             mod_data = dataset[mod]['lu_slope'].values.flatten()
#             data = np.append(data,mod_data)
        
#     elif maptype == 'lc':

#         data = np.empty(1)
#         for mod in models:
#             for lc in ['treeFrac','cropFrac']:
#                 mod_data = dataset[mod]['{}_slope'.format(lc)].values.flatten()
#                 data = np.append(data,mod_data)
               
#     data = data[~np.isnan(data)]
#     return data


# def colormap_details(sequence_string,
#                      data,
#                      null_bnds):

#     # identify colors for land cover transition trends
#     cmap_brbg = plt.cm.get_cmap(sequence_string)
#     cmap55 = cmap_brbg(0.01)
#     cmap50 = cmap_brbg(0.05)   #blue
#     cmap45 = cmap_brbg(0.1)
#     cmap40 = cmap_brbg(0.15)
#     cmap35 = cmap_brbg(0.2)
#     cmap30 = cmap_brbg(0.25)
#     cmap25 = cmap_brbg(0.3)
#     cmap20 = cmap_brbg(0.325)
#     cmap10 = cmap_brbg(0.4)
#     cmap5 = cmap_brbg(0.475)
#     cmap0 = 'gray'
#     cmap_5 = cmap_brbg(0.525)
#     cmap_10 = cmap_brbg(0.6)
#     cmap_20 = cmap_brbg(0.625)
#     cmap_25 = cmap_brbg(0.7)
#     cmap_30 = cmap_brbg(0.75)
#     cmap_35 = cmap_brbg(0.8)
#     cmap_40 = cmap_brbg(0.85)
#     cmap_45 = cmap_brbg(0.9)
#     cmap_50 = cmap_brbg(0.95)  #red
#     cmap_55 = cmap_brbg(0.99)

#     colors = [cmap_45,
#               cmap_35,
#               cmap_30,
#               cmap_25,
#               cmap_10,
#               cmap0,
#               cmap10,
#               cmap25,
#               cmap30,
#               cmap35,
#               cmap45]

#     cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

#     cmap_list.set_over(cmap55)
#     cmap_list.set_under(cmap_55)

#     q_samples = []
#     q_samples.append(np.abs(np.quantile(data,0.99)))
#     q_samples.append(np.abs(np.quantile(data,0.01)))
        
#     start = np.around(np.max(q_samples),decimals=4)
#     inc = start/6
#     values = [-1*start,
#               -1*start+inc,
#               -1*start+inc*2,
#               -1*start+inc*3,
#               -1*start+inc*4,
#               null_bnds[0],
#               null_bnds[1],
#               start-inc*4,
#               start-inc*3,
#               start-inc*2,
#               start-inc,
#               start]

#     tick_locs = [-1*start,
#                  -1*start+inc,
#                  -1*start+inc*2,
#                  -1*start+inc*3,
#                  -1*start+inc*4,
#                  0,
#                  start-inc*4,
#                  start-inc*3,
#                  start-inc*2,
#                  start-inc,
#                  start]

#     tick_labels = [str(np.around(-1*start,decimals=3)),
#                    str(np.around(-1*start+inc,decimals=3)),
#                    str(np.around(-1*start+inc*2,decimals=3)),
#                    str(np.around(-1*start+inc*3,decimals=3)),
#                    str(np.around(-1*start+inc*4,decimals=3)),
#                    str(0),
#                    str(np.around(start-inc*4,decimals=3)),
#                    str(np.around(start-inc*3,decimals=3)),
#                    str(np.around(start-inc*2,decimals=3)),
#                    str(np.around(start-inc,decimals=3)),
#                    str(np.around(start,decimals=3))]

#     norm = mpl.colors.BoundaryNorm(values,cmap_list.N)
    
#     return cmap_list,tick_locs,tick_labels,norm,values