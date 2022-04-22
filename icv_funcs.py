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
import scipy.stats as scp
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
from matplotlib.patches import Rectangle
from copy import deepcopy
import pickle as pk
import geopandas as gp
import mapclassify as mc

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
    da['time'] = np.arange(len(da.time))
    
    if 'height' in da.coords:
        da = da.drop('height')
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
        
    # da = da.resample(time=freq,
    #                  closed='left',
    #                  label='left').mean('time') #this mean doesn't make sense for land cover maps
    
    # for freq of 25Y, collapse time dimenson by taking diff of 2 25 yr periods
    # if freq == '25Y':
        
    #     da = da.isel(time=-1) - da.isel(time=0)
    
    return da

#%%============================================================================

def classifier(value):
    
    one_pc = 1/100
    five_pc = 5/100
    ten_pc = 10/100
    if value < one_pc:
        value = 1
    elif (value >= one_pc) & (value < five_pc):
        value = 2
    elif (value >= five_pc) & (value < ten_pc):
        value = 3
    elif value >= ten_pc:
        value = 4
    return value

#%%============================================================================

def ar6_mask(da):
    
    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)
    ar6_regs = ar6_regs.where(landmask == 0)
    
    return ar6_regs

#%%============================================================================

def cnt_mask(sfDIR,
             da):
    
    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)    
    ar6_regs = ar6_regs.where(landmask == 0)
    ar6_land = xr.where(ar6_regs > 0,1,0)
    os.chdir(sfDIR)
    gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
    gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    cnt_regs = rm.mask_geopandas(gpd_continents,lon,lat)
    cnt_regs = cnt_regs.where((ar6_regs != 0)&(ar6_regs != 20)&(ar6_land == 1))
    
    return cnt_regs

#%%============================================================================

def df_build(
    sfDIR,
    agg,
    continents,
    models
):
    # dataframes
    os.chdir(sfDIR)
    if agg == 'ar6':
        frame = {
            'trnd':[],
            'sgnl':[],
            'trnd_var':[],
            'sgnl_var':[],
            'trnd_p':[],
            'sgnl_p':[],
            'mod':[],
            'ar6':[]
        }
    elif agg == 'continental':
        frame = {
            'trnd':[],
            'sgnl':[],
            'trnd_var':[],
            'sgnl_var':[],
            'trnd_p':[],
            'sgnl_p':[],
            'mod':[],
            'continent':[]        
        }    
    df = pd.DataFrame(data=frame)
    regions = gp.read_file('IPCC-WGI-reference-regions-v4.shp')
    gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
    gpd_continents = gpd_continents[
        (gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')
    ] 
    regions = gp.clip(regions,gpd_continents)
    regions['keep'] = [0]*len(regions.Acronym)

    for c in continents.keys():
        for ar6 in continents[c]:
            regions.at[ar6,'Continent'] = c
            regions.at[ar6,'keep'] = 1  

    regions = regions[regions.keep!=0]  
    regions = regions.drop(columns='keep')

    add_cols = []
    for mod in models:
        for var in ['trnd','sgnl']:
            add_cols.append('{}_{}-p'.format(mod,var))
    regions = pd.concat([regions,pd.DataFrame(columns=add_cols)])
    if agg == 'continental':
        regions = regions.dissolve(by='Continent')
    
    return regions,df

#%%============================================================================

def lreg(x, y):
    # Wrapper around scipy linregress to use in apply_ufunc
    slope, intercept, r_value, p_value, std_err = scp.linregress(x, y)
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
        
    stats = xr.apply_ufunc(
        lreg, 
        da_x, 
        da_y,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[["parameter"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=['float64'],
        output_sizes={"parameter": 3}
    )
    slope = stats.sel(parameter=0) 
    p_value = stats.sel(parameter=1)
    r_value = stats.sel(parameter=2)
    
    return slope,p_value,r_value

#%%============================================================================

def nan_rm(
    da,
    ar6_regs,
    ar6,
    mod
):
    da = da.where(ar6_regs[mod]==ar6)
    da = da.values.flatten()
    da = da[~np.isnan(da)]
    
    return da

# %%
