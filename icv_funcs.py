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
    
    if 'height' in da.coords:
        da = da.drop('height')
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
        
    da = da.resample(time=freq,
                     closed='left',
                     label='left').mean('time') #this mean doesn't make sense for land cover maps
    
    # for freq of 25Y, collapse time dimenson by taking diff of 2 25 yr periods
    if freq == '25Y':
        
        da = da.isel(time=-1) - da.isel(time=0)
    
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

