#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# PCA analysis

# test effect of weighting on solver

# v2:
    # addition of forest + crop cover maps for lu check
    # option for pseudo pc from projecting "lu" onto forest and crop eof


# =============================================================================
# import
# =============================================================================


import sys
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
import xarray as xr
from eofs.xarray import Eof



#==============================================================================
# path
#==============================================================================


curDIR = '/Users/Luke/Documents/PHD/lumip/da'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'data/obs')
modDIR = os.path.join(curDIR, 'data/mod')
# piDIR = os.path.join(curDIR, 'data/pi/final')
mapDIR = os.path.join(curDIR, 'data/map/final')
outDIR = os.path.join(curDIR, 'figures_v3')

# bring in functions
from fp_funcs import *


#==============================================================================
# options - analysis
#==============================================================================


# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_pickle=1     # 0: do not pickle objects
                  # 1: pickle objects after sections 'read' and 'analyze'

# << SELECT >>
flag_svplt=0;     # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_tres=3;    # 0: jja
                # 1: djf
                # 2: annual
                # 3: max month

# << SELECT >>
flag_obs_type=1;   # 0: cru
                   # 1: berkley_earth
                
# << SELECT >>
flag_y1=1;        # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0;        # 0: 50
                   # 1: 100

# << SELECT >>
flag_var=0;  # 0: tasmax


# << SELECT >>
flag_analysis=0;  # 0: projections onto EOF of hist vs histnolu mmm
                  # 1: projections onto EOF of LUH2 forest + crop
                  
# << SELECT >>
flag_landcover=0;  # 0: forest
                   # 1: crops
                   
# << SELECT >>
flag_correlation=0;  # 0: no
                     # 1: yes
                  
# redefine some flags for luh2 testing
if flag_analysis == 1:
    
    flag_y1=0
    flag_len=1

# << SELECT >>
trunc=0

seasons = ['jja',
           'djf',
           'annual',
           'max']
obs_types = ['cru',
             'berkley_earth']
analyses = ['global',
            'continental']
deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['forest',
         'crops',
         'urban']
measures = ['relative_change',
            'absolute_change',
            'area_change',
            'all_pixels']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y']
variables = ['tasmax']
analyses = ['models',
            'luh2']
landcover_types = ['forest',
                   'crops']
correlation_opts = ['no',
                    'yes']

tres = seasons[flag_tres]
obs = obs_types[flag_obs_type]
y1 = start_years[flag_y1]
length = lengths[flag_len]
var = variables[flag_var]
analysis = analyses[flag_analysis]
landcover = landcover_types[flag_landcover]
correlation = correlation_opts[flag_correlation]

# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

exps_start = ['historical',
              'hist-noLu']

#==============================================================================
# mod + obs + luh2 data 
#==============================================================================

mod_files = {}
mod_data = {}
obs_data = {}
luh2_data = {}

# individual model ensembles and lu extraction
for obs in obs_types:
    
    i = 0
    os.chdir(modDIR)
    mod_files[obs] = {}
    mod_data[obs] = {}
    
    # mod data
    for mod in models:
        
        mod_files[obs][mod] = {}
        mod_data[obs][mod] = {}
        
        for exp in exps_start:
            
            for file in [file for file in sorted(os.listdir(modDIR))\
                         if var in file\
                             and mod in file\
                             and exp in file\
                             and tres in file\
                             and 'ensmean' in file\
                             and 'unmasked' in file\
                             and obs in file\
                             and t_ext in file]:
                
                mod_files[obs][mod][exp] = file
                
            if i == 0:
                
                mod_data[obs][mod][exp] = nc_read(mod_files[obs][mod][exp],
                                                  y1,
                                                  var,
                                                  obs)
            
                ar6_regs = ar6_mask(mod_data[obs][mod][exp],
                                    obs)
                ar6_land = xr.where((ar6_regs>0)&(ar6_regs<44),1,0)
                mod_data[obs][mod][exp] = mod_data[obs][mod][exp].where(ar6_land==1)
                i += 1
                
            elif i >= 1:
                
                mod_data[obs][mod][exp] = nc_read(mod_files[obs][mod][exp],
                                                  y1,
                                                  var,
                                                  obs).where(ar6_land==1)
                
        mod_data[obs][mod]['lu'] = mod_data[obs][mod]['historical'] - mod_data[obs][mod]['hist-noLu']
        mod_data[obs][mod]['lu'] = mod_data[obs][mod]['lu'].where(ar6_land==1)
        
    mod_data[obs]['mmm'] = {}  
    
    for exp in ['historical','hist-noLu','lu']:
        
        data = []
        
        for mod in models:
            
            data.append(mod_data[obs][mod][exp])
            
        mod_data[obs]['mmm'][exp] = da_ensembler(data)
        
    # obs data
    os.chdir(obsDIR)      
    
    for file in [file for file in sorted(os.listdir(obsDIR))\
                 if var in file\
                     and 'obs' in file\
                     and obs in file\
                     and 'unmasked' in file\
                     and t_ext in file]:
        
        obs_data[obs] = nc_read(file,
                                y1,
                                var,
                                obs=obs)
        
        if obs == 'berkley_earth':
            
            clim_file = "tasmax_obs_berkley_earth_climatology.nc"
            clim = xr.open_dataset(clim_file,
                                   decode_times=False).tasmax
            clim = clim.squeeze(drop=True)
            clim = clim.rename({'latitude':'lat','longitude':'lon'})
            obs_data[obs] = obs_data[obs] + clim
            obs_data[obs] = obs_data[obs].where(ar6_land==1)
            
        elif obs == "cru":
            
            obs_data[obs] = obs_data[obs].where(ar6_land==1)
         
    # luh2 data
    os.chdir(mapDIR)
    luh2_data[obs] = {}
    
    for lc in landcover_types:
        for file in [file for file in sorted(os.listdir(mapDIR))\
                     if obs in file\
                         and 'grid' in file\
                         and lc in file\
                         and '191501_201412' in file]:
            
            luh2_data[obs][lc] = nc_read(file,
                                         y1,
                                         var='cell_area')
            
            if obs == 'berkley_earth':
                
                luh2_data[obs][lc] = luh2_data[obs][lc].rename({'latitude':'lat','longitude':'lon'})
            
            if correlation == "yes":
                
                # adjust data to be temporally centered and unit variance (check xarray for this)
                    # also adjust model data
                print("not ready yet")
                
            else:
                
                luh2_data[obs][lc] = luh2_data[obs][lc].where(ar6_land==1)
                
            
#==============================================================================
# mutual mask of data 
#==============================================================================


for obs in obs_types:
        
    exp = 'lu'
    arr1 = cp.deepcopy(mod_data[obs]['mmm'][exp].isel(time=0))
    
    if analysis == "models":
    
        arr2 = cp.deepcopy(obs_data[obs].isel(time=0))
        
    elif analysis == "luh2":
        
        arr2 = cp.deepcopy(luh2_data[obs][lc].isel(time=0))
    
    arr1 = arr1.fillna(-999)
    arr2 = arr2.fillna(-999)
    
    mask1 = xr.where(arr2!=-999,1,0)
    mask2 = xr.where(arr1!=-999,1,0).where(mask1==1).fillna(0)
    
    
    if analysis == "models":
        
        for exp in ['historical','hist-noLu','lu']:
            
            mod_data[obs]['mmm'][exp] = mod_data[obs]['mmm'][exp].where(mask1==1)
            obs_data[obs] = obs_data[obs].where(mask2==1)
        
    
    elif analysis == "luh2":
        
        for exp in ['historical','hist-noLu','lu']:
            
            mod_data[obs]['mmm'][exp] = mod_data[obs]['mmm'][exp].where(mask1==1)        
        
        luh2_data[obs][lc] = luh2_data[obs][lc].where(mask2==1)
        
            
#==============================================================================
# pca
#==============================================================================


solver_dict = {}
eof_dict = {}
pcs = {}
pseudo_pcs = {}

if analysis == "models":
    
    for obs in obs_types:
        
        solver_dict[obs] = {}
        eof_dict[obs] = {}
        pcs[obs] = {}
        pseudo_pcs[obs] = {}
        
        weights = np.cos(np.deg2rad(mod_data[obs]['mmm']['lu'].lat))
        weighted_arr = xr.zeros_like(mod_data[obs]['mmm']['lu']).isel(time=0)    
        x = weighted_arr.coords['lon']
        
        for y in weighted_arr.lat.values:
            
            weighted_arr.loc[dict(lon=x,lat=y)] = weights.sel(lat=y).item()
            
        for exp in ['historical','hist-noLu','lu']:
        
            solver_dict[obs][exp] = Eof(mod_data[obs]['mmm'][exp],
                                        weights=weighted_arr) 
            eof_dict[obs][exp] = solver_dict[obs][exp].eofs(neofs=1)
            pcs[obs][exp] = solver_dict[obs][exp].pcs(npcs=1)
            pseudo_pcs[obs][exp] = solver_dict[obs][exp].projectField(obs_data[obs],
                                                                      neofs=1)
            
        solver_dict[obs][obs] = Eof(obs_data[obs],
                                    weights=weighted_arr)
        eof_dict[obs][obs] = solver_dict[obs][obs].eofs(neofs=1)
        
    figure_data = {}
    
    for obs in obs_types:
        
        figure_data[obs] = []
        figure_data[obs].append(eof_dict[obs][obs])
        delta_eof = eof_dict[obs]['historical'] - eof_dict[obs]['hist-noLu']
        figure_data[obs].append(delta_eof)
        bias_hist = eof_dict[obs][obs] - eof_dict[obs]['historical']
        bias_histnolu = eof_dict[obs][obs] - eof_dict[obs]['hist-noLu']
        delta_bias = bias_hist - bias_histnolu
        figure_data[obs].append(delta_bias)
        
elif analysis == "luh2":
    
    for obs in obs_types:
        
        solver_dict[obs] = {}
        eof_dict[obs] = {}
        pcs[obs] = {}
        pseudo_pcs[obs] = {}
        
# =============================================================================
#         weights = np.cos(np.deg2rad(mod_data[obs]['mmm']['lu'].lat))
#         weighted_arr = xr.zeros_like(mod_data[obs]['mmm']['lu']).isel(time=0)    
#         x = weighted_arr.coords['lon']
#         
#         for y in weighted_arr.lat.values:
#             
#             weighted_arr.loc[dict(lon=x,lat=y)] = weights.sel(lat=y).item()
# =============================================================================
            
        for lc in landcover_types:
        
            solver_dict[obs][lc] = Eof(luh2_data[obs][lc]) 
            eof_dict[obs][lc] = solver_dict[obs][lc].eofs(neofs=1)
            pcs[obs][lc] = solver_dict[obs][lc].pcs(npcs=1)
            pseudo_pcs[obs][lc] = solver_dict[obs][lc].projectField(mod_data[obs]['mmm']['lu'],
                                                                    neofs=1)
    
    print("nothing yet")
     
           
# =============================================================================
# plotting        
# =============================================================================


if analysis == "models":
    
    pca_plot(eof_dict,
             pcs,
             pseudo_pcs,
             figure_data,
             obs_types,
             outDIR)
    
elif analysis == "luh2":
    
    print("nothing yet")
            
    
                      
        
        
             
        
        
