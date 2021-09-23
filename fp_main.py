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
    
# why isn't 'cru' being loaded? (solved)

# change plotting for analysis="models":
    # don't want diffrence in eofs but should rather just show hist and histno-lu

# change plots to have mmm time series as one bold line and range as coming from all other mmms

# add pc1

# produce for all ar6 regions

# is decreasing flank of forest pseudo-pcs because major axis of change is deforestation and mid century has reforestation?
    # could i check this against 1900-1950 and 1950-2000 attempts?

# option for standardized data:
    # http://xarray.pydata.org/en/stable/examples/weather-data.html
    # possibly important for comparins pseudo-pcs of "lu" against 
        # (is scale difference in PC and pseudo-PCs due to data not being standardized?)
        # can PCs be interpreted similarly as pseudo PCs?
    # 



#%%==============================================================================
# import
# ===============================================================================


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


#%%==============================================================================
# path
#================================================================================


curDIR = '/home/luke/documents/lumip/d_a/'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'data/obs/final')
modDIR = os.path.join(curDIR, 'data/mod/final')
# piDIR = os.path.join(curDIR, 'data/pi/final')
mapDIR = os.path.join(curDIR, 'data/map/final')
outDIR = os.path.join(curDIR, 'figures_v3')

# bring in functions
from fp_funcs import *


#%%==============================================================================
# options - analysis
#================================================================================


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
flag_y1=0;        # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0;        # 0: 50
                   # 1: 100

# << SELECT >>
flag_var=0;  # 0: tasmax


# << SELECT >>
flag_analysis=1;  # 0: projections onto EOF of hist vs histnolu mmm
                  # 1: projections onto EOF of LUH2 forest + crop
                  
# << SELECT >>
flag_landcover=0;  # 0: forest
                   # 1: crops
                   
# << SELECT >>
flag_standardize=0;  # 0: no (standardization before input to PCA and projections)
                     # 1: yes, standardize 
                   
# << SELECT >>
flag_correlation=0;  # 0: no
                     # 1: yes
                  
# redefine some flags for luh2 testing
# if flag_analysis == 1:
    
#     flag_y1=0
#     flag_len=1

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
standardize_opts = ['no',
                    'yes']
correlation_opts = ['no',
                    'yes']

tres = seasons[flag_tres]
obs = obs_types[flag_obs_type]
y1 = start_years[flag_y1]
length = lengths[flag_len]
var = variables[flag_var]
analysis = analyses[flag_analysis]
landcover = landcover_types[flag_landcover]
standardize = standardize_opts[flag_standardize]
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


#%%==============================================================================
# mod + obs + luh2 data 
#================================================================================


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
        
    # mmm ensembles
    mod_data[obs]['mmm'] = {}  
    
    for exp in ['historical','hist-noLu','lu']:
        
        data = []
        
        for mod in models:
            
            data.append(mod_data[obs][mod][exp])

        mod_data[obs]['mmm'][exp] = da_ensembler(data)

    # mmm k-fold ensembles (leave 1 out)
    for mod in models:

        mod_data[obs]['mmm_no_'+mod] = {}

        for exp in ['historical','hist-noLu','lu']:

            alt_data = []
            cf_mod_list = cp.deepcopy(models)
            cf_mod_list.remove(mod)

            for m in cf_mod_list:

                alt_data.append(mod_data[obs][m][exp])

            mod_data[obs]['mmm_no_'+mod][exp] = da_ensembler(alt_data)
        
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
                                         var='cell_area',
                                         obs=obs)
        
            if correlation == "yes":
                
                # adjust data to be temporally centered and unit variance (check xarray for this)
                    # also adjust model data
                print("not ready yet")
                
            else:
                
                luh2_data[obs][lc] = luh2_data[obs][lc].where(ar6_land==1)
            
    # standardize data
    if standardize == "yes":
        
        for exp in ['historical','hist-noLu','lu']:
            
            mod_data[obs]['mmm'][exp] = standard_data(mod_data[obs]['mmm'][exp])
        
            for mod in models:
                
                mod_data[obs]['mmm_no_'+mod][exp] = standard_data(mod_data[obs]['mmm_no_'+mod][exp])
    
        obs_data[obs] = standard_data(obs_data[obs])
        
        for lc in landcover_types:
        
            luh2_data[obs][lc] = standard_data(luh2_data[obs][lc])
          
            
#%%==============================================================================
# mutual mask of data 
#================================================================================


mod_msk = {}

for obs in obs_types:    
    
    mod_msk[obs] = {}
    
    for lc in landcover_types:
    
        exp = 'lu'
        arr1 = cp.deepcopy(mod_data[obs]['mmm'][exp].isel(time=0)).drop('time')
        
        if analysis == "models":
            
            arr2 = cp.deepcopy(obs_data[obs].isel(time=0)).drop('time')
            
        elif analysis == "luh2":
            
            arr2 = cp.deepcopy(luh2_data[obs][lc].isel(time=0)).drop('time')
            
        arr1 = arr1.fillna(-999)
        arr2 = arr2.fillna(-999)
        
        alt_msk = xr.where(arr2!=-999,1,0)
        mod_msk[obs][lc] = xr.where(arr1!=-999,1,0).where(alt_msk==1).fillna(0)
    
                   
#%%==============================================================================
# pca
#================================================================================


solver_dict = {}
eof_dict = {}
principal_components = {}
pseudo_principal_components = {}
weighted_arr = {}

for obs in obs_types:
    
    solver_dict[obs] = {}
    eof_dict[obs] = {}
    principal_components[obs] = {}
    pseudo_principal_components[obs] = {}
    
    weights = np.cos(np.deg2rad(mod_data[obs]['mmm']['lu'].lat))
    weighted_arr[obs] = xr.zeros_like(mod_data[obs]['mmm']['lu']).isel(time=0)    
    x = weighted_arr[obs].coords['lon']
    
    for y in weighted_arr[obs].lat.values:
        
        weighted_arr[obs].loc[dict(lon=x,lat=y)] = weights.sel(lat=y).item()

if analysis == "models":
    
    for obs in obs_types:
            
        for exp in ['historical','hist-noLu','lu']:
        
            solver_dict[obs][exp] = Eof(mod_data[obs]['mmm'][exp].where(mod_msk[obs][lc]==1),
                                        weights=weighted_arr[obs]) 
            eof_dict[obs][exp] = solver_dict[obs][exp].eofs(neofs=1)
            principal_components[obs][exp] = solver_dict[obs][exp].pcs(npcs=1)
            pseudo_principal_components[obs][exp] = solver_dict[obs][exp].projectField(obs_data[obs].where(mod_msk[obs][lc]==1),
                                                                                       neofs=1)
            
        solver_dict[obs][obs] = Eof(obs_data[obs].where(mod_msk[obs][lc]==1),
                                    weights=weighted_arr[obs])
        eof_dict[obs][obs] = solver_dict[obs][obs].eofs(neofs=1)
        

        
elif analysis == "luh2":
    
    for obs in obs_types:
            
        for lc in landcover_types:
    
            solver_dict[obs][lc] = Eof(luh2_data[obs][lc].where(mod_msk[obs][lc]==1),
                                       weights=weighted_arr[obs])
            eof_dict[obs][lc] = solver_dict[obs][lc].eofs(neofs=1)
            principal_components[obs][lc] = solver_dict[obs][lc].pcs(npcs=1)
            pseudo_principal_components[obs][lc] = {}
            pseudo_principal_components[obs][lc]['mmm'] = solver_dict[obs][lc].projectField(mod_data[obs]['mmm']['lu'].where(mod_msk[obs][lc]==1),
                                                                                            neofs=1)

            for mod in models:

                solver_dict[obs]['mmm_no_'+mod] = Eof(mod_data[obs]['mmm_no_'+mod]['lu'].where(mod_msk[obs][lc]==1),
                                                      weights=weighted_arr[obs])
                eof_dict[obs]['mmm_no_'+mod] = solver_dict[obs]['mmm_no_'+mod].eofs(neofs=1)    
                pseudo_principal_components[obs][lc]['mmm_no_'+mod] = solver_dict[obs][lc].projectField(mod_data[obs]['mmm_no_'+mod]['lu'].where(mod_msk[obs][lc]==1),
                                                                                                        neofs=1)   
                pseudo_principal_components[obs][lc][mod] = solver_dict[obs][lc].projectField(mod_data[obs][mod]['lu'].where(mod_msk[obs][lc]==1),
                                                                                              neofs=1)                                               
                                    

#%%=============================================================================
# plotting        
#===============================================================================


if analysis == "models":

    # projecting obs onto hist and hist-nolu fingerprints
    pca_plot(eof_dict,
             principal_components,
             pseudo_principal_components,
             exps_start,
             obs_types,
             outDIR)
    
elif analysis == "luh2":

    # plotting pseudo-principal_components from projecting variations of "lu" onto luh2 fingerprints    
    pca_plot_luh2(eof_dict,
                  principal_components,
                  pseudo_principal_components,
                  models,
                  landcover_types,
                  obs_types,
                  outDIR)
    
    # separate figure for plotting variations of "lu" fingerprints

    

                  

            
    
                      
        
        
             
        
        
