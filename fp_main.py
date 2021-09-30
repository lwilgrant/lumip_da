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

# tidy up functions by cutting repeated colorbar related code from numerous plotting functions

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
from scipy import stats as sts
from eofs.xarray import Eof


#%%==============================================================================
# path
#================================================================================


curDIR = '/home/luke/documents/lumip/d_a/'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'data/obs/final')
modDIR = os.path.join(curDIR, 'data/mod/obs_grid')
# piDIR = os.path.join(curDIR, 'data/pi/final')
mapDIR = os.path.join(curDIR, 'data/map/final')
outDIR = os.path.join(curDIR, 'figures_pca')

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
flag_standardize=0;  # 0: no (standardization before input to PCA and projections)
                     # 1: yes, standardize 
                     
# << SELECT >>
flag_scale=3;         # 0: global
                      # 1: latitudinal
                      # 2: continental
                      # 3: ar6 regions
                   
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
scale_opts = ['global',
              'latitudinal',
              'continental',
              'ar6']

tres = seasons[flag_tres]
obs = obs_types[flag_obs_type]
y1 = start_years[flag_y1]
length = lengths[flag_len]
var = variables[flag_var]
analysis = analyses[flag_analysis]
landcover = landcover_types[flag_landcover]
standardize = standardize_opts[flag_standardize]
correlation = correlation_opts[flag_correlation]
scale = scale_opts[flag_scale]

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

continents = {}
continents['North America'] = [1,2,3,4,5,6,7]
continents['South America'] = [9,10,11,12,13,14,15]
continents['Europe'] = [16,17,18,19]
continents['Asia'] = [29,30,32,33,34,35,37,38]
continents['Africa'] = [21,22,23,24,25,26]
continents['Australia'] = [39,40,41,42]

continent_names = []
for c in continents.keys():
    continent_names.append(c)

labels = {}
labels['North America'] = ['WNA','CNA','ENA','NCA','SCA']
labels['South America'] = ['NWS','NSA','NES','SAM','SES']
labels['Europe'] = ['NEU','WCE','EEU','MED']
labels['Asia'] = ['WSB','ESB','WCA','EAS','SAS']
labels['Africa'] = ['WAF','CAF','NEAF','SEAF','SWAF','ESAF']
labels['Australia'] = ['NAU','CAU','EAU','SAU']

lat_ranges = {}
lat_ranges['boreal'] = slice(51.5,89.5)
lat_ranges['tropics'] = slice(-23.5,23.5)
lat_ranges['temperate_south'] = slice(-50.5,-24.5)
lat_ranges['temperate_north'] = slice(24.5,50.5)


#%%==============================================================================
# mod + obs + luh2 data 
#================================================================================


mod_files = {}
mod_data = {}
obs_data = {}
luh2_data = {}
ar6 = {}

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
                ar6[obs] = ar6_regs
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
        
# projection of obs onto mmm hist vs mmm histnolu fps
if analysis == "models":
    
    for obs in obs_types:
        
        # global pca
        if scale == 'global':
            
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

        # latitudinal pca            
        elif scale == 'latitudinal':
            
            for exp in ['historical', 'hist-noLu', 'lu']:
                
                solver_dict[obs][exp] = {}
                eof_dict[obs][exp] = {}
                principal_components[obs][exp] = {}
                pseudo_principal_components[obs][exp] = {}
                
                for ltr in lat_ranges.keys():
                    
                    mod_slice = mod_data[obs]['mmm'][exp].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                    solver_dict[obs][exp][ltr] = Eof(mod_slice,
                                                     weights=weighted_arr[obs].sel(lat=lat_ranges[ltr]))
                    eof_dict[obs][exp][ltr] = solver_dict[obs][exp][ltr].eofs(neofs=2)
                    principal_components[obs][exp][ltr] = solver_dict[obs][exp][ltr].pcs(npcs=2)
                    obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                    obs_slice = obs_slice.sel(lat=lat_ranges[ltr])
                    pseudo_principal_components[obs][exp][ltr] = solver_dict[obs][exp][ltr].projectField(obs_slice,
                                                                                                         neofs=2)
                    
            solver_dict[obs][obs] = {}
            eof_dict[obs][obs] = {}
            
            for ltr in lat_ranges.keys():        
                
                obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                obs_slice = obs_slice.sel(lat=lat_ranges[ltr])
                solver_dict[obs][obs][ltr] = Eof(obs_slice,
                                                 weights=weighted_arr[obs].sel(lat=lat_ranges[ltr]))
                eof_dict[obs][obs][ltr] = solver_dict[obs][obs][ltr].eofs(neofs=2)

        # continental pca            
        elif scale == 'continental':
            
            for exp in ['historical', 'hist-noLu', 'lu']:
                
                solver_dict[obs][exp] = {}
                eof_dict[obs][exp] = {}
                principal_components[obs][exp] = {}
                pseudo_principal_components[obs][exp] = {}
                
                for c in continents.keys():                
                    
                    continent = ar6[obs].where(ar6[obs].isin(continents[c]))
                    mod_slice = mod_data[obs]['mmm'][exp].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(continent > 0)
                    solver_dict[obs][exp][c] = Eof(mod_slice,
                                                   weights=weighted_arr[obs].where(continent > 0))
                    eof_dict[obs][exp][c] = solver_dict[obs][exp][c].eofs(neofs=1)
                    principal_components[obs][exp][c] = solver_dict[obs][exp][c].pcs(npcs=1)
                    obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                    obs_slice = obs_slice.where(continent > 0)
                    pseudo_principal_components[obs][exp][c] = solver_dict[obs][exp][c].projectField(obs_slice,
                                                                                                     neofs=1)
                    
            solver_dict[obs][obs] = {}
            eof_dict[obs][obs] = {}
            
            for c in continents.keys():        
                
                continent = ar6[obs].where(ar6[obs].isin(continents[c]))
                obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                obs_slice = obs_slice.where(continent > 0)
                solver_dict[obs][obs][c] = Eof(obs_slice,
                                               weights=weighted_arr[obs].where(continent > 0))
                eof_dict[obs][obs][c] = solver_dict[obs][obs][c].eofs(neofs=1)
                           
        # ar6 pca            
        elif scale == 'ar6':
            
            for exp in ['historical', 'hist-noLu', 'lu']:
                
                solver_dict[obs][exp] = {}
                eof_dict[obs][exp] = {}
                principal_components[obs][exp] = {}
                pseudo_principal_components[obs][exp] = {}
                
                # for c in continents.keys():
                c = "South America"
                    
                for i in continents[c]:
                
                    mod_slice = mod_data[obs]['mmm'][exp].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(ar6[obs] == i)
                    solver_dict[obs][exp][i] = Eof(mod_slice,
                                                        weights=weighted_arr[obs].where(ar6[obs] == i))
                    eof_dict[obs][exp][i] = solver_dict[obs][exp][i].eofs(neofs=1)
                    principal_components[obs][exp][i] = solver_dict[obs][exp][i].pcs(npcs=1)
                    obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                    obs_slice = obs_slice.where(ar6[obs] == i)
                    pseudo_principal_components[obs][exp][i] = solver_dict[obs][exp][i].projectField(obs_slice,
                                                                                                            neofs=1)
            solver_dict[obs][obs] = {}
            eof_dict[obs][obs] = {}
            
            # for c in continents.keys():
            c = "South America"
                
            for i in continents[c]:        
            
                obs_slice = obs_data[obs].where(mod_msk[obs][lc] == 1)
                obs_slice = obs_slice.where(ar6[obs] == i)
                solver_dict[obs][obs][i] = Eof(obs_slice,
                                                weights=weighted_arr[obs].where(ar6[obs] == i))
                eof_dict[obs][obs][i] = solver_dict[obs][obs][i].eofs(neofs=1)

# projection of mmm lu onto luh2 fps   
elif analysis == "luh2":
    
    for obs in obs_types:
        
        if scale == "global":
            
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

        # latitudinal pca            
        elif scale == 'latitudinal':
                
            for lc in landcover_types:
                    
                solver_dict[obs][lc] = {}
                eof_dict[obs][lc] = {}
                principal_components[obs][lc] = {}
                pseudo_principal_components[obs][lc] = {}
                
                for ltr in lat_ranges.keys():
                    
                    luh2_slice = luh2_data[obs][lc].where(mod_msk[obs][lc]==1)
                    luh2_slice = luh2_slice.sel(lat=lat_ranges[ltr])
                    solver_dict[obs][lc][ltr] = Eof(luh2_slice,
                                                    weights=weighted_arr[obs].sel(lat=lat_ranges[ltr]))
                    eof_dict[obs][lc][ltr] = solver_dict[obs][lc][ltr].eofs(neofs=1)
                    principal_components[obs][lc][ltr] = solver_dict[obs][lc][ltr].pcs(npcs=1)
                    pseudo_principal_components[obs][lc][ltr] = {}
                    mod_slice = mod_data[obs]['mmm']['lu'].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                    pseudo_principal_components[obs][lc][ltr]['mmm'] = solver_dict[obs][lc][ltr].projectField(mod_slice,
                                                                                                              neofs=1)

                    for mod in models:

                        mod_slice = mod_data[obs]['mmm_no_'+mod]['lu'].where(mod_msk[obs][lc]==1)
                        mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                        pseudo_principal_components[obs][lc][ltr]['mmm_no_'+mod] = solver_dict[obs][lc][ltr].projectField(mod_slice,
                                                                                                                          neofs=1)   
                        mod_slice = mod_data[obs][mod]['lu'].where(mod_msk[obs][lc]==1)
                        mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                        pseudo_principal_components[obs][lc][ltr][mod] = solver_dict[obs][lc][ltr].projectField(mod_slice,
                                                                                                                neofs=1)     
                
                
        # continental pca            
        elif scale == 'continental':
            
            for lc in landcover_types:
                    
                solver_dict[obs][lc] = {}
                eof_dict[obs][lc] = {}
                principal_components[obs][lc] = {}
                pseudo_principal_components[obs][lc] = {}
                continent = ar6[obs].where(ar6[obs].isin(continents[lc]))
                
                for c in continents.keys():
                    
                    luh2_slice = luh2_data[obs][lc].where(mod_msk[obs][lc]==1)
                    luh2_slice = luh2_slice.where(continent > 0)
                    solver_dict[obs][lc][c] = Eof(luh2_slice,
                                                  weights=weighted_arr[obs])
                    eof_dict[obs][lc][c] = solver_dict[obs][lc][c].eofs(neofs=1)
                    principal_components[obs][lc][c] = solver_dict[obs][lc][c].pcs(npcs=1)
                    pseudo_principal_components[obs][lc][c] = {}
                    mod_slice = mod_data[obs]['mmm']['lu'].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(continent > 0)
                    pseudo_principal_components[obs][lc][c]['mmm'] = solver_dict[obs][lc][c].projectField(mod_slice,
                                                                                                          neofs=1)

                    for mod in models:

                        mod_slice = mod_data[obs]['mmm_no_'+mod]['lu'].where(mod_msk[obs][lc]==1)
                        mod_slice = mod_slice.where(continent > 0)
                        pseudo_principal_components[obs][lc][c]['mmm_no_'+mod] = solver_dict[obs][lc][c].projectField(mod_slice,
                                                                                                                      neofs=1)   
                        mod_slice = mod_data[obs][mod]['lu'].where(mod_msk[obs][lc]==1)
                        mod_slice = mod_slice.where(continent > 0)
                        pseudo_principal_components[obs][lc][c][mod] = solver_dict[obs][lc][c].projectField(mod_slice,
                                                                                                              neofs=1)    
                           
        # ar6 pca            
        elif scale == 'ar6':
            
            for lc in landcover_types:
                    
                solver_dict[obs][lc] = {}
                eof_dict[obs][lc] = {}
                principal_components[obs][lc] = {}
                pseudo_principal_components[obs][lc] = {}
                
                # for c in continents.keys():
                c = "South America"
                for i in continents[c]:
                    
                    luh2_slice = luh2_data[obs][lc].where(mod_msk[obs][lc]==1)
                    luh2_slice = luh2_slice.where(ar6[obs] == i)
                    solver_dict[obs][lc][i] = Eof(luh2_slice,
                                                    weights=weighted_arr[obs])
                    eof_dict[obs][lc][i] = solver_dict[obs][lc][i].eofs(neofs=1)
                    principal_components[obs][lc][i] = solver_dict[obs][lc][i].pcs(npcs=1)
                    pseudo_principal_components[obs][lc][i] = {}
                    mod_slice = mod_data[obs]['mmm']['lu'].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(ar6[obs] == i)
                    pseudo_principal_components[obs][lc][i]['mmm'] = solver_dict[obs][lc][i].projectField(mod_slice,
                                                                                                            neofs=1)
                        
            solver_dict[obs][obs] = {}
            eof_dict[obs][obs] = {}
            
            # for c in continents.keys():
            
            for i in continents[c]:        
            
                for mod in models:

                    mod_slice = mod_data[obs]['mmm_no_'+mod]['lu'].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(ar6[obs] == i)
                    pseudo_principal_components[obs][lc][i]['mmm_no_'+mod] = solver_dict[obs][lc][i].projectField(mod_slice,
                                                                                                                    neofs=1)   
                    mod_slice = mod_data[obs][mod]['lu'].where(mod_msk[obs][lc]==1)
                    mod_slice = mod_slice.where(ar6[obs] == i)
                    pseudo_principal_components[obs][lc][i][mod] = solver_dict[obs][lc][i].projectField(mod_slice,
                                                                                                        neofs=1) 

# from scipy import stats as sts

# testhist = pseudo_principal_components[obs]['historical'][c] - pseudo_principal_components[obs]['historical'][c].mean(dim='time')
# testhistnolu = pseudo_principal_components[obs]['hist-noLu'][c] - pseudo_principal_components[obs]['hist-noLu'][c].mean(dim='time')

# testhist = np.squeeze(testhist.drop('mode').values)
# testhistnolu = np.squeeze(testhistnolu.drop('mode').values)

# x = np.arange(1,len(testhist)+1)

# slope_hist,_,_,_,_ = sts.linregress(x,testhist)
# slope_histnolu,_,_,_,_ = sts.linregress(x,testhistnolu)



#%%=============================================================================
# plotting        
#===============================================================================


if analysis == "models":


# scale_opts = ['global',
#               'latitudinal',
#               'continental',
#               'ar6']
    if scale == 'global':
        
        # projecting obs onto hist and hist-nolu fingerprints
        pca_plot(eof_dict,
                principal_components,
                pseudo_principal_components,
                exps_start,
                obs_types,
                outDIR)

    elif scale == 'latitudinal':
        
        # projecting obs onto hist and hist-nolu fingerprints
        pca_plot_latitudinal(eof_dict,
                             principal_components,
                             pseudo_principal_components,
                             exps_start,
                             lat_ranges,
                             obs_types,
                             outDIR)
        
    elif scale == 'continental':
        
        # projecting obs onto hist and hist-nolu fingerprints
        pca_plot_continental(eof_dict,
                             principal_components,
                             pseudo_principal_components,
                             exps_start,
                             continents,
                             obs_types,
                             outDIR)
        
    elif scale == 'ar6':
        
        # projecting NOT YET READY
        pca_plot_ar6(eof_dict,
                     principal_components,
                     pseudo_principal_components,
                     exps_start,
                     continents,
                     obs_types,
                     outDIR)
    
elif analysis == "luh2":

    if scale == 'global':
        # plotting pseudo-principal_components from projecting variations of "lu" onto luh2 fingerprints    
        pca_plot_luh2(eof_dict,
                    principal_components,
                    pseudo_principal_components,
                    models,
                    landcover_types,
                    obs_types,
                    outDIR)
    
    # separate figure for plotting variations of "lu" fingerprints

    

                  

            
    
                      
        
        
             
        
        
