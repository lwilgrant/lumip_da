#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This subroutine script generates maps of:
    # LUH2 landcover transitions
    # AR6 land maps at model resolution
    # AR6 regions at model resolution


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================
def map_subroutine(map_files,
                   models,
                   mapDIR,
                   sfDIR,
                   lulcc,
                   obs_types,
                   grid,
                   agg,
                   weight,
                   continents,
                   y1,
                   measure,
                   freq,
                   thresh):
    
    # map data
    os.chdir(mapDIR)
    grid_area = {}
    maps = {}
    ar6_regs = {} # per model, data array of ar6 regions painted with integers from "continents[c]"
    ar6_areas = {} # rerun per model, contains areas of each ar6 region for computing model weights for analyses with "agg = ar6"
    ar6_wts = {} # per model, relative weights per continent (largest ar6 = 1)
    ar6_land = {} # per model, land pixels
    cnt_regs = {} # per model, data array of continents painted with integers from "continents[c]"
    cnt_areas = {} # rerun per model, areas of each continent for computing model weights for global scale analysis with "agg = continental"
    cnt_wts = {} # per model, continental weights (asia = 1)
    
    if grid == 'obs':
        
        for obs in obs_types:
            
            maps[obs] = {}
            grid_area[obs] = xr.open_dataset(obs+'_gridarea.nc',decode_times=False)['cell_area']
            if obs == 'berkley_earth':
                grid_area[obs] = grid_area[obs].rename({'latitude':'lat','longitude':'lon'})
            i = 0
            
            for lu in lulcc:
                                    
                # get ar6 from lat/lon of sample model res luh2 file
                if i == 0:
                    template = xr.open_dataset(map_files[obs][lu],decode_times=False).cell_area.isel(time=0).squeeze(drop=True)
                    if 'height' in template.coords:
                        template = template.drop('height')
                    if obs == 'berkley_earth':
                        template = template.rename({'latitude':'lat','longitude':'lon'})
                    ar6_regs[obs] = ar6_mask(template)
                    ar6_land[obs] = xr.where(ar6_regs[obs]>=0,1,0)
                i += 1
                    
                if measure == 'absolute_change':
                    
                    maps[obs][lu] = nc_read(map_files[obs][lu],
                                            y1,
                                            var='cell_area',
                                            obs=obs,
                                            freq=freq)
                    
                    if thresh < 0: # forest
                    
                        da = xr.where(maps[obs][lu] < thresh,1,0).sum(dim='time')
                        maps[obs][lu] = xr.where(da >= 1,1,0)
                        
                    elif thresh > 0: # crops + urban
                    
                        da = xr.where(maps[obs][lu] > thresh,1,0).sum(dim='time')
                        maps[obs][lu] = xr.where(da >= 1,1,0)
                    
                elif measure == 'area_change':
                    
                    maps[obs][lu] = nc_read(map_files[obs][lu],
                                            y1,
                                            var='cell_area',
                                            obs=obs,
                                            freq=freq) * grid_area[obs] / 1e6
                    
                    if thresh < 0: # forest
                    
                        da = xr.where(maps[obs][lu] < thresh,1,0).sum(dim='time')
                        maps[obs][lu] = xr.where(da >= 1,1,0)
                        
                    elif thresh > 0: # crops + urban
                    
                        da = xr.where(maps[obs][lu] > thresh,1,0).sum(dim='time')
                        maps[obs][lu] = xr.where(da >= 1,1,0)
                        
                elif measure == 'all_pixels':
                    
                    maps[obs] = ar6_land[obs]
    
    elif grid == 'model':

        for mod in models:
            
            os.chdir(mapDIR)
            grid_area[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
            cnt_wts[mod] = {}
            ar6_wts[mod] = {}
            i = 0
            
            for lu in lulcc:
                                    
                # get ar6 and continents from lat/lon of sample model res luh2 file (only usnig luh2 for dimensions, not trusting data)
                if i == 0:
                    
                    template = xr.open_dataset(map_files[mod][lu],decode_times=False).cell_area.isel(time=0).squeeze(drop=True)
                    if 'height' in template.coords:
                        template = template.drop('height')
                    ar6_regs[mod] = ar6_mask(template)
                    ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)                        
                    if agg == 'ar6': # ar6
                        for c in continents.keys():
                            ar6_areas[c] = {}
                            ar6_wts[mod][c] = {}
                            for i in continents[c]:
                                ar6_areas[c][i] = grid_area[mod].where(ar6_regs[mod]==i).sum(dim=('lat','lon'))
                            max_area = max(ar6_areas[c].values())
                            for i in continents[c]:
                                ar6_wts[mod][c][i] = ar6_areas[c][i]/max_area
                                
                    elif agg == 'continental': # continents
                        cnt_regs[mod] = cnt_mask(sfDIR, # changes directory, correct elsewhere lower
                                                 template)
                        for c in continents.keys():
                            cnt_areas[c] = grid_area[mod].where(cnt_regs[mod]==continents[c]).sum(dim=('lat','lon'))
                        max_area = max(cnt_areas.values())
                        for c in continents.keys():
                            cnt_wts[mod][c] = cnt_areas[c]/max_area

                i += 1
                    
                if measure == 'absolute_change':
                    
                    os.chdir(mapDIR)
                    maps[mod][lu] = nc_read(map_files[mod][lu],
                                            y1,
                                            var='cell_area',
                                            freq=freq)
                    
                    if thresh < 0: # forest
                    
                        da = xr.where(maps[mod][lu] < thresh,1,0).sum(dim='time')
                        maps[mod][lu] = xr.where(da >= 1,1,0)
                        
                    elif thresh > 0: # crops + urban
                    
                        da = xr.where(maps[mod][lu] > thresh,1,0).sum(dim='time')
                        maps[mod][lu] = xr.where(da >= 1,1,0)
                    
                elif measure == 'area_change':
                    
                    os.chdir(mapDIR)
                    maps[mod][lu] = nc_read(map_files[mod][lu],
                                            y1,
                                            var='cell_area',
                                            freq=freq) * grid_area[mod] / 1e6
                    
                    if thresh < 0: # forest
                    
                        da = xr.where(maps[mod][lu] < thresh,1,0).sum(dim='time')
                        maps[mod][lu] = xr.where(da >= 1,1,0)
                        
                    elif thresh > 0: # crops + urban
                    
                        da = xr.where(maps[mod][lu] > thresh,1,0).sum(dim='time')
                        maps[mod][lu] = xr.where(da >= 1,1,0)
                        
                elif measure == 'all_pixels':
                    
                    maps[mod] = ar6_land[mod]
            
    return maps,ar6_regs,ar6_areas,ar6_wts,ar6_land,cnt_regs,cnt_areas,cnt_wts,grid_area

# %%
