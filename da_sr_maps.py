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
                   lulcc,
                   obs_types,
                   grid,
                   y1,
                   measure,
                   freq,
                   thresh):
    
    # map data
    os.chdir(mapDIR)
    grid_area = {}
    maps = {}
    ar6_regs = {}
    ar6_land = {}
    
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
            
            grid_area[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
            i = 0
            
            for lu in lulcc:
                                    
                # get ar6 from lat/lon of sample model res luh2 file
                if i == 0:
                    template = xr.open_dataset(map_files[mod][lu],decode_times=False).cell_area.isel(time=0).squeeze(drop=True)
                    if 'height' in template.coords:
                        template = template.drop('height')
                    ar6_regs[mod] = ar6_mask(template)
                    ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)
                i += 1
                    
                if measure == 'absolute_change':
                    
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
            
    return maps,ar6_regs,ar6_land
