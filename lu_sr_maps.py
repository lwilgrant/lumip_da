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
from lu_funcs import *


#%%============================================================================
def map_subroutine(map_files,
                   models,
                   mapDIR,
                   lulcc,
                   y1,
                   measure,
                   freq):
    
    # map data
    os.chdir(mapDIR)
    grid_area = {}
    maps = {}
    ar6_regs = {}
    ar6_land = {}

    for mod in models:
        
        maps[mod] = {}
        
        grid_area[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
        i = 0
        
        for lu in lulcc:
                                
            # get ar6 from lat/lon of sample model res luh2 file
            if i == 0:
                template = xr.open_dataset(map_files[mod][lu],decode_times=False)[lu]
                template = template.isel(time=0).squeeze(drop=True)
                if 'height' in template.coords:
                    template = template.drop('height')
                ar6_regs[mod] = ar6_mask(template)
                ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)
            i += 1
                
            if measure == 'absolute_change':
                
                maps[mod][lu] = nc_read(map_files[mod][lu],
                                        y1,
                                        var=lu,
                                        freq=freq)
                
            elif measure == 'area_change':
                
                maps[mod][lu] = nc_read(map_files[mod][lu],
                                        y1,
                                        var=lu,
                                        freq=freq) / 100 * grid_area[mod] / 1e6
                
            maps[mod][lu] = maps[mod][lu].where(ar6_land[mod] ==1)
            
    return maps,ar6_regs,ar6_land

# %%
