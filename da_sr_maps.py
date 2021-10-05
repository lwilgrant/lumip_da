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
def map_subroutine(models,
                   mapDIR,
                   lulcc,
                   y1,
                   measure,
                   freq,
                   thresh):
    # map data
    maps_files = {}
    maps = {}
    gridarea = {}
    ar6_regs = {}
    ar6_land = {}

    for mod in models:
        
        maps_files[mod] = {}
        maps[mod] = {}

        # maps of lulcc data
        os.chdir(mapDIR)
        gridarea[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
        
        i = 0
        for lu in lulcc:
            
            for file in [file for file in sorted(os.listdir(mapDIR))
                        if mod in file
                        and lu in file
                        and str(y1) in file
                        and 'absolute_change' in file]:

                    maps_files[mod][lu] = file
                    
                    # get ar6 from lat/lon of sample model res luh2 file
                    if i == 0:
                        template = xr.open_dataset(file,decode_times=False).cell_area.isel(time=0).squeeze(drop=True)
                        if 'height' in template.coords:
                            template = template.drop('height')
                        ar6_regs[mod] = ar6_mask(template)
                        ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)
                    i += 1
                

            if measure == 'absolute_change':
                
                maps[mod][lu] = nc_read(maps_files[mod][lu],
                                        y1,
                                        var='cell_area',
                                        mod=True,
                                        freq=freq)
                
            elif measure == 'area_change':
                
                maps[mod][lu] = nc_read(maps_files[mod][lu],
                                        y1,
                                        var='cell_area',
                                        mod=True,
                                        freq=freq) * gridarea[mod] / 1e6
                if thresh < 0: # forest
                
                    da = xr.where(maps[mod][lu] <= thresh,1,0).sum(dim='time')
                    maps[mod][lu] = xr.where(da >= 1,1,0)
                    
                elif thresh > 0: # crops + urban
                
                    da = xr.where(maps[mod][lu] >= thresh,1,0).sum(dim='time')
                    maps[mod][lu] = xr.where(da >= 1,1,0)
                    
            elif measure == 'all_pixels':
                
                maps[mod][lu] = ar6_land[mod]
            
    return maps,ar6_regs,ar6_land
