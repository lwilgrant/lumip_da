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
from icv_funcs import *

#%%============================================================================
def map_subroutine(
    models,
    mapDIR,
    sfDIR,
    grid_files,
):
    
    # map data
    os.chdir(mapDIR)
    ar6_regs = {} # per model, data array of ar6 regions painted with integers from "continents[c]"
    cnt_regs = {}

    for mod in models:
        
        os.chdir(mapDIR)
        template = xr.open_dataset(grid_files[mod],decode_times=False)['cell_area']
        if 'height' in template.coords:
            template = template.drop('height')
        ar6_regs[mod] = ar6_mask(template)
        cnt_regs[mod] = cnt_mask(sfDIR,template)

    return ar6_regs,cnt_regs

# %%
