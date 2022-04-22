#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This subroutine script generates data arrays of model data


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from copy import deepcopy
from icv_funcs import *


#%%============================================================================

# mod data
def ensemble_subroutine(
    modDIR,
    models,
    exps,
    var,
    y1,
    freq,
    fp_files,
):

    os.chdir(modDIR)
    mod_data = {}
    mod_ens = {}
    
    i = 0
    
    for mod in models:
        
        mod_data[mod] = {}
        mod_ens[mod] = {}
        
        for exp in exps:
            
            mod_data[mod][exp] = []    
            
            for file in fp_files[mod][exp]:
            
                da = nc_read(
                    file,
                    y1,
                    var,
                    freq=freq
                )
                mod_data[mod][exp].append(da)
                
            concat_dim = np.arange(len(mod_data[mod][exp]))
            aligned = xr.concat(mod_data[mod][exp],dim=concat_dim)
            mod_data[mod][exp] = aligned
            mod_data[mod][exp] = mod_data[mod][exp].rename({'concat_dim':'rls'})

    return mod_data
# %%
