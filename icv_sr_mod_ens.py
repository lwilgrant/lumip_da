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
from da_funcs import *


#%%============================================================================

# mod data
def ensemble_subroutine(
    modDIR,
    models,
    exps,
    var,
    lu_techn,
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
            
            mod_ens[mod][exp] = da_ensembler(mod_data[mod][exp])
        
        if lu_techn == 'mean':
        
            mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']

    return mod_ens
# %%
