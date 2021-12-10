#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This subroutine script generates:
    # model ensemble means for OF (mod_ens)
    # model t-series as ar6-weighted matrices of rows for tsteps and columns for ar6 regions (mod_ts_ens)
        # axis 0 for realisations (realisations x tstep_rows x ar6_columns)
        # these t-series are for box plot data; not used for OF 
        
# To check:
    # check that maps for all cases have 1's for desired locations and 0 otherwise:
        # absoute change, area change and all pixels versions
        # make sure it is working for different observation types


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from copy import deepcopy
from lu_funcs import *


#%%============================================================================

# mod data
def ensemble_subroutine(modDIR,
                        models,
                        exps,
                        var,
                        lu_techn,
                        y1,
                        freq,
                        mod_files):

    os.chdir(modDIR)
    mod_data = {}
    mod_ens = {}
    
    for mod in models:
        
        mod_data[mod] = {}
        mod_ens[mod] = {}
        
        for exp in exps:
            
            mod_data[mod][exp] = []    
            
            for file in mod_files[mod][exp]:
            
                da = nc_read(file,
                             y1,
                             var,
                             freq=freq)
                mod_data[mod][exp].append(da)
            
            mod_ens[mod][exp] = da_ensembler(mod_data[mod][exp])
        
        if lu_techn == 'mean':
        
            mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']
        
    return mod_ens
# %%
