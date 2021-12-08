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
from pca_funcs import *


#%%============================================================================

# pi data
def picontrol_subroutine(piDIR,
                         pi_files,
                         models,
                         flag_temp_center,
                         flag_standardize,
                         var,
                         y1,
                         freq,
                         ar6_land):

    # pi data
    os.chdir(piDIR)
    pi_data = {}
        
    for mod in  models:
        
        pi_data[mod] = []
        
        for file in pi_files[mod]:
            
            # mod data and coords for ar6 mask
            da = nc_read(file,
                         y1,
                         var,
                         flag_temp_center,
                         flag_standardize,
                         freq=freq)
                    
            pi_data[mod].append(da.where(ar6_land[mod]==1)) # 1-D pi array to go into pi-chunks for DA
            
        concat_dim = np.arange(len(pi_data[mod]))
        aligned = xr.concat(pi_data[mod],dim=concat_dim)
        pi_data[mod] = aligned
        pi_data[mod] = pi_data[mod].rename({'concat_dim':'rls'})
                # da = da.rename({'latitude':'lat','longitude':'lon'})
    return pi_data

# %%
