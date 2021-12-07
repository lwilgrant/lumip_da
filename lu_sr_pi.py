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
    # picontrol series for detection and attribution

# nov 29 note: moved lines 159-201 back one indent to get proper min_pi_samp

#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from copy import deepcopy
from lu_funcs import *


#%%============================================================================

# pi data
def picontrol_subroutine(piDIR,
                         pi_files,
                         models,
                         var,
                         y1,
                         freq,
                         maps):

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
                         freq=freq)
                    
            pi_data[mod].append(da) # 1-D pi array to go into pi-chunks for DA
            
        concat_dim = np.arange(len(pi_data[mod]))
        aligned = xr.concat(pi_data[mod],dim=concat_dim)
        pi_data[mod] = aligned
        pi_data[mod] = pi_data[mod].rename({'concat_dim':'rls'})
                # da = da.rename({'latitude':'lat','longitude':'lon'})
    return pi_data

# %%
