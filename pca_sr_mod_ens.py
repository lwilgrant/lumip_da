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

# mod data
def ensemble_subroutine(modDIR,
                        models,
                        exps,
                        flag_temp_center,
                        flag_standardize,
                        mod_files,
                        var,
                        lu_techn,
                        y1,
                        freq,
                        ar6_land):

    os.chdir(modDIR)
    mod_data = {}
    mod_rls = {}
    mod_ens = {}
    
    for mod in models:
        
        mod_data[mod] = {}
        mod_rls[mod] = {}
        mod_ens[mod] = {}
        
        for exp in exps:
            
            mod_data[mod][exp] = []  
            mod_rls[mod][exp] = []  
            
            for file in mod_files[mod][exp]:
            
                da_data = nc_read(
                    file,
                    y1,
                    var,
                    flag_temp_center,
                    flag_standardize=0,
                    freq=freq)
                mod_data[mod][exp].append(da_data.where(ar6_land[mod]==1))
            
                da_rls = nc_read(
                    file,
                    y1,
                    var,
                    flag_temp_center,
                    flag_standardize,
                    freq=freq)
                mod_rls[mod][exp].append(da_rls.where(ar6_land[mod]==1))                
            
            concat_dim = np.arange(len(mod_data[mod][exp]))
            aligned = xr.concat(mod_data[mod][exp],
                                dim=concat_dim)
            mod_ens[mod][exp] = da_ensembler(deepcopy(mod_data[mod][exp]))
            mod_data[mod][exp] = aligned
            mod_data[mod][exp] = mod_data[mod][exp].rename({'concat_dim':'rls'})
            
            concat_dim = np.arange(len(mod_rls[mod][exp]))
            aligned = xr.concat(
                mod_rls[mod][exp],
                dim=concat_dim)
            mod_rls[mod][exp] = aligned
            mod_rls[mod][exp] = mod_rls[mod][exp].rename({'concat_dim':'rls'})            
        
        
        if lu_techn == 'mean':
        
            mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']
            
        mod_ens[mod]['lu'] = standard_data(mod_ens[mod]['lu'])
        
    return mod_ens,mod_data,mod_rls
#%%============================================================================

                
    #     # converting dictionaries of mod_ens to a single dataset/data array with X ar6 regions for space and Y timesteps for time
    #     testmod = models[0]
    #     testexp = exps[0]
    #     nt = len(mod_ens[testmod][testexp].time)
    #     time = mod_ens[testmod][testexp].time
    #     ar6data = weighted_mean(continents,
    #                             mod_ens[testmod][testexp],
    #                             ar6_regs[testmod],
    #                             ns)
    #     ipcc_regs = []
    #     for c in continents.keys():
    #         for r in continents[c]:
    #             ipcc_regs.append(r)
                    
    #     mod_ar6 = xr.DataArray(
    #         data = ar6data,
    #         dims=['time','IPCC_regs'],
    #         coords=dict(
    #             IPCC_regs=ipcc_regs,
    #             time=time
    #         ),
    #         attrs=dict(
    #             exp='historical',
    #             mod='CanESM5',
    #         ))
        
    # mod_ens,ar6data
# %%
