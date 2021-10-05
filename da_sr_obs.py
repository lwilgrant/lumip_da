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
    # obs series for detection and attribution


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# mod data
def obs_subroutine(obsDIR,
                   continents,
                   continent_names,
                   models,
                   var,
                   tres,
                   t_ext,
                   obs,
                   freq,
                   maps,
                   nt,
                   ns):

    os.chdir(obsDIR)
    obs_files = {}
    obs_data = {}
    obs_data_continental = {}
    obs_ts = {}
    obs_mmm = []
    obs_mmm_c = {}
    for c in continents.keys():
        obs_mmm_c[c] = []

    for mod in models:
        
        for file in [file for file in sorted(os.listdir(obsDIR))\
                    if var in file\
                        and mod in file\
                        and tres in file\
                        and 'unmasked' in file\
                        and t_ext in file\
                        and obs in file]:
            
            obs_files[mod] = file
            
            da = nc_read(obs_files[mod],
                        y1,
                        var,
                        mod=True,
                        freq=freq).where(maps[mod][lulcc_type] == 1)
            
            # weighted mean
            obs_ar6 = weighted_mean(continents,
                                    da,
                                    ar6_regs[mod],
                                    nt,
                                    ns)
                
            # remove tsteps with nans (temporal_rows x spatial_cols shaped matrix)
            obs_ar6 = del_rows(obs_ar6)

            obs_mmm.append(obs_ar6)
            
            # temporal centering
            obs_ar6_center = temp_center(ns,
                                        obs_ar6)
            nt=np.shape(obs_ar6)[0]
            
            obs_ts[mod] = obs_ar6_center
            obs_data[mod] = obs_ar6_center.flatten()
            obs_data_continental[mod] = {}
            
            for c in continents.keys():
                
                cnt_idx = continent_names.index(c)
                
                if cnt_idx == 0:
                    
                    strt_idx = 0
                    
                else:
                    
                    strt_idx = 0
                    idxs = np.arange(0,cnt_idx)
                    
                    for i in idxs:
                        strt_idx += len(continents[continent_names[i]])
                        
                n = len(continents[c])
                obs_mmm_c[c].append(obs_ar6[:,strt_idx:strt_idx+n])
                obs_data_continental[mod][c] = obs_ar6[:,strt_idx:strt_idx+n].flatten()

    # mmm of individual obs series for global + continental
    obs_ens,_ = ensembler(obs_mmm,
                        ax=0)
    obs_ts['mmm'] = temp_center(ns,
                                obs_ens)
    obs_data['mmm'] = temp_center(ns,
                                obs_ens).flatten()
    obs_data_continental['mmm'] = {}

    for c in continents.keys():
        
        obs_ens,_ = ensembler(obs_mmm_c[c],
                            ax=0)
        n = len(continents[c])
        obs_data_continental['mmm'][c] = temp_center(n,
                                                    obs_ens).flatten()
        
    return obs_data,obs_data_continental,obs_ts