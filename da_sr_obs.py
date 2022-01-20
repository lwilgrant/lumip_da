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
from copy import deepcopy
from da_funcs import *


#%%============================================================================

# mod data
def obs_subroutine(obsDIR,
                   grid,
                   obs_files,
                   continents,
                   continent_names,
                   obs_types,
                   models,
                   y1,
                   var,
                   maps,
                   ar6_regs,
                   ar6_wts,
                   cnt_regs,
                   cnt_wts,
                   agg,
                   weight,
                   freq,
                   nt,
                   ns):

    os.chdir(obsDIR)

    if grid == 'obs':
        
        obs_data = {}
        obs_data_continental = {}
        obs_data_ar6 = {}
        obs_ts = {}
            
        for obs in obs_types:
            
            obs_data_continental[obs] = {}
            obs_data_ar6[obs] = {}
            da = nc_read(obs_files[obs],
                         y1,
                         var,
                         freq=freq).where(maps[obs] == 1)
            obs_ar6 = ar6_weighted_mean(continents,
                                    da,
                                    ar6_regs[obs],
                                    nt,
                                    ns)
            obs_ar6 = del_rows(obs_ar6)
            input_obs_ar6 = deepcopy(obs_ar6)
            obs_ar6_center = temp_center(ns,
                                         input_obs_ar6)
            obs_ts[obs] = obs_ar6_center
            obs_data[obs] = obs_ar6_center.flatten()
            obs_data_continental[obs] = {}
            obs_data_ar6[obs] = {}
            
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
                obs_data_continental[obs][c] = obs_ar6_center[:,strt_idx:strt_idx+n].flatten()
                
                for ar6, i in zip(continents[c], range(strt_idx,strt_idx+n)):

                    obs_data_ar6[obs][ar6] = obs_ar6_center[:, i].flatten() 

    elif grid == 'model':
        
        obs_data = {}
        obs_data_continental = {}
        obs_data_ar6 = {}
        obs_ts = {}
        obs_mmm = {}
        obs_mmm_c = {}
        obs_mmm_ar6 = {}
            
        for obs in obs_types:
            
            obs_data[obs] = {}
            obs_data_continental[obs] = {}
            obs_data_ar6[obs] = {}
            obs_ts[obs] = {}
            obs_mmm[obs] = []
            obs_mmm_c[obs] = {}
            obs_mmm_ar6[obs] = {}
                    
            if agg == 'ar6':
                
                for c in continents.keys():
                
                    obs_mmm_c[obs][c] = []
                    
                    for ar6 in continents[c]:
                        
                        obs_mmm_ar6[obs][ar6] = []                

                for mod in models:
                        
                    da = nc_read(obs_files[obs][mod],
                                y1,
                                var,
                                freq=freq).where(maps[mod] == 1)
                    obs_ar6 = ar6_weighted_mean(continents,
                                                da,
                                                ar6_regs[mod],
                                                nt,
                                                ns,
                                                weight,
                                                ar6_wts[mod])
                    obs_ar6 = del_rows(obs_ar6)
                    obs_mmm[obs].append(obs_ar6)
                    input_obs_ar6 = deepcopy(obs_ar6)
                    obs_ar6_center = temp_center(
                        nt,
                        ns,
                        input_obs_ar6)
                    obs_ts[obs][mod] = obs_ar6_center
                    obs_data[obs][mod] = obs_ar6_center.flatten()
                    obs_data_continental[obs][mod] = {}
                    obs_data_ar6[obs][mod] = {}
                    
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
                        obs_mmm_c[obs][c].append(obs_ar6[:,strt_idx:strt_idx+n])
                        obs_data_continental[obs][mod][c] = obs_ar6_center[:,strt_idx:strt_idx+n].flatten()
                        
                        for ar6, i in zip(continents[c], range(strt_idx,strt_idx+n)):

                            obs_mmm_ar6[obs][ar6].append(obs_ar6[:, i]) 
                            obs_data_ar6[obs][mod][ar6] = obs_ar6_center[:, i].flatten() 

                # mmm of individual obs series for global + continental + ar6
                obs_ens,_ = ensembler(obs_mmm[obs],
                                        ax=0)
                obs_ts[obs]['mmm'] = temp_center(
                    nt,
                    ns,
                    obs_ens)
                obs_data[obs]['mmm'] = obs_ens.flatten()
                obs_data_continental[obs]['mmm'] = {}
                obs_data_ar6[obs]['mmm'] = {}

                for c in continents.keys():
                    
                    obs_ens,_ = ensembler(obs_mmm_c[obs][c],
                                            ax=0)
                    n = len(continents[c])
                    obs_data_continental[obs]['mmm'][c] = temp_center(
                        nt,
                        n,
                        obs_ens).flatten()
                    
                    for ar6 in continents[c]:
                        
                        obs_ens = np.mean(obs_mmm_ar6[obs][ar6],
                                            axis=0)
                        obs_ens_center = obs_ens - np.mean(obs_ens)
                        obs_data_ar6[obs]['mmm'][ar6] = obs_ens_center.flatten()
                        
            elif agg == 'continental':

                for mod in models:
                        
                    da = nc_read(obs_files[obs][mod],
                                y1,
                                var,
                                freq=freq).where(maps[mod] == 1)
                    obs_cnt = cnt_weighted_mean(continents, # weighted mean
                                                   da,
                                                   cnt_regs[mod],
                                                   nt,
                                                   ns,
                                                   weight,
                                                   cnt_wts[mod])  
                    obs_cnt = del_rows(obs_cnt)
                    obs_mmm[obs].append(obs_cnt)
                    input_obs_cnt = deepcopy(obs_cnt)
                    obs_cnt_center = temp_center(
                        nt,
                        ns,
                        input_obs_cnt)
                    obs_ts[obs][mod] = obs_cnt_center
                    obs_data[obs][mod] = obs_cnt_center.flatten()
                    obs_data_continental[obs][mod] = {}
                    obs_data_ar6[obs][mod] = {}

                # mmm of individual obs series for global + continental + ar6
                obs_ens,_ = ensembler(obs_mmm[obs],
                                        ax=0)
                obs_ts[obs]['mmm'] = temp_center(
                    nt,
                    ns,
                    obs_ens)
                obs_data[obs]['mmm'] = obs_ens.flatten()                    
            
    return obs_data,obs_data_continental,obs_data_ar6,obs_ts
# %%
