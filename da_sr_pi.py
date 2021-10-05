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


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# mod data
def picontrol_subroutine(piDIR,
                         models,
                         continents,
                         continent_names,
                         var,
                         tres,
                         t_ext,
                         y1,
                         freq,
                         maps,
                         ar6_regs,
                         ns,
                         nt):

    # pi data
    os.chdir(piDIR)
    pi_files = {}
    pi_data = {}
    pi_data_continental = {}
    ctl_data = {}
    ctl_data_continental = {}

    for mod in models:
        
        pi_files[mod] = []
        pi_data[mod] = []
        pi_data_continental[mod] = {}
        ctl_data_continental[mod] = {}
        
        for c in continents.keys():
            pi_data_continental[mod][c] = []
        
        for file in [file for file in sorted(os.listdir(piDIR))\
                    if var in file\
                        and mod in file\
                            and tres in file\
                                and 'unmasked' in file\
                                    and t_ext in file]:
            
            pi_files[mod].append(file)
        
        shuffle(pi_files[mod])
        
        for file in pi_files[mod]:
            
            # mod data and coords for ar6 mask
            da = nc_read(file,
                         y1,
                         var,
                         mod=True,
                         freq=freq).where(maps[mod][lulcc_type] == 1)
            
            # weighted mean
            pi_ar6 = weighted_mean(continents,
                                   da,
                                   ar6_regs[mod],
                                   nt,
                                   ns)
                
            # remove tsteps with nans (temporal x spatial shaped matrix)
            pi_ar6 = del_rows(pi_ar6)
                    
            # temporal centering
            pi_ar6 = temp_center(ns,
                                 pi_ar6)
            
            # 1-D pi array to go into pi-chunks for DA
            pi_data[mod].append(pi_ar6.flatten())
            
            # 1-D pi array per continent
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
                pi_data_continental[mod][c].append(pi_ar6[:,strt_idx:strt_idx+n].flatten())
            
        ctl_data[mod] = np.stack(pi_data[mod],axis=0)
        
        for c in continents.keys():
            
            ctl_data_continental[mod][c] = np.stack(pi_data_continental[mod][c],axis=0)
            
    # collect all pi data for mmm approach
    ctl_list = []
    for mod in models:
        
        ctl_list.append(ctl_data[mod])
        
    ctl_data['mmm'] = np.concatenate(ctl_list,
                                     axis=0)
    ctl_list_c = {}
    ctl_data_continental['mmm'] = {}

    for c in continents.keys():
        
        ctl_list_c[c] = []
        
        for mod in models:
            
            ctl_list_c[c].append(ctl_data_continental[mod][c])
            
        ctl_data_continental['mmm'][c] = np.concatenate(ctl_list_c[c],
                                                        axis=0)
        
    return ctl_data,ctl_data_continental
