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
from copy import deepcopy
from da_funcs import *


#%%============================================================================

# mod data
def picontrol_subroutine(piDIR,
                         pi_files,
                         grid,
                         models,
                         obs_types,
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
    pi_data = {}
    pi_data_continental = {}
    pi_data_ar6 = {}
    ctl_data = {}
    ctl_data_continental = {}
    ctl_data_ar6 = {}
    
    if grid == 'obs':
        
        len_list = []

        for mod in models:
            
            len_list.append(len(pi_files[mod][obs_types[0]]))
            pi_data[mod] = {}
            pi_data_continental[mod] = {}
            pi_data_ar6[mod] = {}
            ctl_data[mod] = {}
            ctl_data_continental[mod] = {}
            ctl_data_ar6[mod] = {}
            
            for obs in obs_types:
                
                pi_data[mod][obs] = []
                pi_data_continental[mod][obs] = {}
                pi_data_ar6[mod][obs] = {}
                ctl_data_continental[mod][obs] = {}
                ctl_data_ar6[mod][obs] = {}
            
                for c in continents.keys():
                    
                    pi_data_continental[mod][obs][c] = []
                    
                    for ar6 in continents[c]:
                        
                        pi_data_ar6[mod][obs][ar6] = []
                
                shuffle(pi_files[mod][obs])
                
                for file in pi_files[mod][obs]:
                    
                    # mod data and coords for ar6 mask
                    da = nc_read(file,
                                y1,
                                var,
                                freq=freq).where(maps[obs] == 1)
                    pi_ar6 = weighted_mean(continents, # weighted mean
                                           da,
                                           ar6_regs[obs],
                                           nt,
                                           ns)
                    pi_ar6 = del_rows(pi_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    input_pi_ar6 = deepcopy(pi_ar6) # temporal centering
                    pi_ar6 = temp_center(ns,
                                         input_pi_ar6)
                    pi_data[mod][obs].append(pi_ar6.flatten()) # 1-D pi array to go into pi-chunks for DA
                    
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
                        pi_data_continental[mod][obs][c].append(pi_ar6[:,strt_idx:strt_idx+n].flatten())
                        
                        for ar6,i in zip(continents[c],range(strt_idx,strt_idx+n)):

                            pi_data_ar6[mod][obs][ar6].append(pi_ar6[:,i].flatten()) # is this flattening unnecessary or negatively impactful? already 1D (check this)
                    
                ctl_data[mod][obs] = np.stack(pi_data[mod][obs],axis=0)
                
                for c in continents.keys():
                    
                    ctl_data_continental[mod][obs][c] = np.stack(pi_data_continental[mod][obs][c],
                                                                 axis=0)
                    
                    for ar6 in continents[c]:
                        
                        ctl_data_ar6[mod][obs][ar6] = np.stack(pi_data_ar6[mod][obs][ar6],
                                                               axis=0)
                    
            # collect all pi data for mmm approach (balance contribution of noise data from each model with taking minimum # samples)
            min_pi_samp = np.min(len_list)
            ctl_list = []

            for mod in models:
                
                ctl_list.append(ctl_data[mod][obs][:min_pi_samp])
                
            ctl_data['mmm'] = np.concatenate(ctl_list,
                                             axis=0)
            ctl_list_c = {}
            ctl_list_ar6 = {}
            ctl_data_continental['mmm'] = {}
            ctl_data_ar6['mmm'] = {}
            
            for obs in obs_types:
                
                ctl_list_c[obs] = {}
                ctl_list_ar6[obs] = {}
                ctl_data_continental['mmm'][obs] = {}
                ctl_data_ar6['mmm'][obs] = {}

                for c in continents.keys():
                    
                    ctl_list_c[obs][c] = []
                    
                    for mod in models:
                        
                        ctl_list_c[obs][c].append(ctl_data_continental[mod][obs][c][:min_pi_samp])
                        
                    ctl_data_continental['mmm'][obs][c] = np.concatenate(ctl_list_c[obs][c],
                                                                         axis=0)
                    
                    for ar6 in continents[c]:
                        
                        ctl_list_ar6[obs][ar6] = []
                        
                        for mod in models:
                            
                            ctl_list_ar6[obs][ar6].append(ctl_data_ar6[mod][obs][ar6][:min_pi_samp])
                            
                        ctl_data_ar6['mmm'][obs][ar6] = np.concatenate(ctl_list_ar6[obs][ar6],
                                                                       axis=0)
    
    elif grid == 'model':

        len_list = []

        for mod in models:
            
            len_list.append(len(pi_files[mod]))
            pi_data[mod] = []
            pi_data_continental[mod] = {}
            pi_data_ar6[mod] = {}
            ctl_data_continental[mod] = {}
            ctl_data_ar6[mod] = {}
            
            for c in continents.keys():
                
                pi_data_continental[mod][c] = []
                
                for ar6 in continents[c]:
                    
                    pi_data_ar6[mod][ar6] = []
            
            shuffle(pi_files[mod])
            
            for file in pi_files[mod]:
                
                # mod data and coords for ar6 mask
                da = nc_read(file,
                             y1,
                             var,
                             freq=freq).where(maps[mod] == 1)
                pi_ar6 = weighted_mean(continents, # weighted mean
                                       da,
                                       ar6_regs[mod],
                                       nt,
                                       ns)
                pi_ar6 = del_rows(pi_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                input_pi_ar6 = deepcopy(pi_ar6) # temporal centering
                pi_ar6 = temp_center(ns,
                                     input_pi_ar6)
                pi_data[mod].append(pi_ar6.flatten()) # 1-D pi array to go into pi-chunks for DA
                
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
                    
                    for ar6,i in zip(continents[c],range(strt_idx,strt_idx+n)):

                        pi_data_ar6[mod][ar6].append(pi_ar6[:,i].flatten()) # is this flattening unnecessary or negatively impactful? already 1D (check this)
                
            ctl_data[mod] = np.stack(pi_data[mod],axis=0)
            
            for c in continents.keys():
                
                ctl_data_continental[mod][c] = np.stack(pi_data_continental[mod][c],
                                                        axis=0)
                
                for ar6 in continents[c]:
                    
                    ctl_data_ar6[mod][ar6] = np.stack(pi_data_ar6[mod][ar6],
                                                      axis=0)
                
        # collect all pi data for mmm approach (balance contribution of noise data from each model with taking minimum # samples)
        min_pi_samp = np.min(len_list)
        ctl_list = []

        for mod in models:
            
            ctl_list.append(ctl_data[mod][:min_pi_samp])
            
        ctl_data['mmm'] = np.concatenate(ctl_list,
                                         axis=0)
        ctl_list_c = {}
        ctl_list_ar6 = {}
        ctl_data_continental['mmm'] = {}
        ctl_data_ar6['mmm'] = {}

        for c in continents.keys():
            
            ctl_list_c[c] = []
            
            for mod in models:
                
                ctl_list_c[c].append(ctl_data_continental[mod][c][:min_pi_samp])
                
            ctl_data_continental['mmm'][c] = np.concatenate(ctl_list_c[c],
                                                            axis=0)
            
            for ar6 in continents[c]:
                
                ctl_list_ar6[ar6] = []
                
                for mod in models:
                    
                    ctl_list_ar6[ar6].append(ctl_data_ar6[mod][ar6][:min_pi_samp])
                    
                ctl_data_ar6['mmm'][ar6] = np.concatenate(ctl_list_ar6[ar6],
                                                          axis=0)
        
    return ctl_data,ctl_data_continental,ctl_data_ar6
