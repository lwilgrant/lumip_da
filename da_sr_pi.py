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
from da_funcs import *


#%%============================================================================

# mod data
def picontrol_subroutine(
    piDIR,
    mapDIR,
    allpiDIR,
    sfDIR,
    pi_files,
    grid,
    agg,
    pi,
    models,
    obs_types,
    continents,
    continent_names,
    var,
    y1,
    freq,
    maps,
    grid_area,
    ar6_regs,
    ar6_wts,
    ar6_areas,
    ar6_land,
    cnt_regs,
    cnt_wts,
    cnt_areas,
    weight,
    ns,
    nt
):

    # pi data
    os.chdir(piDIR)
    pi_data = {}
    pi_data_continental = {}
    pi_data_ar6 = {}
    ctl_data = {}
    ctl_data_continental = {}
    ctl_data_ar6 = {}
    pi_ts_ens = {}
    
    if grid == 'obs':
        
        len_list = []
        pi_ts_ens['mmm'] = {}
        
        for obs in obs_types:
            
            pi_ts_ens['mmm'][obs] = []

        for mod in models:
            
            len_list.append(len(pi_files[mod][obs_types[0]]))
            
        min_pi_samp = np.min(len_list)
            
        for mod in models:
            
            pi_data[mod] = {}
            pi_data_continental[mod] = {}
            pi_data_ar6[mod] = {}
            ctl_data[mod] = {}
            ctl_data_continental[mod] = {}
            ctl_data_ar6[mod] = {}
            pi_ts_ens[mod] = {}
            
            for obs in obs_types:
                
                pi_data[mod][obs] = []
                pi_data_continental[mod][obs] = {}
                pi_data_ar6[mod][obs] = {}
                ctl_data_continental[mod][obs] = {}
                ctl_data_ar6[mod][obs] = {}
                pi_ts_ens[mod][obs] = []
            
                for c in continents.keys():
                    
                    pi_data_continental[mod][obs][c] = []
                    
                    for ar6 in continents[c]:
                        
                        pi_data_ar6[mod][obs][ar6] = []
                
                shuffle(pi_files[mod][obs])
                
                picm = 0
                for file in pi_files[mod][obs]:
                    
                    # mod data and coords for ar6 mask
                    da = nc_read(
                        file,
                        y1,
                        var,
                        obs=obs,
                        freq=freq
                    ).where(maps[obs] == 1)
                    pi_ar6 = ar6_weighted_mean(
                        continents, # weighted mean
                        da,
                        ar6_regs[obs],
                        nt,
                        ns
                    )
                    pi_ar6 = del_rows(pi_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    input_pi_ar6 = deepcopy(pi_ar6) # temporal centering
                    pi_ar6 = temp_center(
                        ns,
                        input_pi_ar6
                    )
                    pi_ts_ens[mod][obs].append(pi_ar6)
                    
                    if picm <= min_pi_samp:
                        
                        pi_ts_ens['mmm'][obs].append(pi_ar6)
                        
                    pi_data[mod][obs].append(pi_ar6.flatten()) # 1-D pi array to go into pi-chunks for DA
                    picm += 1
                    
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
                        
                # pi tseries data
                pi_ts_ens[mod][obs] = np.stack(pi_ts_ens[mod][obs],axis=0)
                    
        # collect all pi data for mmm approach (balance contribution of noise data from each model with taking minimum # samples)
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
                    
            pi_ts_ens['mmm'][obs] = np.stack(pi_ts_ens['mmm'][obs],axis=0)
    
    elif grid == 'model':
        
        if pi == 'model':
            
            len_list = []
            pi_ts_ens['mmm'] = []

            for mod in models:
                
                len_list.append(len(pi_files[mod]))
                
            min_pi_samp = np.min(len_list)
                
            for mod in  models:
                
                pi_data[mod] = []
                pi_data_continental[mod] = {}
                pi_data_ar6[mod] = {}
                ctl_data_continental[mod] = {}
                ctl_data_ar6[mod] = {}
                pi_ts_ens[mod] = []
                
                for c in continents.keys():
                    
                    pi_data_continental[mod][c] = []
                    
                    for ar6 in continents[c]:
                        
                        pi_data_ar6[mod][ar6] = []
                
                shuffle(pi_files[mod])
                
                picm = 0
                for file in pi_files[mod]:
                    
                    # mod data and coords for ar6 mask
                    da = nc_read(file,
                                y1,
                                var,
                                freq=freq).where(maps[mod] == 1)
                    pi_ar6 = ar6_weighted_mean(continents, # weighted mean
                                        da,
                                        ar6_regs[mod],
                                        nt,
                                        ns)
                    pi_ar6 = del_rows(pi_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    input_pi_ar6 = deepcopy(pi_ar6) # temporal centering
                    pi_ar6 = temp_center(
                        nt,
                        ns,
                        input_pi_ar6)
                    pi_ts_ens[mod].append(pi_ar6)
                        
                    if picm <= min_pi_samp:
                        
                        pi_ts_ens['mmm'].append(pi_ar6)
                            
                    pi_data[mod].append(pi_ar6.flatten()) # 1-D pi array to go into pi-chunks for DA
                    picm += 1
                    
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
            
                # pi tseries data
                pi_ts_ens[mod] = np.stack(pi_ts_ens[mod],axis=0)
                    
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
                    
            # pi tseries data
            pi_ts_ens['mmm'] = np.stack(pi_ts_ens['mmm'],axis=0)
            
        elif pi == 'allpi':
            
            os.chdir(allpiDIR)
            # list of models in allpi case for min length
            allmodels = []
            len_list = []
            
            for rls in pi_files:
                
                if 'piControl' in rls:
                    
                    allmodels.append(rls.split('_')[2])
                    
            allmodels = list(dict.fromkeys(allmodels))
            
            mod_pi_availability = {}
            allmodels_pi_files = {}
            
            for mod in allmodels:
                
                os.chdir(mapDIR)    
                grid_area[mod] = xr.open_dataset(mod+'_gridarea.nc',decode_times=False)['cell_area']
                allmodels_pi_files[mod] = []
                cnt_wts[mod] = {}
                ar6_wts[mod] = {}
                i=0
                
                for rls in pi_files:
                    
                    if rls.split('_')[2] == mod:
                       
                        if i == 0: # for new models in full piC ensemble,add ar6_land template to maps[mod] for area-weighted mean
                            
                            os.chdir(allpiDIR)
                            template = xr.open_dataset(rls,decode_times=False).tasmax.isel(time=0).squeeze(drop=True)
                            if 'height' in template.coords:
                                template = template.drop('height')
                            
                            ar6_regs[mod] = ar6_mask(template)
                            ar6_land[mod] = xr.where(ar6_regs[mod]>=0,1,0)
                            maps[mod] = ar6_land[mod]
                            
                            if agg == 'ar6':    
                                for c in continents.keys():
                                    ar6_areas[c] = {}
                                    ar6_wts[mod][c] = {}
                                    for i in continents[c]:
                                        ar6_areas[c][i] = grid_area[mod].where(ar6_regs[mod]==i).sum(dim=('lat','lon'))
                                    max_area = max(ar6_areas[c].values())
                                    for i in continents[c]:
                                        ar6_wts[mod][c][i] = ar6_areas[c][i]/max_area

                            if agg == 'continental':
                                cnt_regs[mod] = cnt_mask(sfDIR, # changes directory, correct elsewhere lower
                                                        template)
                                for c in continents.keys():
                                    cnt_areas[c] = grid_area[mod].where(cnt_regs[mod]==continents[c]).sum(dim=('lat','lon'))
                                max_area = max(cnt_areas.values())
                                for c in continents.keys():
                                    cnt_wts[mod][c] = cnt_areas[c]/max_area
                                                                        
                        i+=1
                        allmodels_pi_files[mod].append(rls)
                        
                mod_pi_availability[mod] = deepcopy(i)
                
            for mod in allmodels:
                
                len_list.append(mod_pi_availability[mod])
                
            min_pi_samp = np.min(len_list)                
            
            pi_ts_ens = []
            pi_data = []
            pi_data_continental = {}
            pi_data_ar6 = {}
            ctl_data_continental = {}
            ctl_data_ar6 = {}      
                
            shuffle(pi_files)
            os.chdir(allpiDIR)
            
            if agg == 'ar6':
                
                for c in continents.keys():
                    
                    pi_data_continental[c] = []
                    
                    for ar6 in continents[c]:
                        
                        pi_data_ar6[ar6] = []                
            
                for mod in allmodels:
                    
                    picm = 0
                    for file in allmodels_pi_files[mod]:
                        
                        # mod data and coords for ar6 mask
                        da = nc_read(file,
                                    y1,
                                    var,
                                    freq=freq).where(maps[mod] == 1)
                        pi_ar6 = ar6_weighted_mean(continents, # weighted mean
                                                   da,
                                                   ar6_regs[mod],
                                                   nt,
                                                   ns,
                                                   weight,
                                                   ar6_wts[mod])
                        pi_ar6 = del_rows(pi_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                        input_pi_ar6 = deepcopy(pi_ar6) # temporal centering
                        pi_ar6 = temp_center(
                            nt,
                            ns,
                            input_pi_ar6)
                            
                        # if picm <= min_pi_samp:
                            
                        pi_ts_ens.append(pi_ar6) 
                        pi_data.append(pi_ar6.flatten()) # 1-D pi array to go into pi-chunks for DA
                            
                        picm += 1
                            
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
                            pi_data_continental[c].append(pi_ar6[:,strt_idx:strt_idx+n].flatten())
                            
                            for ar6,i in zip(continents[c],range(strt_idx,strt_idx+n)):

                                pi_data_ar6[ar6].append(pi_ar6[:,i].flatten()) # is this flattening unnecessary or negatively impactful? already 1D (check this)
                    
                ctl_data = np.stack(pi_data,axis=0)
                    
                for c in continents.keys():
                    
                    ctl_data_continental[c] = np.stack(pi_data_continental[c],
                                                    axis=0)
                    
                    for ar6 in continents[c]:
                        
                        ctl_data_ar6[ar6] = np.stack(pi_data_ar6[ar6],
                                                    axis=0)
            
                # pi tseries data
                pi_ts_ens = np.stack(pi_ts_ens,
                                    axis=0)          
                
            elif agg == 'continental':
                
                for mod in allmodels:
                    
                    picm = 0
                    for file in allmodels_pi_files[mod]:
                        
                        # mod data and coords for ar6 mask
                        da = nc_read(file,
                                    y1,
                                    var,
                                    freq=freq).where(maps[mod] == 1)
                        pi_cnt = cnt_weighted_mean(continents, # weighted mean
                                                   da,
                                                   cnt_regs[mod],
                                                   nt,
                                                   ns,
                                                   weight,
                                                   cnt_wts[mod])                      
                        
                        pi_cnt = del_rows(pi_cnt) # remove tsteps with nans (temporal x spatial shaped matrix)
                        input_pi_cnt = deepcopy(pi_cnt) # temporal centering
                        pi_cnt = temp_center(
                            nt,
                            ns,
                            input_pi_cnt)
                            
                        pi_ts_ens.append(pi_cnt) 
                        pi_data.append(pi_cnt.flatten()) # 1-D pi array to go into pi-chunks for DA
                        
                    
                ctl_data = np.stack(pi_data,axis=0)
            
                # pi tseries data
                pi_ts_ens = np.stack(pi_ts_ens,
                                    axis=0)                     
            
    return ctl_data,ctl_data_continental,ctl_data_ar6,pi_ts_ens

# %%
