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
    # model ensembles for d&a
    # model t-series as ar6-weighted matrices of rows for tsteps and columns for ar6 regions


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# mod data
def ensemble_subroutine(modDIR,
                        maps,
                        exps,
                        exps_start,
                        var,
                        tres,
                        t_ext,
                        grid_type,
                        freq,
                        obs,
                        continents,
                        ns,
                        ar6_regs):
    os.chdir(modDIR)

    fp_files = {}
    mod_data = {}
    mod_ens = {}
    mod_ens['mmm'] = {}
    mod_ts_ens = {}
    mod_ts_ens['mmm'] = {}
    for exp in exps:
        mod_ts_ens['mmm'][exp] = []
        

    # individual model ensembles and lu extraction
    for mod in models:
        
        fp_files[mod] = {}
        mod_data[mod] = {}
        mod_ens[mod] = {}
        
        for exp in exps_start:
            
            fp_files[mod][exp] = []
            mod_data[mod][exp] = []
            
            if grid_type == 'obs':
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                                and mod in file\
                                and exp in file\
                                and tres in file\
                                and 'unmasked' in file\
                                and t_ext in file\
                                and obs in file]:
                    
                    fp_files[mod][exp].append(file)
                    
            elif grid_type == 'model':
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                                and mod in file\
                                and exp in file\
                                and tres in file\
                                and 'unmasked' in file\
                                and t_ext in file\
                                and not obs in file]:
                    
                    fp_files[mod][exp].append(file)                
                
            for file in fp_files[mod][exp]:
            
                # mod data and coords for ar6 
                da = nc_read(file,
                             y1,
                             var,
                             mod=True,
                             freq=freq)
                
                mod_data[mod][exp].append(da.where(maps[mod][lulcc_type] == 1))
                
            mod_ens[mod][exp] = da_ensembler(mod_data[mod][exp])
        
        mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']
        
        mod_ts_ens[mod] = {}
        
        for exp in exps:
            
            fp_files[mod][exp] = []
            mod_ts_ens[mod][exp] = []
        
            if grid_type == 'obs':
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                                and mod in file\
                                and exp in file\
                                and tres in file\
                                and 'unmasked' in file\
                                and t_ext in file\
                                and obs in file]:
                    
                    fp_files[mod][exp].append(file)
                    
            elif grid_type == 'model':
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                                and mod in file\
                                and exp in file\
                                and tres in file\
                                and 'unmasked' in file\
                                and t_ext in file\
                                and not obs in file]:
                    
                    fp_files[mod][exp].append(file)  
                
            for file in fp_files[mod][exp]:
            
                # mod data and coords for ar6 mask
                da = nc_read(file,
                            y1,
                            var,
                            mod=True,
                            freq=freq)
                
                nt = len(da.time.values)
                
                # weighted mean
                mod_ar6 = weighted_mean(continents,
                                        da.where(maps[mod][lulcc_type] == 1),
                                        ar6_regs[mod],
                                        nt,
                                        ns)
                        
                # remove tsteps with nans (temporal x spatial shaped matrix)
                mod_ar6 = del_rows(mod_ar6)
                            
                # temporal centering
                mod_ar6_center = temp_center(ns,
                                             mod_ar6)
                mod_ts_ens[mod][exp].append(mod_ar6_center)
                mod_ts_ens['mmm'][exp].append(mod_ar6_center)
                
        
            mod_ts_ens[mod][exp] = np.stack(mod_ts_ens[mod][exp],axis=0)

    for exp in exps:        
        mod_ts_ens['mmm'][exp] = np.stack(mod_ts_ens['mmm'][exp],axis=0)
        
    return mod_ens,mod_ts_ens,nt