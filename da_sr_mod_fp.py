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
    # model fingerprints from model ensembles
        # globally (per mod and mmmm); flattened series of ar6 weighted means for all regions
        # continental (per mod and mmm); flattened series of ar6 weighted emans for continents


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# mod data
def fingerprint_subroutine(obs_types,
                           grid,
                           ns,
                           nt,
                           mod_ens,
                           exps,
                           models,
                           ar6_regs,
                           continents,
                           continent_names):

    # fingerprint dictionaries
    fp_data = {} # data cross all exps/models/obs as ar6-weighted mean arrays 
    fp = {} # contains histnolu and lu ar6-weighted mean arrays stacked for global OF analysis
    fp_data_continental = {} # data cross all exps/models/obs as ar6-weighted mean arrays per continent
    fp_continental = {} # contains histnolu and lu ar6-weighted mean arrays for continental analysis
    fp_data_ar6 = {} # data cross all exps/models/obs as ar6-weighted mean arrays per ar6
    fp_ar6 = {} # contains histnolu and lu ar6-weighted mean arrays for ar6 analysis
    nx = {}

    # lists for ensemble of mod_ar6 tstep x region arrays and continental slices
    mmm = {} # for global analysis of ar6 weighted means
    mmm_c = {} # for continental analysis of ar6 weighted means
    mmm_ar6 = {} # for continental analysis of ar6 weighted means
    
    # observation resolution data
    if grid == 'obs':
        
        for exp in exps:
            
            mmm[exp] = {}
            mmm_c[exp] = {}
            
            for c in continents.keys():
                
                mmm_c[exp][c] = {}
                
                for obs in obs_types:
                    
                    mmm_c[exp][c][obs] = []
                    
            for obs in obs_types:
                
                mmm[exp][obs] = []

        # model mmm fingerprints
        for mod in models:
            
            fp_data[mod] = {}
            fp[mod] = {}
            fp_data_continental[mod] = {}
            fp_continental[mod] = {}
            nx[mod] = {}
            
            for exp in exps:
                
                fp_data[mod][exp] = {}
                fp_data_continental[mod][exp] = {}
                nx[mod][exp] = {}
                
                for obs in obs_types:
                
                    # weighted mean
                    mod_ar6 = weighted_mean(continents,
                                            mod_ens[mod][exp][obs],
                                            ar6_regs[obs],
                                            nt,
                                            ns)
                    mod_ar6 = del_rows(mod_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    mmm[exp][obs].append(mod_ar6) # set aside mod_ar6 for mmm global analysis of ar6 (will center after ensemble mean)
                    mod_ar6_center = temp_center(ns,
                                                 mod_ar6) # temporal centering
                    fp_data[mod][exp][obs] = mod_ar6_center.flatten() # 1-D mod array to go into  DA
                    fp_data_continental[mod][exp][obs] = {} # 1-D mod arrays per continent for continental DA
                    
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
                        mmm_c[exp][c][obs].append(mod_ar6[:,strt_idx:strt_idx+n])
                        fp_data_continental[mod][exp][obs][c] = mod_ar6_center[:,strt_idx:strt_idx+n].flatten()
            
            for obs in obs_types:
                        
                fp[mod][obs] = np.stack([fp_data[mod]['hist-noLu'][obs],
                                        fp_data[mod]['lu'][obs]],
                                        axis=0)
                fp_continental[mod][obs] = {}
                
                for c in continents.keys():
                    
                    fp_continental[mod][obs][c] = np.stack([fp_data_continental[mod]['hist-noLu'][obs][c],
                                                            fp_data_continental[mod]['lu'][obs][c]],
                                                           axis=0)
                
        # mmm of individual model means for global + continental
        fp_data['mmm'] = {}
        fp_data_continental['mmm'] = {}

        for exp in exps:
            
            fp_data['mmm'][exp] = {}
            fp_data_continental['mmm'][exp] = {}
            
            for obs in obs_types:

                ens,_ = ensembler(mmm[exp][obs],
                                  ax=0)
                ens_center = temp_center(ns,
                                         ens)
                fp_data['mmm'][exp][obs] = ens_center.flatten()
                fp_data_continental['mmm'][exp][obs] = {}
                
                for c in continents.keys():
                    
                    n = len(continents[c])
                    ens,_ = ensembler(mmm_c[exp][c][obs],
                                      ax=0)
                    ens_center = temp_center(n,
                                             ens)
                    fp_data_continental['mmm'][exp][obs][c] = ens_center.flatten()

        # mmm fingerprints
        fp['mmm'] = {}
        fp_continental['mmm'] = {}
        
        for obs in obs_types:
            
            fp['mmm'][obs] = np.stack([fp_data['mmm']['hist-noLu'][obs],
                                       fp_data['mmm']['lu'][obs]],
                                      axis=0)
            fp_continental['mmm'][obs] = {}

            for c in continents.keys():
                
                fp_continental['mmm'][obs][c] = np.stack([fp_data_continental['mmm']['hist-noLu'][obs][c],
                                                          fp_data_continental['mmm']['lu'][obs][c]],
                                                         axis=0)
        
    # model resolution data
    if grid == 'model':

        for exp in exps:
            
            mmm[exp] = {}
            mmm_c[exp] = {}
            
            for c in continents.keys():
                mmm_c[exp][c] = []

        # models + mmm fingerprints
        for mod in models:
            
            fp_data[mod] = {}
            fp[mod] = {}
            fp_data_continental[mod] = {}
            fp_continental[mod] = {}
            nx[mod] = {}
            
            for exp in exps:
                
                # weighted mean
                mod_ar6 = weighted_mean(continents,
                                        mod_ens[mod][exp],
                                        ar6_regs[mod],
                                        nt,
                                        ns)
                # remove tsteps with nans (temporal x spatial shaped matrix)
                mod_ar6 = del_rows(mod_ar6)
                # set aside mod_ar6 for mmm global analysis of ar6 (will center after ensemble mean)
                mmm[exp].append(mod_ar6)
                # temporal centering
                mod_ar6_center = temp_center(ns,
                                            mod_ar6)
                # 1-D mod array to go into  DA
                fp_data[mod][exp] = mod_ar6_center.flatten()
                # 1-D mod arrays per continent for continental DA
                fp_data_continental[mod][exp] = {}
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
                    mmm_c[exp][c].append(mod_ar6[:,strt_idx:strt_idx+n])
                    fp_data_continental[mod][exp][c] = mod_ar6_center[:,strt_idx:strt_idx+n].flatten()
                
            fp[mod] = np.stack([fp_data[mod]['hist-noLu'],
                                fp_data[mod]['lu']],
                               axis=0)
            
            for c in continents.keys():
                
                fp_continental[mod][c] = np.stack([fp_data_continental[mod]['hist-noLu'][c],
                                                   fp_data_continental[mod]['lu'][c]],
                                                  axis=0)
            
        # mmm of individual model means for global + continental
        fp_data['mmm'] = {}
        fp_data_continental['mmm'] = {}

        for exp in exps:
            
            ens,_ = ensembler(mmm[exp],
                            ax=0)
            ens_center = temp_center(ns,
                                    ens)
            fp_data['mmm'][exp] = ens_center.flatten()
            fp_data_continental['mmm'][exp] = {}
            
            for c in continents.keys():
                
                n = len(continents[c])
                ens,_ = ensembler(mmm_c[exp][c],
                                ax=0)
                ens_center = temp_center(n,
                                        ens)
                fp_data_continental['mmm'][exp][c] = ens_center.flatten()

        # mmm fingerprints
        fp['mmm'] = np.stack([fp_data['mmm']['hist-noLu'],
                              fp_data['mmm']['lu']],
                             axis=0)
        fp_continental['mmm'] = {}

        for c in continents.keys():
            
            fp_continental['mmm'][c] = np.stack([fp_data_continental['mmm']['hist-noLu'][c],
                                                fp_data_continental['mmm']['lu'][c]],
                                                axis=0)
            
    return fp,fp_continental,nx