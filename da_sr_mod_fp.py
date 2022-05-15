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


# %%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from copy import deepcopy
from da_funcs import *


# %%============================================================================

# mod data
def fingerprint_subroutine(
    obs_types,
    grid,
    agg,
    ns,
    nt,
    mod_ens,
    exps,
    models,
    ar6_regs,
    ar6_wts,
    cnt_regs,
    cnt_wts,
    weight,
    continents,
    continent_names,
    exp_list
):


    # fingerprint dictionaries
    fp_data = {}  # data cross all exps/models/obs as ar6-weighted mean arrays
    fp = {}  # contains histnolu and lu ar6-weighted mean arrays stacked for global OF analysis
    fp_data_continental = {} # data cross all exps/models/obs as ar6-weighted mean arrays per continent
    fp_continental = {} # contains histnolu and lu ar6-weighted mean arrays for continental analysis
    fp_data_ar6 = {}  # data cross all exps/models/obs as ar6-weighted mean arrays per ar6
    fp_ar6 = {}  # contains histnolu and lu ar6-weighted mean arrays for ar6 analysis

    # lists for ensemble of mod_ar6 tstep x region arrays and continental slices
    mmm = {}  # for global analysis of ar6 weighted means
    mmm_c = {}  # for continental analysis of ar6 weighted means
    mmm_ar6 = {}  # for continental analysis of ar6 weighted means

    # observation resolution data
    if grid == 'obs':

        for exp in exps:

            mmm[exp] = {}
            mmm_c[exp] = {}
            mmm_ar6[exp] = {}

            for c in continents.keys():

                mmm_c[exp][c] = {}

                for obs in obs_types:

                    mmm_c[exp][c][obs] = []

                for ar6 in continents[c]:

                    mmm_ar6[exp][ar6] = {}

                    for obs in obs_types:

                        mmm_ar6[exp][ar6][obs] = []

            for obs in obs_types:

                mmm[exp][obs] = []

        # model mmm fingerprints
        for mod in models:

            fp_data[mod] = {}
            fp[mod] = {}
            fp_data_continental[mod] = {}
            fp_continental[mod] = {}
            fp_data_ar6[mod] = {}
            fp_ar6[mod] = {}

            for exp in exps:

                fp_data[mod][exp] = {}
                fp_data_continental[mod][exp] = {}
                fp_data_ar6[mod][exp] = {}

                for obs in obs_types:

                    # weighted mean
                    mod_ar6 = ar6_weighted_mean(
                        continents,
                        mod_ens[mod][exp][obs],
                        ar6_regs[obs],
                        nt,
                        ns
                    )
                    mod_ar6 = del_rows(mod_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    mmm[exp][obs].append(mod_ar6) # set aside mod_ar6 for mmm global analysis of ar6 (will center after ensemble mean)
                    input_mod_ar6 = deepcopy(mod_ar6)
                    mod_ar6_center = temp_center(ns,
                                                 input_mod_ar6)  # temporal centering
                    fp_data[mod][exp][obs] = mod_ar6_center.flatten() # 1-D mod array to go into  DA
                    fp_data_continental[mod][exp][obs] = {} # 1-D mod arrays per continent for continental DA
                    fp_data_ar6[mod][exp][obs] = {} # 1-D mod arrays per ar6 region for ar6 DA

                    for c in continents.keys():

                        cnt_idx = continent_names.index(c)

                        if cnt_idx == 0:

                            strt_idx = 0

                        else:

                            strt_idx = 0
                            idxs = np.arange(0, cnt_idx)

                            for i in idxs:

                                strt_idx += len(continents[continent_names[i]])

                        n = len(continents[c])
                        mmm_c[exp][c][obs].append(mod_ar6[:, strt_idx:strt_idx+n]) # again, take uncentered/unflattened version for ensembling
                        fp_data_continental[mod][exp][obs][c] = mod_ar6_center[:,strt_idx:strt_idx+n].flatten()

                        for ar6, i in zip(continents[c], range(strt_idx,strt_idx+n)):

                            mmm_ar6[exp][ar6][obs].append(mod_ar6[:, i]) # again, take uncentered/unflattened version for ensembling
                            fp_data_ar6[mod][exp][obs][ar6] = mod_ar6_center[:, i].flatten() 

        fp_data['mmm'] = {}
        fp_data_continental['mmm'] = {}
        fp_data_ar6['mmm'] = {}

        for exp in exps:

            fp_data['mmm'][exp] = {}
            fp_data_continental['mmm'][exp] = {}
            fp_data_ar6['mmm'][exp] = {}

            for obs in obs_types:

                ens, _ = ensembler(mmm[exp][obs],
                                   ax=0)
                input_ens = deepcopy(ens)
                ens_center = temp_center(ns,
                                         input_ens)
                fp_data['mmm'][exp][obs] = ens_center.flatten()
                fp_data_continental['mmm'][exp][obs] = {}
                fp_data_ar6['mmm'][exp][obs] = {}

                for c in continents.keys():

                    n = len(continents[c])
                    ens, _ = ensembler(mmm_c[exp][c][obs],
                                       ax=0)
                    input_ens = deepcopy(ens)
                    ens_center = temp_center(n,
                                             input_ens)
                    fp_data_continental['mmm'][exp][obs][c] = ens_center.flatten()

                    for ar6 in continents[c]:

                        ens = np.mean(mmm_ar6[exp][ar6][obs],
                                      axis=0)
                        ens_center = ens - np.mean(ens)
                        fp_data_ar6['mmm'][exp][obs][ar6] = ens_center.flatten()
        
        # select fp data for analysis
        for mod in models:
            
            for obs in obs_types:

                fp_stack = [fp_data[mod][exp][obs] for exp in fp_data[mod].keys() if exp in exp_list]
                fp[mod][obs] = np.stack(fp_stack,
                                        axis=0)
                fp_continental[mod][obs] = {}
                fp_ar6[mod][obs] = {}

                for c in continents.keys():

                    fp_stack = [fp_data_continental[mod][exp][obs][c] for exp in fp_data_continental[mod].keys() if exp in exp_list]
                    fp_continental[mod][obs][c] = np.stack(fp_stack,
                                                            axis=0)

                    for ar6 in continents[c]:

                        fp_stack = [fp_data_ar6[mod][exp][obs][ar6] for exp in fp_data_ar6[mod].keys() if exp in exp_list]
                        fp_ar6[mod][obs][ar6] = np.stack(fp_stack,
                                                         axis=0)

        fp['mmm'] = {}
        fp_continental['mmm'] = {}
        fp_ar6['mmm'] = {}

        for obs in obs_types:

            fp_stack = [fp_data['mmm'][exp][obs] for exp in fp_data['mmm'].keys() if exp in exp_list]
            fp['mmm'][obs] = np.stack(fp_stack,
                                      axis=0)
            fp_continental['mmm'][obs] = {}
            fp_ar6['mmm'][obs] = {}

            for c in continents.keys():

                fp_stack = [fp_data_continental['mmm'][exp][obs][c] for exp in fp_data_continental['mmm'].keys() if exp in exp_list]
                fp_continental['mmm'][obs][c] = np.stack(fp_stack,
                                                         axis=0)
                
                for ar6 in continents[c]:
                    
                    fp_stack = [fp_data_ar6['mmm'][exp][obs][ar6] for exp in fp_data_ar6['mmm'].keys() if exp in exp_list]
                    fp_ar6['mmm'][obs][ar6] = np.stack(fp_stack,
                                                       axis=0)    

    # model resolution data
    if grid == 'model':
        
        # aggregating at ar6 level
        if agg == 'ar6':

            for exp in exps:

                mmm[exp] = []
                mmm_c[exp] = {}
                mmm_ar6[exp] = {}

                for c in continents.keys():

                    mmm_c[exp][c] = []

                    for ar6 in continents[c]:

                        mmm_ar6[exp][ar6] = []

            # models + mmm fingerprints
            for mod in models:

                fp_data[mod] = {}
                fp[mod] = {}
                fp_data_continental[mod] = {}
                fp_continental[mod] = {}
                fp_data_ar6[mod] = {}
                fp_ar6[mod] = {}

                for exp in exps:

                    # weighted mean
                    mod_ar6 = ar6_weighted_mean(
                        continents,
                        mod_ens[mod][exp],
                        ar6_regs[mod],
                        nt,
                        ns,
                        weight,
                        ar6_wts[mod]
                    )
                    mod_ar6 = del_rows(mod_ar6) # remove tsteps with nans (temporal x spatial shaped matrix)
                    mmm[exp].append(mod_ar6) # set aside mod_ar6 for mmm global analysis of ar6 (will center after ensemble mean)
                    input_mod_ar6 = deepcopy(mod_ar6)
                    mod_ar6_center = temp_center(
                            nt,
                            ns,
                            input_mod_ar6
                        )  # temporal centering
                    fp_data[mod][exp] = mod_ar6_center.flatten() # 1-D mod array to go into DA
                    fp_data_continental[mod][exp] = {} # 1-D mod arrays per continent for continental DA
                    fp_data_ar6[mod][exp] = {} # 1-D mod arrays per continent for ar6 DA

                    for c in continents.keys():

                        cnt_idx = continent_names.index(c)

                        if cnt_idx == 0:

                            strt_idx = 0

                        else:

                            strt_idx = 0
                            idxs = np.arange(0, cnt_idx)

                            for i in idxs:

                                strt_idx += len(continents[continent_names[i]])

                        n = len(continents[c])
                        mmm_c[exp][c].append(mod_ar6[:, strt_idx:strt_idx+n])
                        fp_data_continental[mod][exp][c] = mod_ar6_center[:,strt_idx:strt_idx+n].flatten()

                        for ar6, i in zip(continents[c], range(strt_idx, strt_idx+n)):

                            mmm_ar6[exp][ar6].append(mod_ar6[:, i]) # again, take uncentered/unflattened version for ensembling
                            fp_data_ar6[mod][exp][ar6] = mod_ar6_center[:,i].flatten()

            # mmm of individual model means for global + continental
            fp_data['mmm'] = {}
            fp_data_continental['mmm'] = {}
            fp_data_ar6['mmm'] = {}

            for exp in exps:

                ens, _ = ensembler(
                    mmm[exp],
                    ax=0
                )
                input_ens = deepcopy(ens)
                ens_center = temp_center(
                    nt,
                    ns,
                    input_ens
                )
                fp_data['mmm'][exp] = ens_center.flatten()
                fp_data_continental['mmm'][exp] = {}
                fp_data_ar6['mmm'][exp] = {}

                for c in continents.keys():

                    n = len(continents[c])
                    ens, _ = ensembler(
                        mmm_c[exp][c],
                        ax=0
                    )
                    input_ens = deepcopy(ens)
                    ens_center = temp_center(
                        nt,
                        n,
                        ens
                    )
                    fp_data_continental['mmm'][exp][c] = ens_center.flatten()

                    for ar6 in continents[c]:
                        
                        ens = np.mean(
                            mmm_ar6[exp][ar6],
                            axis=0
                        )
                        ens_center = ens - np.mean(ens)
                        fp_data_ar6['mmm'][exp][ar6] = ens_center.flatten()
                        
            # select fp data for analysis
            for mod in models:

                fp_stack = [fp_data[mod][exp] for exp in fp_data[mod].keys() if exp in exp_list]
                fp[mod] = np.stack(
                    fp_stack,
                    axis=0
                )

                for c in continents.keys():

                    fp_stack = [fp_data_continental[mod][exp][c] for exp in fp_data_continental[mod].keys() if exp in exp_list]
                    fp_continental[mod][c] = np.stack(
                        fp_stack,
                        axis=0
                    )

                    for ar6 in continents[c]:

                        fp_stack = [fp_data_ar6[mod][exp][ar6] for exp in fp_data_ar6[mod].keys() if exp in exp_list]
                        fp_ar6[mod][ar6] = np.stack(
                            fp_stack,
                            axis=0
                        )
                
            fp_stack = [fp_data['mmm'][exp] for exp in fp_data['mmm'].keys() if exp in exp_list]
            fp['mmm'] = np.stack(
                fp_stack,
                axis=0
            )
            fp_continental['mmm'] = {}
            fp_ar6['mmm'] = {}

            for c in continents.keys():
                
                fp_stack = [fp_data_continental['mmm'][exp][c] for exp in fp_data_continental['mmm'].keys() if exp in exp_list]
                fp_continental['mmm'][c] = np.stack(
                    fp_stack,
                    axis=0
                )
                
                for ar6 in continents[c]:
                    
                    fp_stack = [fp_data_ar6['mmm'][exp][ar6] for exp in fp_data_ar6['mmm'].keys() if exp in exp_list]
                    fp_ar6['mmm'][ar6] = np.stack(
                        fp_stack,
                        axis=0
                    )
              
        # aggregate at continental level
        if agg == 'continental':

            for exp in exps:

                mmm[exp] = []

            # models + mmm fingerprints
            for mod in models:

                fp_data[mod] = {}
                fp[mod] = {}

                for exp in exps:

                    # weighted mean
                    mod_cnt = cnt_weighted_mean(
                        continents,
                        mod_ens[mod][exp],
                        cnt_regs[mod],
                        nt,
                        ns,
                        weight,
                        cnt_wts[mod]
                    )
                    mod_cnt = del_rows(mod_cnt) # remove tsteps with nans (temporal x spatial shaped matrix)
                    mmm[exp].append(mod_cnt) # set aside mod_ar6 for mmm global analysis of ar6 (will center after ensemble mean)
                    input_mod_cnt = deepcopy(mod_cnt)
                    mod_cnt_center = temp_center(
                        nt,
                        ns,
                        input_mod_cnt
                    )  # temporal centering
                    fp_data[mod][exp] = mod_cnt_center.flatten() # 1-D mod array to go into DA

            # mmm of individual model means for global + continental
            fp_data['mmm'] = {}

            for exp in exps:

                ens, _ = ensembler(
                    mmm[exp],
                    ax=0
                )
                input_ens = deepcopy(ens)
                ens_center = temp_center(
                    nt,
                    ns,
                    input_ens
                )
                fp_data['mmm'][exp] = ens_center.flatten()
                        
            # select fp data for analysis
            for mod in models:

                fp_stack = [fp_data[mod][exp] for exp in fp_data[mod].keys() if exp in exp_list]
                fp[mod] = np.stack(
                    fp_stack,
                    axis=0
                )
                
            fp_stack = [fp_data['mmm'][exp] for exp in fp_data['mmm'].keys() if exp in exp_list]
            fp['mmm'] = np.stack(
                fp_stack,
                axis=0
            )

    return fp,fp_continental,fp_ar6


# %%
