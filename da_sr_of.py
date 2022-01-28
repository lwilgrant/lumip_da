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
    # optimal fingerprinting results 


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from da_funcs import *


#%%============================================================================

# mod data
def of_subroutine(grid,
                  models,
                  nx,
                  analysis,
                  exp_list,
                  obs_types,
                  pi,
                  obs_data,
                  obs_data_continental,
                  obs_data_ar6,
                  fp,
                  fp_continental,
                  fp_ar6,
                  ctl_data,
                  ctl_data_continental,
                  ctl_data_ar6,
                  bs_reps,
                  ns,
                  nt,
                  reg,
                  cons_test,
                  formule_ic_tls,
                  trunc,
                  ci_bnds,
                  continents):

    var_sfs = {}
    bhi = {}
    b = {}
    blow = {}
    pval = {}
    var_fin = {}
    var_ctlruns = {}
    U = {}
    yc = {}
    Z1c = {}
    Z2c = {}
    Xc = {}
    Cf1 = {}
    Ft = {}
    beta_hat = {}

    models.append('mmm')   

    if grid == 'obs':
        
        for obs in obs_types:
            
            var_sfs[obs] = {}
            bhi[obs] = {}
            b[obs] = {}
            blow[obs] = {}
            pval[obs] = {}
            var_fin[obs] = {}
            var_ctlruns[obs] = {}
            U[obs] = {}
            yc[obs] = {}
            Z1c[obs] = {}
            Z2c[obs] = {}
            Xc[obs] = {}
            Cf1[obs] = {}
            Ft[obs] = {}
            beta_hat[obs] = {}
                
            for mod in models:

                #==============================================================================
                
                # global analysis
                if analysis == "global":
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = []
                        b[obs][mod][exp] = []
                        blow[obs][mod][exp] = []
                        pval[obs][mod][exp] = []
                    
                    y = obs_data[obs]
                    X = fp[mod][obs]
                    ctl = ctl_data[mod][obs]
                    nb_runs_x= nx[mod]
                    
                    if bs_reps == 0: # for no bs, run ROF once
                    
                        bs_reps += 1
                    
                    for i in np.arange(0,bs_reps):
                        
                        # shuffle rows of ctl
                        ctl = np.take(ctl,
                                      np.random.permutation(ctl.shape[0]),
                                      axis=0)
                        
                        # run detection and attribution
                        var_sfs[obs][mod],\
                        var_ctlruns[obs][mod],\
                        proj,\
                        U[obs][mod],\
                        yc[obs][mod],\
                        Z1c[obs][mod],\
                        Z2c[obs][mod],\
                        Xc[obs][mod],\
                        Cf1[obs][mod],\
                        Ft[obs][mod],\
                        beta_hat[obs][mod] = da_run(y,
                                                    X,
                                                    ctl,
                                                    nb_runs_x,
                                                    ns,
                                                    nt,
                                                    reg,
                                                    cons_test,
                                                    formule_ic_tls,
                                                    trunc,
                                                    ci_bnds)
                        
                        # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                        for i,exp in enumerate(exp_list):
                            
                            bhi[obs][mod][exp].append(var_sfs[obs][mod][2,i])
                            b[obs][mod][exp].append(var_sfs[obs][mod][1,i])
                            blow[obs][mod][exp].append(var_sfs[obs][mod][0,i])
                            pval[obs][mod][exp].append(var_sfs[obs][mod][3,i])
                    
                    for exp in exp_list:
                        
                        bhi_med = np.median(bhi[obs][mod][exp])
                        b_med = np.median(b[obs][mod][exp])
                        blow_med = np.median(blow[obs][mod][exp])
                        pval_med = np.median(pval[obs][mod][exp])
                        var_fin[obs][mod][exp] = [bhi_med,
                                                b_med,
                                                blow_med,
                                                pval_med]
                
                #==============================================================================
                    
                # continental analysis
                elif analysis == 'continental':
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    var_sfs[obs][mod] = {}
                    var_ctlruns[obs][mod] = {}
                    U[obs][mod] = {}
                    yc[obs][mod] = {}
                    Z1c[obs][mod] = {}
                    Z2c[obs][mod] = {}
                    Xc[obs][mod] = {}
                    Cf1[obs][mod] = {}
                    Ft[obs][mod] = {}
                    beta_hat[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = {}
                        b[obs][mod][exp] = {}
                        blow[obs][mod][exp] = {}
                        pval[obs][mod][exp] = {}
                        var_fin[obs][mod][exp] = {}
                        
                        for c in continents.keys():
                            
                            bhi[obs][mod][exp][c] = []
                            b[obs][mod][exp][c] = []
                            blow[obs][mod][exp][c] = []
                            pval[obs][mod][exp][c] = []
                        
                    for c in continents.keys():
                    
                        y = obs_data_continental[obs][mod][c]
                        X = fp_continental[mod][c]
                        ctl = ctl_data_continental[mod][c]
                        nb_runs_x= nx[mod]
                        ns = len(continents[c])
                    
                        if bs_reps == 0: # for no bs, run ROF once
                        
                            bs_reps += 1
                        
                        for i in np.arange(0,bs_reps):
                            
                            # shuffle rows of ctl
                            ctl = np.take(ctl,
                                          np.random.permutation(ctl.shape[0]),
                                          axis=0)
                            
                            # run detection and attribution
                            var_sfs[obs][mod][c],\
                            var_ctlruns[obs][mod][c],\
                            proj,\
                            U[obs][mod][c],\
                            yc[obs][mod][c],\
                            Z1c[obs][mod][c],\
                            Z2c[obs][mod][c],\
                            Xc[obs][mod][c],\
                            Cf1[obs][mod][c],\
                            Ft[obs][mod][c],\
                            beta_hat[obs][mod][c] = da_run(y,
                                                        X,
                                                        ctl,
                                                        nb_runs_x,
                                                        ns,
                                                        nt,
                                                        reg,
                                                        cons_test,
                                                        formule_ic_tls,
                                                        trunc,
                                                        ci_bnds)
                            
                            # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                            for i,exp in enumerate(exp_list):
                                
                                bhi[obs][mod][exp][c].append(var_sfs[obs][mod][c][2,i])
                                b[obs][mod][exp][c].append(var_sfs[obs][mod][c][1,i])
                                blow[obs][mod][exp][c].append(var_sfs[obs][mod][c][0,i])
                                pval[obs][mod][exp][c].append(var_sfs[obs][mod][c][3,i])
                        
                    for exp in exp_list:
                        
                        for c in continents.keys():
                            
                            bhi_med = np.median(bhi[obs][mod][exp][c])
                            b_med = np.median(b[obs][mod][exp][c])
                            blow_med = np.median(blow[obs][mod][exp][c])
                            pval_med = np.median(pval[obs][mod][exp][c])
                            var_fin[obs][mod][exp][c] = [bhi_med,
                                                         b_med,
                                                         blow_med,
                                                         pval_med]
                                
                #==============================================================================
                    
                # continental analysis
                elif analysis == 'ar6':
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    var_sfs[obs][mod] = {}
                    var_ctlruns[obs][mod] = {}
                    U[obs][mod] = {}
                    yc[obs][mod] = {}
                    Z1c[obs][mod] = {}
                    Z2c[obs][mod] = {}
                    Xc[obs][mod] = {}
                    Cf1[obs][mod] = {}
                    Ft[obs][mod] = {}
                    beta_hat[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = {}
                        b[obs][mod][exp] = {}
                        blow[obs][mod][exp] = {}
                        pval[obs][mod][exp] = {}
                        var_fin[obs][mod][exp] = {}
                        
                        for c in continents.keys():
                            
                            for ar6 in continents[c]:
                            
                                bhi[obs][mod][exp][ar6] = []
                                b[obs][mod][exp][ar6] = []
                                blow[obs][mod][exp][ar6] = []
                                pval[obs][mod][exp][ar6] = []
                        
                    for c in continents.keys():
                        
                        for ar6 in continents[c]:
                    
                            y = obs_data_ar6[obs][mod][ar6]
                            X = fp_ar6[mod][ar6]
                            ctl = ctl_data_ar6[mod][ar6]
                            nb_runs_x= nx[mod]
                            ns = 1
                        
                            if bs_reps == 0: # for no bs, run ROF once
                            
                                bs_reps += 1
                            
                            for i in np.arange(0,bs_reps):
                                
                                # shuffle rows of ctl
                                ctl = np.take(ctl,
                                            np.random.permutation(ctl.shape[0]),
                                            axis=0)
                                
                                # run detection and attribution
                                var_sfs[obs][mod][ar6],\
                                var_ctlruns[obs][mod][ar6],\
                                proj,\
                                U[obs][mod][ar6],\
                                yc[obs][mod][ar6],\
                                Z1c[obs][mod][ar6],\
                                Z2c[obs][mod][ar6],\
                                Xc[obs][mod][ar6],\
                                Cf1[obs][mod][ar6],\
                                Ft[obs][mod][ar6],\
                                beta_hat[obs][mod][ar6] = da_run(y,
                                                                X,
                                                                ctl,
                                                                nb_runs_x,
                                                                ns,
                                                                nt,
                                                                reg,
                                                                cons_test,
                                                                formule_ic_tls,
                                                                trunc,
                                                                ci_bnds)
                                
                                # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                                for i,exp in enumerate(exp_list):
                                    
                                    bhi[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][2,i])
                                    b[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][1,i])
                                    blow[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][0,i])
                                    pval[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][3,i])
                            
                    for exp in exp_list:
                        
                        for c in continents.keys():
                            
                            for ar6 in continents[c]:
                            
                                bhi_med = np.median(bhi[obs][mod][exp][ar6])
                                b_med = np.median(b[obs][mod][exp][ar6])
                                blow_med = np.median(blow[obs][mod][exp][ar6])
                                pval_med = np.median(pval[obs][mod][exp][ar6])
                                var_fin[obs][mod][exp][ar6] = [bhi_med,
                                                               b_med,
                                                               blow_med,
                                                               pval_med]


    elif grid == 'model':
    
        for obs in obs_types:
            
            var_sfs[obs] = {}
            bhi[obs] = {}
            b[obs] = {}
            blow[obs] = {}
            pval[obs] = {}
            var_fin[obs] = {}
            var_ctlruns[obs] = {}
            U[obs] = {}
            yc[obs] = {}
            Z1c[obs] = {}
            Z2c[obs] = {}
            Xc[obs] = {}
            Cf1[obs] = {}
            Ft[obs] = {}
            beta_hat[obs] = {}
                
            for mod in models:

                #==============================================================================
                
                # global analysis
                if analysis == "global":
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = []
                        b[obs][mod][exp] = []
                        blow[obs][mod][exp] = []
                        pval[obs][mod][exp] = []
                    
                    y = obs_data[obs][mod]
                    X = fp[mod]
                    if pi == 'model':
                        ctl = ctl_data[mod]
                    elif pi == 'allpi':
                        ctl = ctl_data
                    nb_runs_x= nx[mod]
                    
                    if bs_reps == 0: # for no bs, run ROF once
                    
                        bs_reps += 1
                    
                    for i in np.arange(0,bs_reps):
                        
                        # shuffle rows of ctl
                        ctl = np.take(ctl,
                                      np.random.permutation(ctl.shape[0]),
                                      axis=0)
                        
                        # run detection and attribution
                        var_sfs[obs][mod],\
                        var_ctlruns[obs][mod],\
                        proj,\
                        U[obs][mod],\
                        yc[obs][mod],\
                        Z1c[obs][mod],\
                        Z2c[obs][mod],\
                        Xc[obs][mod],\
                        Cf1[obs][mod],\
                        Ft[obs][mod],\
                        beta_hat[obs][mod] = da_run(y,
                                                    X,
                                                    ctl,
                                                    nb_runs_x,
                                                    ns,
                                                    nt,
                                                    reg,
                                                    cons_test,
                                                    formule_ic_tls,
                                                    trunc,
                                                    ci_bnds)
                        
                        # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                        for i,exp in enumerate(exp_list):
                            
                            bhi[obs][mod][exp].append(var_sfs[obs][mod][2,i])
                            b[obs][mod][exp].append(var_sfs[obs][mod][1,i])
                            blow[obs][mod][exp].append(var_sfs[obs][mod][0,i])
                            pval[obs][mod][exp].append(var_sfs[obs][mod][3,i])
                    
                    for exp in exp_list:
                        
                        bhi_med = np.median(bhi[obs][mod][exp])
                        b_med = np.median(b[obs][mod][exp])
                        blow_med = np.median(blow[obs][mod][exp])
                        pval_med = np.median(pval[obs][mod][exp])
                        var_fin[obs][mod][exp] = [bhi_med,
                                                  b_med,
                                                  blow_med,
                                                  pval_med]
                
                #==============================================================================
                    
                # continental analysis
                elif analysis == 'continental':
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    var_sfs[obs][mod] = {}
                    var_ctlruns[obs][mod] = {}
                    U[obs][mod] = {}
                    yc[obs][mod] = {}
                    Z1c[obs][mod] = {}
                    Z2c[obs][mod] = {}
                    Xc[obs][mod] = {}
                    Cf1[obs][mod] = {}
                    Ft[obs][mod] = {}
                    beta_hat[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = {}
                        b[obs][mod][exp] = {}
                        blow[obs][mod][exp] = {}
                        pval[obs][mod][exp] = {}
                        var_fin[obs][mod][exp] = {}
                        
                        for c in continents.keys():
                            
                            bhi[obs][mod][exp][c] = []
                            b[obs][mod][exp][c] = []
                            blow[obs][mod][exp][c] = []
                            pval[obs][mod][exp][c] = []
                        
                    for c in continents.keys():
                    
                        y = obs_data_continental[obs][mod][c]
                        X = fp_continental[mod][c]
                        if pi == 'model':
                            ctl = ctl_data_continental[mod][c]
                        elif pi == 'allpi':
                            ctl = ctl_data_continental[c]
                        nb_runs_x= nx[mod]
                        ns = len(continents[c])
                    
                        if bs_reps == 0: # for no bs, run ROF once
                        
                            bs_reps += 1
                        
                        for i in np.arange(0,bs_reps):
                            
                            # shuffle rows of ctl
                            ctl = np.take(ctl,
                                          np.random.permutation(ctl.shape[0]),
                                          axis=0)
                            
                            # run detection and attribution
                            var_sfs[obs][mod][c],\
                            var_ctlruns[obs][mod][c],\
                            proj,\
                            U[obs][mod][c],\
                            yc[obs][mod][c],\
                            Z1c[obs][mod][c],\
                            Z2c[obs][mod][c],\
                            Xc[obs][mod][c],\
                            Cf1[obs][mod][c],\
                            Ft[obs][mod][c],\
                            beta_hat[obs][mod][c] = da_run(y,
                                                           X,
                                                           ctl,
                                                           nb_runs_x,
                                                           ns,
                                                           nt,
                                                           reg,
                                                           cons_test,
                                                           formule_ic_tls,
                                                           trunc,
                                                           ci_bnds)
                            
                            # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                            for i,exp in enumerate(exp_list):
                                
                                bhi[obs][mod][exp][c].append(var_sfs[obs][mod][c][2,i])
                                b[obs][mod][exp][c].append(var_sfs[obs][mod][c][1,i])
                                blow[obs][mod][exp][c].append(var_sfs[obs][mod][c][0,i])
                                pval[obs][mod][exp][c].append(var_sfs[obs][mod][c][3,i])
                        
                    for exp in exp_list:
                        
                        for c in continents.keys():
                            
                            bhi_med = np.median(bhi[obs][mod][exp][c])
                            b_med = np.median(b[obs][mod][exp][c])
                            blow_med = np.median(blow[obs][mod][exp][c])
                            pval_med = np.median(pval[obs][mod][exp][c])
                            var_fin[obs][mod][exp][c] = [bhi_med,
                                                         b_med,
                                                         blow_med,
                                                         pval_med]
                                
                #==============================================================================
                    
                # continental analysis
                elif analysis == 'ar6':
                    
                    bhi[obs][mod] = {}
                    b[obs][mod] = {}
                    blow[obs][mod] = {}
                    pval[obs][mod] = {}
                    var_fin[obs][mod] = {}
                    var_sfs[obs][mod] = {}
                    var_ctlruns[obs][mod] = {}
                    U[obs][mod] = {}
                    yc[obs][mod] = {}
                    Z1c[obs][mod] = {}
                    Z2c[obs][mod] = {}
                    Xc[obs][mod] = {}
                    Cf1[obs][mod] = {}
                    Ft[obs][mod] = {}
                    beta_hat[obs][mod] = {}
                    
                    for exp in exp_list:
                        
                        bhi[obs][mod][exp] = {}
                        b[obs][mod][exp] = {}
                        blow[obs][mod][exp] = {}
                        pval[obs][mod][exp] = {}
                        var_fin[obs][mod][exp] = {}
                        
                        for c in continents.keys():
                            
                            for ar6 in continents[c]:
                            
                                bhi[obs][mod][exp][ar6] = []
                                b[obs][mod][exp][ar6] = []
                                blow[obs][mod][exp][ar6] = []
                                pval[obs][mod][exp][ar6] = []
                        
                    for c in continents.keys():
                        
                        for ar6 in continents[c]:
                    
                            y = obs_data_ar6[obs][mod][ar6]
                            X = fp_ar6[mod][ar6]
                            if pi == 'model':
                                ctl = ctl_data_ar6[mod][ar6]
                            elif pi == 'allpi':
                                ctl = ctl_data_ar6[ar6]
                            nb_runs_x= nx[mod]
                            ns = 1
                        
                            if bs_reps == 0: # for no bs, run ROF once
                            
                                bs_reps += 1
                            
                            for i in np.arange(0,bs_reps):
                                
                                # shuffle rows of ctl
                                ctl = np.take(ctl,
                                            np.random.permutation(ctl.shape[0]),
                                            axis=0)
                                
                                # run detection and attribution
                                var_sfs[obs][mod][ar6],\
                                var_ctlruns[obs][mod][ar6],\
                                proj,\
                                U[obs][mod][ar6],\
                                yc[obs][mod][ar6],\
                                Z1c[obs][mod][ar6],\
                                Z2c[obs][mod][ar6],\
                                Xc[obs][mod][ar6],\
                                Cf1[obs][mod][ar6],\
                                Ft[obs][mod][ar6],\
                                beta_hat[obs][mod][ar6] = da_run(y,
                                                                X,
                                                                ctl,
                                                                nb_runs_x,
                                                                ns,
                                                                nt,
                                                                reg,
                                                                cons_test,
                                                                formule_ic_tls,
                                                                trunc,
                                                                ci_bnds)
                                
                                # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                                for i,exp in enumerate(exp_list):
                                    
                                    bhi[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][2,i])
                                    b[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][1,i])
                                    blow[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][0,i])
                                    pval[obs][mod][exp][ar6].append(var_sfs[obs][mod][ar6][3,i])
                            
                    for exp in exp_list:
                        
                        for c in continents.keys():
                            
                            for ar6 in continents[c]:
                            
                                bhi_med = np.median(bhi[obs][mod][exp][ar6])
                                b_med = np.median(b[obs][mod][exp][ar6])
                                blow_med = np.median(blow[obs][mod][exp][ar6])
                                pval_med = np.median(pval[obs][mod][exp][ar6])
                                var_fin[obs][mod][exp][ar6] = [bhi_med,
                                                            b_med,
                                                            blow_med,
                                                            pval_med]

    return var_sfs,var_ctlruns,proj,U,yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat,var_fin,models

# %%
