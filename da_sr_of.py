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
def of_subroutine(models,
                  nx,
                  analysis,
                  exps,
                  obs_data,
                  obs_data_continental,
                  fp,
                  fp_continental,
                  ctl_data,
                  ctl_data_continental,
                  bs_reps,
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

    # details check
    U = {}
    yc = {}
    Z1c = {}
    Z2c = {}
    Xc = {}
    Cf1 = {}
    Ft = {}
    beta_hat = {}

    models.append('mmm')   
    nx['mmm'] = [] 
    for mod in models:

        #==============================================================================
        
        # global analysis
        if analysis == "global":
            
            bhi[mod] = {}
            b[mod] = {}
            blow[mod] = {}
            pval[mod] = {}
            var_fin[mod] = {}
            
            for exp in exps:
                
                bhi[mod][exp] = []
                b[mod][exp] = []
                blow[mod][exp] = []
                pval[mod][exp] = []
            
            y = obs_data[mod]
            X = fp[mod]
            ctl = ctl_data[mod]
            nb_runs_x= nx[mod]
            
            if bs_reps == 0: # for no bs, run ROF once
            
                bs_reps += 1
            
            for i in np.arange(0,bs_reps):
                
                # shuffle rows of ctl
                ctl = np.take(ctl,
                              np.random.permutation(ctl.shape[0]),
                              axis=0)
                
                # run detection and attribution
                var_sfs[mod],\
                var_ctlruns[mod],\
                proj,\
                U[mod],\
                yc[mod],\
                Z1c[mod],\
                Z2c[mod],\
                Xc[mod],\
                Cf1[mod],\
                Ft[mod],\
                beta_hat[mod] = da_run(y,
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
                for i,exp in enumerate(exps):
                    
                    bhi[mod][exp].append(var_sfs[mod][2,i])
                    b[mod][exp].append(var_sfs[mod][1,i])
                    blow[mod][exp].append(var_sfs[mod][0,i])
                    pval[mod][exp].append(var_sfs[mod][3,i])
            
            for exp in exps:
                
                bhi_med = np.median(bhi[mod][exp])
                b_med = np.median(b[mod][exp])
                blow_med = np.median(blow[mod][exp])
                pval_med = np.median(pval[mod][exp])
                var_fin[mod][exp] = [bhi_med,
                                     b_med,
                                     blow_med,
                                     pval_med]
        
        #==============================================================================
            
        # continental analysis
        elif analysis == 'continental':
            
            bhi[mod] = {}
            b[mod] = {}
            blow[mod] = {}
            pval[mod] = {}
            var_fin[mod] = {}
            var_sfs[mod] = {}
            var_ctlruns[mod] = {}
            
            U[mod] = {}
            yc[mod] = {}
            Z1c[mod] = {}
            Z2c[mod] = {}
            Xc[mod] = {}
            Cf1[mod] = {}
            Ft[mod] = {}
            beta_hat[mod] = {}
            
            for exp in exps:
                
                bhi[mod][exp] = {}
                b[mod][exp] = {}
                blow[mod][exp] = {}
                pval[mod][exp] = {}
                var_fin[mod][exp] = {}
                
                for c in continents.keys():
                    
                    bhi[mod][exp][c] = []
                    b[mod][exp][c] = []
                    blow[mod][exp][c] = []
                    pval[mod][exp][c] = []
                
            for c in continents.keys():
            
                y = obs_data_continental[mod][c]
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
                    var_sfs[mod][c],\
                    var_ctlruns[mod][c],\
                    proj,\
                    U[mod][c],\
                    yc[mod][c],\
                    Z1c[mod][c],\
                    Z2c[mod][c],\
                    Xc[mod][c],\
                    Cf1[mod][c],\
                    Ft[mod][c],\
                    beta_hat[mod][c] = da_run(y,
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
                    for i,exp in enumerate(exps):
                        
                        bhi[mod][exp][c].append(var_sfs[mod][c][2,i])
                        b[mod][exp][c].append(var_sfs[mod][c][1,i])
                        blow[mod][exp][c].append(var_sfs[mod][c][0,i])
                        pval[mod][exp][c].append(var_sfs[mod][c][3,i])
                
                for exp in exps:
                    for c in continents.keys():
                        
                        bhi_med = np.median(bhi[mod][exp][c])
                        b_med = np.median(b[mod][exp][c])
                        blow_med = np.median(blow[mod][exp][c])
                        pval_med = np.median(pval[mod][exp][c])
                        var_fin[mod][exp][c] = [bhi_med,
                                                b_med,
                                                blow_med,
                                                pval_med]

    return var_sfs,\
            var_ctlruns,\
            proj,\
            U,\
            yc,\
            Z1c,\
            Z2c,\
            Xc,\
            Cf1,\
            Ft,\
            beta_hat,\
            var_fin
