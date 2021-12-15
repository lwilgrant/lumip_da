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
from eofs.xarray import Eof
from pca_funcs import *


#%%============================================================================

# mod data
def pca_sn(scale,
           models,
           lulcc,
           mod_data,
           pi_data,
           continents,
           lat_ranges,
           pspc,
           pc,
           flag_inverse):


    if scale == 'global':
        
        lu_S = np.empty((len(models),len(lulcc)))
        lc_S = np.empty((len(models),len(lulcc)))
        N = np.empty((len(models),len(lulcc)))
        lu_S_N = np.empty((len(models),len(lulcc)))
        lc_S_N = np.empty((len(models),len(lulcc)))
        
        if flag_inverse == 0:
            
            pass
        
        elif flag_inverse == 1:
            
            lulcc = ['treeFrac', 'treeFrac_inv']

        sig_noise = xr.Dataset(
            
            data_vars={
                'lu_S': (['models','landcover'],lu_S),
                'lc_S': (['models','landcover'],lc_S),
                'N': (['models','landcover'],N),
                'lu_S_N': (['models','landcover'],lu_S_N),
                'lc_S_N': (['models','landcover'],lc_S_N),
                
                'lu_rls_S_{}'.format(models[0]): (['lu_rls_{}'.format(models[0]),'landcover'],
                                                  np.empty((len(mod_data[models[0]]['lu']),len(lulcc)))),
                'lu_rls_S_{}'.format(models[1]): (['lu_rls_{}'.format(models[1]),'landcover'],
                                                  np.empty((len(mod_data[models[1]]['lu']),len(lulcc)))),
                'lu_rls_S_{}'.format(models[2]): (['lu_rls_{}'.format(models[2]),'landcover'],
                                                  np.empty((len(mod_data[models[2]]['lu']),len(lulcc)))),
                'lu_rls_S_{}'.format(models[3]): (['lu_rls_{}'.format(models[3]),'landcover'],
                                                  np.empty((len(mod_data[models[3]]['lu']),len(lulcc)))),
                
                'lu_rls_S_N_{}'.format(models[0]): (['lu_rls_{}'.format(models[0]),'landcover'],
                                                    np.empty((len(mod_data[models[0]]['lu']),len(lulcc)))),
                'lu_rls_S_N_{}'.format(models[1]): (['lu_rls_{}'.format(models[1]),'landcover'],
                                                    np.empty((len(mod_data[models[1]]['lu']),len(lulcc)))),
                'lu_rls_S_N_{}'.format(models[2]): (['lu_rls_{}'.format(models[2]),'landcover'],
                                                    np.empty((len(mod_data[models[2]]['lu']),len(lulcc)))),
                'lu_rls_S_N_{}'.format(models[3]): (['lu_rls_{}'.format(models[3]),'landcover'],
                                                    np.empty((len(mod_data[models[3]]['lu']),len(lulcc)))),
                
                # CHANGE TO FOR MOD IN MODELS OUTSIDE DATASET CALL TO SHORTEN
                'pic_S_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],
                                               np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover']
                                               ,np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover']
                                               ,np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover']
                                               ,np.empty((len(pi_data[models[3]]),len(lulcc)))),
                
                'pic_S_N_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],
                                                 np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],
                                                 np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],
                                                 np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],
                                                 np.empty((len(pi_data[models[3]]),len(lulcc)))),
            },
            
            coords={
                'models': ('models', models),
                'landcover': ('landcover', lulcc),
                
                'lu_rls_{}'.format(models[0]): ('lu_rls_{}'.format(models[0]),range(len(mod_data[models[0]]['lu']))),
                'lu_rls_{}'.format(models[1]): ('lu_rls_{}'.format(models[1]),range(len(mod_data[models[1]]['lu']))),
                'lu_rls_{}'.format(models[2]): ('lu_rls_{}'.format(models[2]),range(len(mod_data[models[2]]['lu']))),
                'lu_rls_{}'.format(models[3]): ('lu_rls_{}'.format(models[3]),range(len(mod_data[models[3]]['lu']))),
                
                                
                'pic_rls_{}'.format(models[0]): ('pic_rls_{}'.format(models[0]),range(len(pi_data[models[0]]))),
                'pic_rls_{}'.format(models[1]): ('pic_rls_{}'.format(models[1]),range(len(pi_data[models[1]]))),
                'pic_rls_{}'.format(models[2]): ('pic_rls_{}'.format(models[2]),range(len(pi_data[models[2]]))),
                'pic_rls_{}'.format(models[3]): ('pic_rls_{}'.format(models[3]),range(len(pi_data[models[3]])))
            }
        )

        for mod in models:
                
            for lc in lulcc:
                
                # pic slopes
                pic_slope = []
                
                for i in pspc[mod][lc]['pi_rls']:
                    
                    slope = i.polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    i.attrs['slope'] = slope
                    pic_slope.append(slope)
                    
                sig_noise['pic_S_{}'.format(mod)].loc[dict(
                        landcover=lc
                    )] = pic_slope
                
                # noise N
                N = sig_noise['pic_S_{}'.format(mod)].sel(landcover=lc).std(dim='pic_rls_{}'.format(mod))
                
                # pic signal to noise ratio
                sig_noise['pic_S_N_{}'.format(mod)].loc[dict(
                        landcover=lc
                    )] = pic_slope / N.values
                
                # lu slopes
                lu_slope = []
                
                # individual lu realisations
                for i in pspc[mod][lc]['lu_rls']:
                    
                    slope = i.polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    i.attrs['slope'] = slope
                    lu_slope.append(slope)  
                    
                sig_noise['lu_rls_S_{}'.format(mod)].loc[dict(
                        landcover=lc
                    )] = lu_slope
                
                sig_noise['lu_rls_S_N_{}'.format(mod)].loc[dict(
                        landcover=lc
                    )] = lu_slope / N.values            
                
                # lu
                slope = pspc[mod][lc]['lu'].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                sig_noise['lu_S'].loc[dict(
                        models=mod,
                        landcover=lc
                    )] = slope
                
                # lu signal to noise ratio
                sig_noise['lu_S_N'].loc[dict(
                        models=mod,
                        landcover=lc
                    )] = slope / N.values
                
                # lc slopes
                slope = pc[mod][lc].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                sig_noise['lc_S'].loc[dict(
                        models=mod,
                        landcover=lc
                    )] = slope
                
                # lu signal to noise ratio
                sig_noise['lc_S_N'].loc[dict(
                        models=mod,
                        landcover=lc
                    )] = slope / N.values           

    elif scale == 'latitudinal':
        
        lu_S = np.empty((len(models),len(lulcc)))
        lc_S = np.empty((len(models),len(lulcc)))
        N = np.empty((len(models),len(lulcc)))
        lu_S_N = np.empty((len(models),len(lulcc)))
        lc_S_N = np.empty((len(models),len(lulcc)))

        sig_noise = xr.Dataset(
            
            data_vars={
                'lu_S': (['models','landcover'],lu_S),
                'lc_S': (['models','landcover'],lc_S),
                'N': (['models','landcover'],N),
                'lu_S_N': (['models','landcover'],lu_S_N),
                'lc_S_N': (['models','landcover'],lc_S_N),
                'pic_S_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
            },
            
            coords={
                'models': ('models', models),
                'landcover': ('landcover', lulcc),
                'pic_rls_{}'.format(models[0]): ('pic_rls_{}'.format(models[0]),range(len(pi_data[models[0]]))),
                'pic_rls_{}'.format(models[1]): ('pic_rls_{}'.format(models[1]),range(len(pi_data[models[1]]))),
                'pic_rls_{}'.format(models[2]): ('pic_rls_{}'.format(models[2]),range(len(pi_data[models[2]]))),
                'pic_rls_{}'.format(models[3]): ('pic_rls_{}'.format(models[3]),range(len(pi_data[models[3]]))),
            }
        )

        for ltr in lat_ranges.keys():

            for mod in models:
                
                for lc in lulcc:
                    
                    # pic slopes
                    pic_slope = []
                    
                    for i in pspc[mod][lc][ltr]['pi_rls']:
                        
                        slope = i.polyfit('time',1)['polyfit_coefficients'][0].values.item()
                        i.attrs['slope'] = slope
                        pic_slope.append(slope)
                        
                    sig_noise['pic_S_{}'.format(mod)].loc[dict(
                            landcover=lc
                        )] = pic_slope
                    
                    # noise N
                    N = sig_noise['pic_S_{}'.format(mod)].sel(landcover=lc).std(dim='pic_rls_{}'.format(mod))
                    
                    # pic signal to noise ratio
                    sig_noise['pic_S_N_{}'.format(mod)].loc[dict(
                            landcover=lc
                        )] = pic_slope / N.values
                    
                    # lu slopes
                    slope = pspc[mod][lc][ltr]['lu'].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    sig_noise['lu_S'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope
                    
                    # lu signal to noise ratio
                    sig_noise['lu_S_N'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope / N.values
                    
                    # lc slopes
                    slope = pc[mod][lc][ltr].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    sig_noise['lc_S'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope
                    
                    # lu signal to noise ratio
                    sig_noise['lc_S_N'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope / N.values

    elif scale == 'continental':
        
        lu_S = np.empty((len(models),len(lulcc)))
        lc_S = np.empty((len(models),len(lulcc)))
        N = np.empty((len(models),len(lulcc)))
        lu_S_N = np.empty((len(models),len(lulcc)))
        lc_S_N = np.empty((len(models),len(lulcc)))

        sig_noise = xr.Dataset(
            
            data_vars={
                'lu_S': (['models','landcover'],lu_S),
                'lc_S': (['models','landcover'],lc_S),
                'N': (['models','landcover'],N),
                'lu_S_N': (['models','landcover'],lu_S_N),
                'lc_S_N': (['models','landcover'],lc_S_N),
                'pic_S_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
            },
            
            coords={
                'models': ('models', models),
                'landcover': ('landcover', lulcc),
                'pic_rls_{}'.format(models[0]): ('pic_rls_{}'.format(models[0]),range(len(pi_data[models[0]]))),
                'pic_rls_{}'.format(models[1]): ('pic_rls_{}'.format(models[1]),range(len(pi_data[models[1]]))),
                'pic_rls_{}'.format(models[2]): ('pic_rls_{}'.format(models[2]),range(len(pi_data[models[2]]))),
                'pic_rls_{}'.format(models[3]): ('pic_rls_{}'.format(models[3]),range(len(pi_data[models[3]]))),
            }
        )

        for c in continents.keys():

            for mod in models:
                
                for lc in lulcc:
                    
                    # pic slopes
                    pic_slope = []
                    
                    for i in pspc[mod][lc][c]['pi_rls']:
                        
                        slope = i.polyfit('time',1)['polyfit_coefficients'][0].values.item()
                        i.attrs['slope'] = slope
                        pic_slope.append(slope)
                        
                    sig_noise['pic_S_{}'.format(mod)].loc[dict(
                            landcover=lc
                        )] = pic_slope
                    
                    # noise N
                    N = sig_noise['pic_S_{}'.format(mod)].sel(landcover=lc).std(dim='pic_rls_{}'.format(mod))
                    
                    # pic signal to noise ratio
                    sig_noise['pic_S_N_{}'.format(mod)].loc[dict(
                            landcover=lc
                        )] = pic_slope / N.values
                    
                    # lu slopes
                    slope = pspc[mod][lc][c]['lu'].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    sig_noise['lu_S'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope
                    
                    # lu signal to noise ratio
                    sig_noise['lu_S_N'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope / N.values
                    
                    # lc slopes
                    slope = pc[mod][lc][c].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                    sig_noise['lc_S'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope
                    
                    # lu signal to noise ratio
                    sig_noise['lc_S_N'].loc[dict(
                            models=mod,
                            landcover=lc
                        )] = slope / N.values
            
    elif scale == 'ar6':
        
        lu_S = np.empty((len(models),len(lulcc)))
        lc_S = np.empty((len(models),len(lulcc)))
        N = np.empty((len(models),len(lulcc)))
        lu_S_N = np.empty((len(models),len(lulcc)))
        lc_S_N = np.empty((len(models),len(lulcc)))

        sig_noise = xr.Dataset(
            
            data_vars={
                'lu_S': (['models','landcover'],lu_S),
                'lc_S': (['models','landcover'],lc_S),
                'N': (['models','landcover'],N),
                'lu_S_N': (['models','landcover'],lu_S_N),
                'lc_S_N': (['models','landcover'],lc_S_N),
                'pic_S_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[0]): (['pic_rls_{}'.format(models[0]),'landcover'],np.empty((len(pi_data[models[0]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[1]): (['pic_rls_{}'.format(models[1]),'landcover'],np.empty((len(pi_data[models[1]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[2]): (['pic_rls_{}'.format(models[2]),'landcover'],np.empty((len(pi_data[models[2]]),len(lulcc)))),
                'pic_S_N_{}'.format(models[3]): (['pic_rls_{}'.format(models[3]),'landcover'],np.empty((len(pi_data[models[3]]),len(lulcc)))),
            },
            
            coords={
                'models': ('models', models),
                'landcover': ('landcover', lulcc),
                'pic_rls_{}'.format(models[0]): ('pic_rls_{}'.format(models[0]),range(len(pi_data[models[0]]))),
                'pic_rls_{}'.format(models[1]): ('pic_rls_{}'.format(models[1]),range(len(pi_data[models[1]]))),
                'pic_rls_{}'.format(models[2]): ('pic_rls_{}'.format(models[2]),range(len(pi_data[models[2]]))),
                'pic_rls_{}'.format(models[3]): ('pic_rls_{}'.format(models[3]),range(len(pi_data[models[3]]))),
            }
        )

        for c in continents.keys():
            
            for ar6 in continents[c]:

                for mod in models:
                    
                    for lc in lulcc:
                        
                        # pic slopes
                        pic_slope = []
                        
                        for i in pspc[mod][lc][ar6]['pi_rls']:
                            
                            slope = i.polyfit('time',1)['polyfit_coefficients'][0].values.item()
                            i.attrs['slope'] = slope
                            pic_slope.append(slope)
                            
                        sig_noise['pic_S_{}'.format(mod)].loc[dict(
                                landcover=lc
                            )] = pic_slope
                        
                        # noise N
                        N = sig_noise['pic_S_{}'.format(mod)].sel(landcover=lc).std(dim='pic_rls_{}'.format(mod))
                        
                        # pic signal to noise ratio
                        sig_noise['pic_S_N_{}'.format(mod)].loc[dict(
                                landcover=lc
                            )] = pic_slope / N.values
                        
                        # lu slopes
                        slope = pspc[mod][lc][ar6]['lu'].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                        sig_noise['lu_S'].loc[dict(
                                models=mod,
                                landcover=lc
                            )] = slope
                        
                        # lu signal to noise ratio
                        sig_noise['lu_S_N'].loc[dict(
                                models=mod,
                                landcover=lc
                            )] = slope / N.values
                        
                        # lc slopes
                        slope = pc[mod][lc][ar6].polyfit('time',1)['polyfit_coefficients'][0].values.item()
                        sig_noise['lc_S'].loc[dict(
                                models=mod,
                                landcover=lc
                            )] = slope
                        
                        # lu signal to noise ratio
                        sig_noise['lc_S_N'].loc[dict(
                                models=mod,
                                landcover=lc
                            )] = slope / N.values
                        
    return sig_noise,lulcc

# %%
