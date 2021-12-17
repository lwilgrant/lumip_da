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
def pca_subroutine(lulcc,
                   models,
                   maps,
                   mod_ens,
                   mod_data,
                   pi_data,
                   continents,
                   lat_ranges,
                   ar6_regs,
                   scale,
                   flag_inverse):

    mod_msk = {}

    for mod in models: 
        
        mod_msk[mod] = {}
        
        for lc in lulcc:
        
            arr1 = cp.deepcopy(mod_ens[mod]['lu'].isel(time=0)).drop('time')
            arr2 = cp.deepcopy(maps[mod][lc].mean(dim='time'))
            arr1 = arr1.fillna(-999)
            arr2 = arr2.fillna(-999)
            alt_msk = xr.where(arr2!=-999,1,0)
            mod_msk[mod][lc] = xr.where(arr1!=-999,1,0).where(alt_msk==1).fillna(0)

    solver_dict = {}
    eof_dict = {}
    pc = {}
    pspc = {}
    weighted_arr = {}

    for mod in models:
        
        solver_dict[mod] = {}
        eof_dict[mod] = {}
        pc[mod] = {}
        pspc[mod] = {}
        weights = np.cos(np.deg2rad(mod_ens[mod]['lu'].lat))
        weighted_arr[mod] = xr.zeros_like(mod_ens[mod]['lu']).isel(time=0)    
        x = weighted_arr[mod].coords['lon']
        
        for y in weighted_arr[mod].lat.values:
            
            weighted_arr[mod].loc[dict(lon=x,lat=y)] = weights.sel(lat=y).item()
            
        
    if scale == "global":
        
        for mod in models:
            
            if flag_inverse == 0:
        
                for lc in lulcc:

                    solver_dict[mod][lc] = Eof(maps[mod][lc].where(mod_msk[mod][lc]==1),
                                                weights=weighted_arr[mod])
                    eof_dict[mod][lc] = solver_dict[mod][lc].eofs(neofs=1)
                    pc[mod][lc] = solver_dict[mod][lc].pcs(npcs=1)
                    pspc[mod][lc] = {}
                    pspc[mod][lc]['lu'] = solver_dict[mod][lc].projectField(mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1),
                                                                                                     neofs=1)
                    pspc[mod][lc]['lu_rls'] = []
                    pspc[mod][lc]['pic_rls'] = []
                    
                    for i in mod_data[mod]['lu'].rls:
                        
                        pspc[mod][lc]['lu_rls'].append(
                            solver_dict[mod][lc].projectField(
                                mod_data[mod]['lu'].sel(rls=i).where(mod_msk[mod][lc]==1),
                                neofs=1))
                    
                    for i in pi_data[mod].rls:
                        
                        pspc[mod][lc]['pic_rls'].append(solver_dict[mod][lc].projectField(pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1),
                                                                                                            neofs=1))

            elif flag_inverse == 1:
                
                lc = 'treeFrac'
                
                startmap = maps[mod][lc].where(mod_msk[mod][lc]==1)
                
                for fp in ['treeFrac', 'treeFrac_inv']:
                    
                    if fp == 'treeFrac_inv':
                        
                        startmap *= -1
                        
                    else:
                        
                        pass

                    solver_dict[mod][fp] = Eof(startmap,
                                               weights=weighted_arr[mod])        
                        
                    eof_dict[mod][fp] = solver_dict[mod][fp].eofs(neofs=1)
                    pc[mod][fp] = solver_dict[mod][fp].pcs(npcs=1)
                    pspc[mod][fp] = {}
                    pspc[mod][fp]['lu'] = solver_dict[mod][fp].projectField(mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1),
                                                                                                     neofs=1)
                    pspc[mod][fp]['lu_rls'] = []
                    pspc[mod][fp]['pic_rls'] = []
                    
                    for i in mod_data[mod]['lu'].rls:
                        
                        pspc[mod][fp]['lu_rls'].append(
                            solver_dict[mod][fp].projectField(
                                mod_data[mod]['lu'].sel(rls=i).where(mod_msk[mod][lc]==1),
                                neofs=1))                        
                    
                    for i in pi_data[mod].rls:
                        
                        pspc[mod][fp]['pic_rls'].append(solver_dict[mod][fp].projectField(pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1),
                                                                                                            neofs=1))             

    # latitudinal pca            
    elif scale == 'latitudinal':
        
        for mod in models:
            
            if flag_inverse == 0:
                
                for lc in lulcc:
                        
                    solver_dict[mod][lc] = {}
                    eof_dict[mod][lc] = {}
                    pc[mod][lc] = {}
                    pspc[mod][lc] = {}
                    
                    for ltr in lat_ranges.keys():
                        
                        luh2_slice = maps[mod][lc].where(mod_msk[mod][lc]==1)
                        luh2_slice = luh2_slice.sel(lat=lat_ranges[ltr])
                        solver_dict[mod][lc][ltr] = Eof(luh2_slice,
                                                        weights=weighted_arr[mod].sel(lat=lat_ranges[ltr]))
                        eof_dict[mod][lc][ltr] = solver_dict[mod][lc][ltr].eofs(neofs=1)
                        pc[mod][lc][ltr] = solver_dict[mod][lc][ltr].pcs(npcs=1)
                        pspc[mod][lc][ltr] = {}
                        
                        mod_slice = mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1)
                        mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                        pspc[mod][lc][ltr]['lu'] = solver_dict[mod][lc][ltr].projectField(mod_slice,
                                                                                          neofs=1)
                        pspc[mod][lc][ltr]['pic_rls'] = []

                        for i in pi_data[mod].rls:

                            pi_slice = pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1)
                            pi_slice = pi_slice.sel(lat=lat_ranges[ltr])
                            pspc[mod][lc][ltr]['pic_rls'].append(solver_dict[mod][lc][ltr].projectField(pi_slice,
                                                                                                        neofs=1))
                            
            elif flag_inverse == 1:
                
                for fp in ['treeFrac', 'treeFrac_inv']:
                        
                    solver_dict[mod][fp] = {}
                    eof_dict[mod][fp] = {}
                    pc[mod][fp] = {}
                    pspc[mod][fp] = {}
                    
                    lc = 'treeFrac'
                    startmap = maps[mod][lc].where(mod_msk[mod][lc]==1)  
                    
                    if fp == 'treeFrac_inv':
                        
                        startmap *= -1
                        
                    else:
                        
                        pass
                    
                    for ltr in lat_ranges.keys():
                        
                        latmap = startmap.sel(lat=lat_ranges[ltr])
                        solver_dict[mod][fp][ltr] = Eof(deepcopy(latmap),
                                                        weights=weighted_arr[mod].sel(lat=lat_ranges[ltr]))
                        eof_dict[mod][fp][ltr] = solver_dict[mod][fp][ltr].eofs(neofs=1)
                        pc[mod][fp][ltr] = solver_dict[mod][fp][ltr].pcs(npcs=1)
                        pspc[mod][fp][ltr] = {}
                        
                        mod_slice = mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1)
                        mod_slice = mod_slice.sel(lat=lat_ranges[ltr])
                        pspc[mod][fp][ltr]['lu'] = solver_dict[mod][fp][ltr].projectField(mod_slice,
                                                                                          neofs=1)
                        pspc[mod][fp][ltr]['lu_rls'] = []
                        pspc[mod][fp][ltr]['pic_rls'] = []
                        
                        for i in mod_data[mod]['lu'].rls:
                            
                            mod_rls_slice = mod_data[mod]['lu'].sel(rls=i).where(mod_msk[mod][lc]==1)
                            mod_rls_slice = mod_rls_slice.sel(lat=lat_ranges[ltr])
                            
                            pspc[mod][fp][ltr]['lu_rls'].append(
                                solver_dict[mod][fp][ltr].projectField(
                                    mod_rls_slice,
                                    neofs=1))                               

                        for i in pi_data[mod].rls:

                            pi_slice = pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1)
                            pi_slice = pi_slice.sel(lat=lat_ranges[ltr])
                            pspc[mod][fp][ltr]['pic_rls'].append(solver_dict[mod][fp][ltr].projectField(pi_slice,
                                                                                                        neofs=1)) 
                            
                            
                    # solver_dict[mod][fp] = Eof(startmap,
                    #                            weights=weighted_arr[mod])        
                        
                    # eof_dict[mod][fp] = solver_dict[mod][fp].eofs(neofs=1)
                    # pc[mod][fp] = solver_dict[mod][fp].pcs(npcs=1)
                    # pspc[mod][fp] = {}
                    # pspc[mod][fp]['lu'] = solver_dict[mod][fp].projectField(mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1),
                    #                                                                                  neofs=1)
                    # pspc[mod][fp]['lu_rls'] = []
                    # pspc[mod][fp]['pic_rls'] = []
                    
                    # for i in mod_data[mod]['lu'].rls:
                        
                    #     pspc[mod][fp]['lu_rls'].append(
                    #         solver_dict[mod][fp].projectField(
                    #             mod_data[mod]['lu'].sel(rls=i).where(mod_msk[mod][lc]==1),
                    #             neofs=1))                        
                    
                    # for i in pi_data[mod].rls:
                        
                    #     pspc[mod][fp]['pic_rls'].append(solver_dict[mod][fp].projectField(pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1),
                    #                                                                                         neofs=1))                       
                        
                        
                
                
    # continental pca            
    elif scale == 'continental':
        
        for mod in models:    
            
            for lc in lulcc:
                    
                solver_dict[mod][lc] = {}
                eof_dict[mod][lc] = {}
                pc[mod][lc] = {}
                pspc[mod][lc] = {}
                
                for c in continents.keys():
                    
                    continent = ar6_regs[mod].where(ar6_regs[mod].isin(continents[c]))
                    luh2_slice = maps[mod][lc].where(mod_msk[mod][lc]==1)
                    luh2_slice = luh2_slice.where(continent > 0)
                    solver_dict[mod][lc][c] = Eof(luh2_slice,
                                                  weights=weighted_arr[mod])
                    eof_dict[mod][lc][c] = solver_dict[mod][lc][c].eofs(neofs=1)
                    pc[mod][lc][c] = solver_dict[mod][lc][c].pcs(npcs=1)
                    pspc[mod][lc][c] = {}
                    
                    mod_slice = mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1)
                    mod_slice = mod_slice.where(continent > 0)
                    pspc[mod][lc][c]['lu'] = solver_dict[mod][lc][c].projectField(mod_slice,
                                                                                                         neofs=1)
                    pspc[mod][lc][c]['pic_rls'] = []

                    for i in pi_data[mod].rls:
                        
                        pi_slice = pi_data[mod].sel(rls=i).where(mod_msk[mod][lc]==1)
                        pi_slice = pi_slice.where(continent > 0)
                        pspc[mod][lc][c]['pic_rls'].append(solver_dict[mod][lc][c].projectField(pi_slice,
                                                                                                                  neofs=1))
                           
    # ar6 pca            
    elif scale == 'ar6':
    
        for mod in models:
            
            for lc in lulcc:
                    
                solver_dict[mod][lc] = {}
                eof_dict[mod][lc] = {}
                pc[mod][lc] = {}
                pspc[mod][lc] = {}
                
                for c in continents.keys():

                    for i in continents[c]:
                        
                        luh2_slice = maps[mod][lc].where(mod_msk[mod][lc]==1)
                        luh2_slice = luh2_slice.where(ar6_regs[mod] == i)
                        solver_dict[mod][lc][i] = Eof(luh2_slice,
                                                      weights=weighted_arr[mod])
                        eof_dict[mod][lc][i] = solver_dict[mod][lc][i].eofs(neofs=1)
                        pc[mod][lc][i] = solver_dict[mod][lc][i].pcs(npcs=1)
                        pspc[mod][lc][i] = {}
                        
                        mod_slice = mod_ens[mod]['lu'].where(mod_msk[mod][lc]==1)
                        mod_slice = mod_slice.where(ar6_regs[mod] == i)
                        pspc[mod][lc][i]['lu'] = solver_dict[mod][lc][i].projectField(mod_slice,
                                                                                                             neofs=1)
                        pspc[mod][lc][i]['pic_rls'] = []
                        
                        for j in pi_data[mod].rls:
                            
                            pi_slice = pi_data[mod].sel(rls=j).where(mod_msk[mod][lc]==1)
                            pi_slice = pi_slice.where(ar6_regs[mod] == i)
                            pspc[mod][lc][i]['pic_rls'].append(solver_dict[mod][lc][i].projectField(pi_slice,
                                                                                                                      neofs=1))                            
                            

    return solver_dict,eof_dict,pc,pspc

        
# %%
