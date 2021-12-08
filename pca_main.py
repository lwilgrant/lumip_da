#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# PCA analysis
# update November 13:
    # adding subroutines from da*.py to here; adjusting for this analysis
    # instead of making 2d arrays for ar6 organization, make new data arrays (xr) for ar6 means
    # run pca on global series of ar6 means as well as earlier scales
        # potential for ar6-weighted mean run to then show likewise regions from OF contributing to signal 
        # (e.g. detected regions from OF will have high loadings)
    # check that eofs of individual model means differ between historical and hist-noLu

# repeat latitudinal, continental etc options for luh2
# pseudo-pcs differ for 2nd eof (proj obs onto hist vs histnolu)
    # n-temperate hist has more cooling in NA, for example, but histnolu has morxe cooling in China
    # project LU onto 2nd eof for historical vs hist-nolu?

# removed pi files from fp_sr_file_alloc temporarily, until I get new organization with sr's done
# send modres crop and deforest bash scripts for redoing lu/lulcc eofs ()

#%%==============================================================================
# import
# ===============================================================================


import sys
import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
import xarray as xr
from scipy import stats as sts
import seaborn as sns
from eofs.xarray import Eof


#%%==============================================================================
# path
#================================================================================


# curDIR = '/home/luke/documents/lumip/d_a/'
# curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
# curDIR = '/Users/Luke/Documents/PHD/lumip/da'
# curDIR = '//wsl$/Ubuntu//home//luke//mnt//c//Users//lgrant//documents//repos//lumip_da'
curDIR = 'C:/Users/lgrant/Documents/repos/lumip_da'
os.chdir(curDIR)

# data input directories
obsDIR = os.path.join(curDIR, 'obs')
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
mapDIR = os.path.join(curDIR, 'map')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from pca_funcs import *


#%%==============================================================================
# options - analysis
#================================================================================


# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

flag_pickle=1     # 0: do not pickle objects
                  # 1: pickle objects after sections 'read' and 'analyze'

# << SELECT >>
flag_svplt=1      # 0: do not save plot
                  # 1: save plot in picDIR
                  
# # << SELECT >>
# flag_lulcc=0      # 0: treeFrac
#                   # 1: cropFrac

# << SELECT >>
flag_lulcc_measure=0    # 0: absolute change
                        # 1: area change
                        
# << SELECT >>
flag_lulcc_stat=1    # 0: annual mean cover
                     # 1: annual max cover
                        
# << SELECT >>
flag_lu_technique=1     # 0: lu as mean of individual (historical - hist-nolu)
                        # 1: lu as mean(historical) - mean(hist-nolu)

# << SELECT >>
flag_y1=1         # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0        # 0: 50
                  # 1: 100

# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means
                   
# << SELECT >>
flag_temp_center=0 # 0: raw data input to EOF
                   # 1: temporal center data on read-in                   

# << SELECT >>
flag_var=0   # 0: tasmax
                   
# << SELECT >>
flag_standardize=1;  # 0: no (standardization before input to PCA and projections)
                     # 1: yes, standardize 
                     
# << SELECT >>
flag_scale=2;         # 0: global
                      # 1: latitudinal
                      # 2: continental
                      # 3: ar6 regions
                      
# << SELECT >>
flag_lulcc=0      # 0: forest loss
                  # 1: crop expansion                      
                   
# << SELECT >>
flag_correlation=0;  # 0: no
                     # 1: yes
                     
# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means                     
                  

# << SELECT >>
trunc=0


deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['treeFrac',
         'cropFrac']
lulcc_stats = ['mean',
               'max']
measures = ['absolute_change',
            'area_change',
            'all_pixels']
lu_techniques = ['individual',
                 'mean']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y']
variables = ['tasmax']
analyses = ['models',
            'luh2']
landcover_types = ['forest',
                   'crops']
standardize_opts = ['no',
                    'yes']
correlation_opts = ['no',
                    'yes']
scale_opts = ['global',
              'latitudinal',
              'continental',
              'ar6']
resample=['5Y',
          '10Y']

y1 = start_years[flag_y1]
length = lengths[flag_len]
var = variables[flag_var]
lulcc_type = lulcc[flag_lulcc]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
standardize = standardize_opts[flag_standardize]
correlation = correlation_opts[flag_correlation]
scale = scale_opts[flag_scale]
freq = resample[flag_resample]
stat = lulcc_stats[flag_lulcc_stat]


# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

exps_start = ['historical',
              'hist-noLu']

exps = ['historical',
        'hist-noLu',
        'lu']

continents = {}
continents['North America'] = [1,2,3,4,5,6,7]
continents['South America'] = [9,10,11,12,13,14,15]
continents['Europe'] = [16,17,18,19]
continents['Asia'] = [28,29,30,31,32,33,34,35,37,38]
continents['Africa'] = [21,22,23,24,25,26]
continents['Australia'] = [39,40,41,42]

ns = 0
continent_names = []
for c in continents.keys():
    continent_names.append(c)
    for i in continents[c]:
        ns += 1

labels = {}
labels['North America'] = ['WNA','CNA','ENA','NCA','SCA']
labels['South America'] = ['NWS','NSA','NES','SAM','SES']
labels['Europe'] = ['NEU','WCE','EEU','MED']
labels['Asia'] = ['WSB','ESB','WCA','EAS','SAS']
labels['Africa'] = ['WAF','CAF','NEAF','SEAF','SWAF','ESAF']
labels['Australia'] = ['NAU','CAU','EAU','SAU']

lat_ranges = {}
lat_ranges['boreal'] = slice(51.5,89.5)
lat_ranges['tropics'] = slice(-23.5,23.5)
lat_ranges['temperate_south'] = slice(-50.5,-24.5)
lat_ranges['temperate_north'] = slice(24.5,50.5)

letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']


#================================================================================
# mod + obs + luh2 data 
#================================================================================

#%%============================================================================
from pca_sr_file_alloc import *
map_files,grid_files,mod_files,pi_files = file_subroutine(mapDIR,
                                                          modDIR,
                                                          piDIR,
                                                          stat,
                                                          lulcc,
                                                          y1,
                                                          y2,
                                                          t_ext,
                                                          models,
                                                          exps,
                                                          var)

#%%============================================================================

# luh2 maps and ar6 regions
os.chdir(curDIR)
from pca_sr_maps import *
maps,ar6_regs,ar6_land = map_subroutine(map_files,
                                        models,
                                        mapDIR,
                                        flag_temp_center,
                                        flag_standardize,
                                        lulcc,
                                        y1,
                                        measure,
                                        freq)            

#%%============================================================================

# mod ensembles
os.chdir(curDIR)
from pca_sr_mod_ens import *
mod_ens = ensemble_subroutine(modDIR,
                              models,
                              exps,
                              flag_temp_center,
                              flag_standardize,
                              mod_files,
                              var,
                              lu_techn,
                              y1,
                              freq,
                              ar6_land)

#%%============================================================================

# pi data
os.chdir(curDIR)
from pca_sr_pi import *
pi_data = picontrol_subroutine(piDIR,
                               pi_files,
                               models,
                               flag_temp_center,
                               flag_standardize,
                               var,
                               y1,
                               freq,
                               ar6_land)
           
#%%============================================================================

# pca work
os.chdir(curDIR)
from pca_sr_eof_proj import *
solver_dict,eof_dict,pc,pspc = pca_subroutine(lulcc,
                                              models,
                                              maps,
                                              mod_ens,
                                              pi_data,
                                              continents,
                                              lat_ranges,
                                              ar6_regs,
                                              scale)

#%%============================================================================

if scale == 'global':
    
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
            'pic_rls_{}'.format(models[3]): ('pic_rls_{}'.format(models[3]),range(len(pi_data[models[3]])))
        }
    )

    for mod in models:
        
        for lc in lulcc:
            
            # pic slopes
            pic_slope = []
            
            for i in pspc[mod][lc]['pi']:
                
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

    x=10
    y=15
    eof_color='BrBG'
    cmap=plt.cm.get_cmap(eof_color)
    colors = {}
    colors['treeFrac'] = cmap(0.85)
    colors['lu_treeFrac'] = cmap(0.95)
    colors['cropFrac'] = cmap(0.15)
    colors['lu_cropFrac'] = cmap(0.05)
    le_x0 = 0.8
    le_y0 = 0.6
    le_xlen = 0.2
    le_ylen = 0.5

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 2

    # f,axes = plt.subplots(nrows=len(models),
    #                       ncols=len(lulcc),
    #                       figsize=(x,y))
    f,axes = plt.subplots(nrows=len(models),
                        ncols=1,
                        figsize=(x,y))
    i = 0
    for mod,ax in zip(models,axes):
    # for mod,row_axes in zip(models,axes):
        
        # for lc,ax in zip(lulcc,row_axes):
        for lc in lulcc:
            
            sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                        ax=ax,
                        fit=sts.norm,
                        fit_kws={"color":colors[lc]},
                        color = colors[lc],
                        label='PIC_{}'.format(lc),
                        kde=False)
            
            # plot lu s/n
            ax.vlines(x=sig_noise['lu_S_N'.format(mod)].sel(models=mod,landcover=lc),
                    ymin=0,
                    ymax=0.5,
                    colors=colors['lu_{}'.format(lc)],
                    label='lu_{}'.format(lc))
            
            # # plot lc s/n
            # ax.vlines(x=sig_noise['lc_S_N'.format(mod)].sel(models=mod,landcover=lc),
            #           ymin=0,
            #           ymax=a[0].max(),
            #           colors='blue')
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(None)
        ax.set_title(letters[i],
                    loc='left',
                    fontweight='bold')
        ax.set_xlabel(None)
        ax.xaxis.set_ticklabels([])
        ax.set_xlim(-4,4)
        
        if mod == 'CanESM5':
            height = 0.4
        else:
            height= 0.3

        ax.set_ylabel('Frequency',
                        fontsize=12)
        ax.text(-0.15,
                height,
                mod,
                fontweight='bold',
                rotation='vertical',
                transform=ax.transAxes) 
        
        if mod == models[-1]:
            ax.set_xlabel('S/N')
            ax.xaxis.set_ticklabels(np.arange(-4,5,1))
        
        if i == 0:
            ax.legend(frameon=False,
                    bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                    labelspacing=legend_entrypad)
            
        i += 1

    f.savefig(outDIR+'/pca_noise_global.png')

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
                
                for i in pspc[mod][lc][ltr]['pi']:
                    
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

        x=10
        y=15
        eof_color='BrBG'
        cmap=plt.cm.get_cmap(eof_color)
        colors = {}
        colors['treeFrac'] = cmap(0.85)
        colors['lu_treeFrac'] = cmap(0.95)
        colors['cropFrac'] = cmap(0.15)
        colors['lu_cropFrac'] = cmap(0.05)
        le_x0 = 0.8
        le_y0 = 0.6
        le_xlen = 0.2
        le_ylen = 0.5

        # space between entries
        legend_entrypad = 0.5

        # length per entry
        legend_entrylen = 2

        # f,axes = plt.subplots(nrows=len(models),
        #                       ncols=len(lulcc),
        #                       figsize=(x,y))
        f,axes = plt.subplots(nrows=len(models),
                            ncols=1,
                            figsize=(x,y))
        i = 0
        for mod,ax in zip(models,axes):
        # for mod,row_axes in zip(models,axes):
            
            # for lc,ax in zip(lulcc,row_axes):
            for lc in lulcc:
                
                sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                            ax=ax,
                            fit=sts.norm,
                            fit_kws={"color":colors[lc]},
                            color = colors[lc],
                            label='PIC_{}'.format(lc),
                            kde=False)
                
                # plot lu s/n
                ax.vlines(x=sig_noise['lu_S_N'.format(mod)].sel(models=mod,landcover=lc),
                        ymin=0,
                        ymax=0.5,
                        colors=colors['lu_{}'.format(lc)],
                        label='lu_{}'.format(lc))
                
                # # plot lc s/n
                # ax.vlines(x=sig_noise['lc_S_N'.format(mod)].sel(models=mod,landcover=lc),
                #           ymin=0,
                #           ymax=a[0].max(),
                #           colors='blue')
                
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(None)
            ax.set_title(letters[i],
                        loc='left',
                        fontweight='bold')
            ax.set_xlabel(None)
            ax.xaxis.set_ticklabels([])
            ax.set_xlim(-4,4)
            
            if mod == 'CanESM5':
                height = 0.4
            else:
                height= 0.3

            ax.set_ylabel('Frequency',
                            fontsize=12)
            ax.text(-0.15,
                    height,
                    mod,
                    fontweight='bold',
                    rotation='vertical',
                    transform=ax.transAxes) 
            
            if mod == models[-1]:
                ax.set_xlabel('S/N')
                ax.xaxis.set_ticklabels(np.arange(-4,5,1))
            
            if i == 0:
                ax.legend(frameon=False,
                        bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                        labelspacing=legend_entrypad)
                
            i += 1

        f.savefig(outDIR+'/pca_noise_{}.png'.format(ltr))

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
                
                for i in pspc[mod][lc][c]['pi']:
                    
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

        x=10
        y=15
        eof_color='BrBG'
        cmap=plt.cm.get_cmap(eof_color)
        colors = {}
        colors['treeFrac'] = cmap(0.85)
        colors['lu_treeFrac'] = cmap(0.95)
        colors['cropFrac'] = cmap(0.15)
        colors['lu_cropFrac'] = cmap(0.05)
        le_x0 = 0.8
        le_y0 = 0.6
        le_xlen = 0.2
        le_ylen = 0.5

        # space between entries
        legend_entrypad = 0.5

        # length per entry
        legend_entrylen = 2

        # f,axes = plt.subplots(nrows=len(models),
        #                       ncols=len(lulcc),
        #                       figsize=(x,y))
        f,axes = plt.subplots(nrows=len(models),
                            ncols=1,
                            figsize=(x,y))
        i = 0
        for mod,ax in zip(models,axes):
        # for mod,row_axes in zip(models,axes):
            
            # for lc,ax in zip(lulcc,row_axes):
            for lc in lulcc:
                
                sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                            ax=ax,
                            fit=sts.norm,
                            fit_kws={"color":colors[lc]},
                            color = colors[lc],
                            label='PIC_{}'.format(lc),
                            kde=False)
                
                # plot lu s/n
                ax.vlines(x=sig_noise['lu_S_N'.format(mod)].sel(models=mod,landcover=lc),
                        ymin=0,
                        ymax=0.5,
                        colors=colors['lu_{}'.format(lc)],
                        label='lu_{}'.format(lc))
                
                # # plot lc s/n
                # ax.vlines(x=sig_noise['lc_S_N'.format(mod)].sel(models=mod,landcover=lc),
                #           ymin=0,
                #           ymax=a[0].max(),
                #           colors='blue')
                
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(None)
            ax.set_title(letters[i],
                        loc='left',
                        fontweight='bold')
            ax.set_xlabel(None)
            ax.xaxis.set_ticklabels([])
            ax.set_xlim(-4,4)
            
            if mod == 'CanESM5':
                height = 0.4
            else:
                height= 0.3

            ax.set_ylabel('Frequency',
                            fontsize=12)
            ax.text(-0.15,
                    height,
                    mod,
                    fontweight='bold',
                    rotation='vertical',
                    transform=ax.transAxes) 
            
            if mod == models[-1]:
                ax.set_xlabel('S/N')
                ax.xaxis.set_ticklabels(np.arange(-4,5,1))
            
            if i == 0:
                ax.legend(frameon=False,
                        bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                        labelspacing=legend_entrypad)
                
            i += 1

        f.savefig(outDIR+'/pca_noise_{}.png'.format(c))
        
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
                    
                    for i in pspc[mod][lc][ar6]['pi']:
                        
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

            x=10
            y=15
            eof_color='BrBG'
            cmap=plt.cm.get_cmap(eof_color)
            colors = {}
            colors['treeFrac'] = cmap(0.85)
            colors['lu_treeFrac'] = cmap(0.95)
            colors['cropFrac'] = cmap(0.15)
            colors['lu_cropFrac'] = cmap(0.05)
            le_x0 = 0.8
            le_y0 = 0.6
            le_xlen = 0.2
            le_ylen = 0.5

            # space between entries
            legend_entrypad = 0.5

            # length per entry
            legend_entrylen = 2

            # f,axes = plt.subplots(nrows=len(models),
            #                       ncols=len(lulcc),
            #                       figsize=(x,y))
            f,axes = plt.subplots(nrows=len(models),
                                ncols=1,
                                figsize=(x,y))
            i = 0
            for mod,ax in zip(models,axes):
            # for mod,row_axes in zip(models,axes):
                
                # for lc,ax in zip(lulcc,row_axes):
                for lc in lulcc:
                    
                    sns.distplot(sig_noise['pic_S_N_{}'.format(mod)].sel(landcover=lc),
                                ax=ax,
                                fit=sts.norm,
                                fit_kws={"color":colors[lc]},
                                color = colors[lc],
                                label='PIC_{}'.format(lc),
                                kde=False)
                    
                    # plot lu s/n
                    ax.vlines(x=sig_noise['lu_S_N'.format(mod)].sel(models=mod,landcover=lc),
                            ymin=0,
                            ymax=0.5,
                            colors=colors['lu_{}'.format(lc)],
                            label='lu_{}'.format(lc))
                    
                    # # plot lc s/n
                    # ax.vlines(x=sig_noise['lc_S_N'.format(mod)].sel(models=mod,landcover=lc),
                    #           ymin=0,
                    #           ymax=a[0].max(),
                    #           colors='blue')
                    
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_title(None)
                ax.set_title(letters[i],
                            loc='left',
                            fontweight='bold')
                ax.set_xlabel(None)
                ax.xaxis.set_ticklabels([])
                ax.set_xlim(-4,4)
                
                if mod == 'CanESM5':
                    height = 0.4
                else:
                    height= 0.3

                ax.set_ylabel('Frequency',
                                fontsize=12)
                ax.text(-0.15,
                        height,
                        mod,
                        fontweight='bold',
                        rotation='vertical',
                        transform=ax.transAxes) 
                
                if mod == models[-1]:
                    ax.set_xlabel('S/N')
                    ax.xaxis.set_ticklabels(np.arange(-4,5,1))
                
                if i == 0:
                    ax.legend(frameon=False,
                            bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                            labelspacing=legend_entrypad)
                    
                i += 1

            f.savefig(outDIR+'/pca_noise_{}.png'.format(ar6))

    # %%
