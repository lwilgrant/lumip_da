#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script generates correlation analysis results on model LU and LC transition data

# maps of trends
    # marked by significance
    # also at ar6 scale (and marked at centroids)
    # control map of mean pi trends with significance marking?
        # would help answer, can we have spurious trends of this significance with only noise?

# maps of correlation at pixel and ar6 scale (question over corr of mean series or mean of corr at pixel scale)
        
# use pixel scale trend info from pic and lu to compare their distributions
    # do this where correlation is good?
    # latitudinally? use trendist stuff, ar6?

#%%============================================================================
# import
# =============================================================================

import sys
import os
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import matplotlib as mpl
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D

#%%============================================================================
# path
#==============================================================================

# curDIR = '/home/luke/documents/lumip/d_a/'
# curDIR = '/theia/data/brussel/vo/000/bvo00012/vsc10116/lumip/d_a'
# curDIR = '/Users/Luke/Documents/PHD/lumip/da'
curDIR = r'C:/Users/lgrant/Documents/repos/lumip_da'
os.chdir(curDIR)

# data input directories
modDIR = os.path.join(curDIR, 'mod')
piDIR = os.path.join(curDIR, 'pi')
mapDIR = os.path.join(curDIR, 'map')
sfDIR = os.path.join(curDIR, 'shapefiles')
outDIR = os.path.join(curDIR, 'figures')

# bring in functions
from lu_funcs import *

#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
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
flag_lulcc_stat=0    # 0: annual mean cover
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
flag_var=0   # 0: tasmax


# << SELECT >>
trunc=0

seasons = ['jja',
           'djf',
           'annual',
           'max']
analyses = ['global',
            'continental',
            'ar6']
deforest_options = ['all',
                    'defor',
                    'ar6']
lulcc = ['treeFrac',
         'cropFrac']
lulcc_stats = ['mean',
               'max']
grids = ['model',
         'obs']
factors = [['hist-noLu','lu'],
           ['historical'],
           ['hist-noLu']]
obs_types = ['cru',
             'berkley_earth']
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

stat = lulcc_stats[flag_lulcc_stat]
measure = measures[flag_lulcc_measure]
lu_techn = lu_techniques[flag_lu_technique]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]

# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

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

continent_names = []
for c in continents.keys():
    continent_names.append(c)

labels = {}
labels['North America'] = ['WNA','CNA','ENA','SCA']
labels['South America'] = ['NWS','NSA','NES','SAM','SWS','SES']
labels['Europe'] = ['NEU','WCE','EEU','MED']
labels['Asia'] = ['WSB','ESB','TIB','EAS','SAS','SEA']
labels['Africa'] = ['WAF','CAF','NEAF','SEAF','ESAF']
    
ns = 0
for c in continents.keys():
    for i in continents[c]:
        ns += 1
        
letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']

#==============================================================================
# get data 
#==============================================================================

#%%============================================================================
from lu_sr_file_alloc import *
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
from lu_sr_maps import *
maps,ar6_regs,ar6_land = map_subroutine(map_files,
                                        models,
                                        mapDIR,
                                        lulcc,
                                        y1,
                                        measure,
                                        freq)            

#%%============================================================================

# mod ensembles
os.chdir(curDIR)
from lu_sr_mod_ens import *
mod_ens = ensemble_subroutine(modDIR,
                              models,
                              exps,
                              var,
                              lu_techn,
                              y1,
                              freq,
                              mod_files)

#%%============================================================================

# pi data
os.chdir(curDIR)
from lu_sr_pi import *
pi_data = picontrol_subroutine(piDIR,
                               pi_files,
                               models,
                               var,
                               y1,
                               freq,
                               maps)
           
#%%============================================================================

# trends and lu/lc corr
stats_ds = {}

for mod in models:    
    
    stats_ds[mod] = xr.Dataset(
        coords=dict(
            lat = ('lat',ar6_land[mod].lat.values),
            lon = ('lon',ar6_land[mod].lon.values)
        )
    )
    mod_slope,mod_p,_ = vectorize_lreg(mod_ens[mod]['lu'])
    stats_ds[mod]['lu_slope'] = mod_slope
    stats_ds[mod]['lu_p'] = mod_p # maybe change to 0/1 for significance
    
    for lu in lulcc:
        
        # trends
        lc_slope,lc_p,_ = vectorize_lreg(maps[mod][lu])
        stats_ds[mod]['{}_slope'.format(lu)] = lc_slope
        stats_ds[mod]['{}_p'.format(lu)] = lc_p # maybe change to 0/1 for significance
        
        # correlation
        _,_,corr = vectorize_lreg(mod_ens[mod]['lu'],
                                  da_x=maps[mod][lu])
        stats_ds[mod]['lu-{}_corr'.format(lu)] = corr
        

#%%============================================================================


col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.75   # linewidth for coastlines

# fonts
title_font = 20
cbtitle_font = 20
tick_font = 18
legend_font=12

x=25
y=15

# placment lu trends cbar
cb_lu_x0 = 0.1275
cb_lu_y0 = 0.05
cb_lu_xlen = 0.225
cb_lu_ylen = 0.015

# boundaries of 0 for lu
null_bnds_lu = [-0.0050,
                0.0050]

# placment lc trends cbar
cb_lc_x0 = 0.43
cb_lc_y0 = 0.05
cb_lc_xlen = 0.45
cb_lc_ylen = 0.015

# boundaries of 0 for lc
null_bnds_lc = [-0.10,
                0.10]

# extent
east = 180
west = -180
north = 80
south = -60
extent = [west,east,south,north]

cmap_list_lu,tick_locs_lu,tick_labels_lu,norm_lu,levels_lu = colormap_details('RdBu',
                                                                              data_lumper(stats_ds,
                                                                                          models,
                                                                                          maptype='lu'),
                                                                              null_bnds_lu)

cmap_list_lc,tick_locs_lc,tick_labels_lc,norm_lc,levels_lc = colormap_details('BrBG_r',
                                                                              data_lumper(stats_ds,
                                                                                          models,
                                                                                          maptype='lc'),
                                                                              null_bnds_lc)

f, axes = plt.subplots(nrows=len(models),
                       ncols=3,
                       figsize=(x,y),
                       subplot_kw={'projection':ccrs.PlateCarree()})

cbax_lu = f.add_axes([cb_lu_x0, 
                      cb_lu_y0, 
                      cb_lu_xlen, 
                      cb_lu_ylen])
cbax_lc = f.add_axes([cb_lc_x0, 
                      cb_lc_y0, 
                      cb_lc_xlen, 
                      cb_lc_ylen])


# f, ax = plt.subplots(nrows=1,
#                        ncols=1,
#                        figsize=(x,y),
#                        subplot_kw={'projection':ccrs.PlateCarree()})
# cbax_lc = f.add_axes([cb_lc_x0, 
#                       cb_lc_y0, 
#                       cb_lc_xlen, 
#                       cb_lc_ylen])
# stats_ds['CanESM5']['treeFrac_slope'].plot(ax=ax,
#                                            cbar_ax = cbax_lc,
#                                            cmap=cmap_list_lc,
#                                            levels=levels_lc,
#                                            extend='both',
#                                            center=0)

i = 0


# stats_ds['CanESM5']['treeFrac_slope'] = xr.where((stats_ds['CanESM5']['treeFrac_slope']>null_bnds_lc[0])&\
#     (stats_ds['CanESM5']['treeFrac_slope']<null_bnds_lc[1]),
#     0,
#     stats_ds['CanESM5']['treeFrac_slope'])

# stats_ds['CanESM5']['treeFrac_slope'] = stats_ds['CanESM5']['treeFrac_slope'].where(ar6_land['CanESM5']==1)
    
    

for mod,row_axes in zip(models,axes):
    
    # stats_ds['CanESM5']['treeFrac_slope'].plot(cmap=cmap_list_lc,levels=levels_lc,extend='both',center=0)
    stats_ds[mod]['lu_slope'].plot(ax=row_axes[0],
                                   cmap=cmap_list_lu,
                                   cbar_ax=cbax_lu,
                                   levels=levels_lu,
                                   extend='both',
                                   center=0,
                                   add_labels=False)
    
    stats_ds[mod]['treeFrac_slope'].plot(ax=row_axes[1],
                                         cmap=cmap_list_lc,
                                         cbar_ax=cbax_lc,
                                         levels=levels_lc,
                                         extend='both',
                                         center=0,
                                         add_labels=False)
    
    stats_ds[mod]['cropFrac_slope'].plot(ax=row_axes[2],
                                         cmap=cmap_list_lc,
                                         cbar_ax=cbax_lc,
                                         levels=levels_lc,
                                         extend='both',
                                         center=0,
                                         add_labels=False)
    
    for ax,column in zip(row_axes,['LU response','treeFrac','cropFrac']):
        
        ax.set_extent(extent,
                      crs=ccrs.PlateCarree())
        ax.set_title(letters[i],
                     loc='left',
                     fontsize = title_font,
                     fontweight='bold')
        ax.coastlines(linewidth=cstlin_lw)
        
        if column == 'LU response':
            
                if mod == 'CanESM5':
                    height = 0.3
                else:
                    height= 0
                ax.text(-0.1,
                        height,
                        mod,
                        fontsize=title_font,
                        fontweight='bold',
                        rotation='vertical',
                        transform=ax.transAxes)
        
        if i < 3:
            
            ax.set_title(column,
                         loc='center',
                         fontsize = title_font,
                         fontweight='bold')
            
        i += 1

# lu response pattern colorbar
cb_lu = mpl.colorbar.ColorbarBase(ax=cbax_lu, 
                                  cmap=cmap_list_lu,
                                  norm=norm_lu,
                                  spacing='uniform',
                                  orientation='horizontal',
                                  extend='both',
                                  ticks=tick_locs_lu,
                                  drawedges=False)
cb_lu.set_label('LU trends (°C/5-years)',
                 size=title_font)
cb_lu.ax.xaxis.set_label_position('top')
cb_lu.ax.tick_params(labelcolor=col_cbticlbl,
                     labelsize=tick_font,
                     color=col_cbtic,
                     length=cb_ticlen,
                     width=cb_ticwid,
                     direction='out'); 
cb_lu.ax.set_xticklabels(tick_labels_lu,
                         rotation=45)
cb_lu.outline.set_edgecolor(col_cbedg)
cb_lu.outline.set_linewidth(cb_edgthic)

# lc pattern colorbar
cb_lc = mpl.colorbar.ColorbarBase(ax=cbax_lc, 
                                  cmap=cmap_list_lc,
                                  norm=norm_lc,
                                  spacing='uniform',
                                  orientation='horizontal',
                                  extend='both',
                                  ticks=tick_locs_lc,
                                  drawedges=False)
cb_lc.set_label('LC trends (% landcover/5-years)',
                 size=title_font)
cb_lc.ax.xaxis.set_label_position('top')
cb_lc.ax.tick_params(labelcolor=col_cbticlbl,
                     labelsize=tick_font,
                     color=col_cbtic,
                     length=cb_ticlen,
                     width=cb_ticwid,
                     direction='out'); 
cb_lc.ax.set_xticklabels(tick_labels_lc,
                         rotation=45)
cb_lc.outline.set_edgecolor(col_cbedg)
cb_lc.outline.set_linewidth(cb_edgthic)

f.savefig(outDIR+'/lu_lc_trends_v2.png')
    
#%%============================================================================

# cmap_list_corr,tick_locs_corr,tick_labels_corr,norm_corr = colormap_details('RdBu_r',
#                                                                             data_lumper(stats_ds,
#                                                                                         models,
#                                                                                         maptype='corr'),
#                                                                             null_bnds_corr)

# fig size
x=18
y=15

# tight layout or gspec boundaries, maybe not necessary
# t_left = 0.0
# t_bottom = 0.0
# t_right = 1
# t_top = 1
# t_rect = [t_left, t_bottom, t_right, t_top]

col_cbticlbl = '0'   # colorbar color of tick labels
col_cbtic = '0.5'   # colorbar color of ticks
col_cbedg = '0.9'   # colorbar color of edge
cb_ticlen = 3.5   # colorbar length of ticks
cb_ticwid = 0.4   # colorbar thickness of ticks
cb_edgthic = 0   # colorbar thickness of edges between colors
cblabel = 'corr'  # colorbar label
sbplt_lw = 0.1   # linewidth on projection panels
cstlin_lw = 0.75   # linewidth for coastlines

# fonts
title_font = 20
cbtitle_font = 20
tick_font = 18
legend_font=12

# placment lu trends cbar
cb_x0 = 0.275
cb_y0 = 0.05
cb_xlen = 0.5
cb_ylen = 0.015

# extent
east = 180
west = -180
north = 80
south = -60
extent = [west,east,south,north]

# identify colors
cmap_whole = plt.cm.get_cmap('RdBu_r')
cmap55 = cmap_whole(0.01)
cmap50 = cmap_whole(0.05)   #blue
cmap45 = cmap_whole(0.1)
cmap40 = cmap_whole(0.15)
cmap35 = cmap_whole(0.2)
cmap30 = cmap_whole(0.25)
cmap25 = cmap_whole(0.3)
cmap20 = cmap_whole(0.325)
cmap10 = cmap_whole(0.4)
cmap5 = cmap_whole(0.475)
cmap0 = 'gray'
cmap_5 = cmap_whole(0.525)
cmap_10 = cmap_whole(0.6)
cmap_20 = cmap_whole(0.625)
cmap_25 = cmap_whole(0.7)
cmap_30 = cmap_whole(0.75)
cmap_35 = cmap_whole(0.8)
cmap_40 = cmap_whole(0.85)
cmap_45 = cmap_whole(0.9)
cmap_50 = cmap_whole(0.95)  #red
cmap_55 = cmap_whole(0.99)

colors = [cmap_45,cmap_35,cmap_25,cmap_10,
            cmap0,
            cmap10,cmap25,cmap35,cmap45]

# declare list of colors for discrete colormap of colorbar
cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

# colorbar args
values = [-1,-.75,-.50,-.25,-0.01,0.01,.25,.5,.75,1]
tick_locs = [-1,-.75,-.50,-.25,0,.25,.5,.75,1]
tick_labels = ['-1','-0.75','-0.50','-0.25','0','0.25','0.50','0.75','1']
norm = mpl.colors.BoundaryNorm(values,cmap_list.N)

f, axes = plt.subplots(nrows=len(models),
                       ncols=2,
                       figsize=(x,y),
                       subplot_kw={'projection':ccrs.PlateCarree()})

cbax = f.add_axes([cb_x0, 
                   cb_y0, 
                   cb_xlen, 
                   cb_ylen])

i = 0

for mod,row_axes in zip(models,axes):
    
    for ax,lc in zip(row_axes,['treeFrac','cropFrac']):
    
        plottable = stats_ds[mod]['lu-{}_corr'.format(lc)].where((stats_ds[mod]['{}_slope'.format(lc)]<null_bnds_lc[0])|\
            (stats_ds[mod]['{}_slope'.format(lc)]>null_bnds_lc[1]))
        plottable.plot(ax=ax,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap_list,
                       cbar_ax=cbax,
                       levels=values,
                       center=0,
                       add_labels=False)
        
            # stats_ds[mod]['lu_slope'].plot(ax=row_axes[0],
            #                        cmap=cmap_list_lu,
            #                        cbar_ax=cbax_lu,
            #                        levels=levels_lu,
            #                        extend='both',
            #                        center=0,
            #                        add_labels=False)
    
    
    for ax,column in zip(row_axes,['LU,treeFrac','LU,cropFrac']):
        
        ax.set_extent(extent,
                      crs=ccrs.PlateCarree())
        ax.set_title(letters[i],
                     loc='left',
                     fontsize=title_font,
                     fontweight='bold')
        ax.coastlines(linewidth=cstlin_lw)
        
        if column == 'LU,treeFrac':
            
                if mod == 'CanESM5':
                    height = 0.3
                else:
                    height= 0
                ax.text(-0.1,
                        height,
                        mod,
                        fontsize=title_font,
                        fontweight='bold',
                        rotation='vertical',
                        transform=ax.transAxes)
        
        if i < 2:
            
            ax.set_title(column,
                         loc='center',
                         fontsize = title_font,
                         fontweight='bold')
            
        i += 1
        
cb = mpl.colorbar.ColorbarBase(ax=cbax, 
                                   cmap=cmap_list,
                                   norm=norm,
                                   spacing='uniform',
                                   orientation='horizontal',
                                   extend='neither',
                                   ticks=tick_locs,
                                   drawedges=False)
cb.set_label('Correlation',
                 size=title_font)
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(labelcolor=col_cbticlbl,
                  labelsize=tick_font,
                  color=col_cbtic,
                  length=cb_ticlen,
                  width=cb_ticwid,
                  direction='out'); 
cb.ax.set_xticklabels(tick_labels,
                      rotation=45)
cb.outline.set_edgecolor(col_cbedg)
cb.outline.set_linewidth(cb_edgthic)

f.savefig(outDIR+'/lu_lc_correlation_v2.png')

# %%
    # test_lc = xr.where((stats_ds[mod]['treeFrac_slope']<null_bnds_lc[0]) | \
    #                 (stats_ds[mod]['treeFrac_slope']>null_bnds_lc[1]),
    #                 1,
    #                 0)
    # stats_ds[mod]['treeFrac_slope'] = stats_ds[mod]['treeFrac_slope'].where(test_lc == 1,
    #                                                                         other = 0)
    # stats_ds[mod]['treeFrac_slope'] = stats_ds[mod]['treeFrac_slope'].where(ar6_land[mod] == 1)
    
    #     stats_ds[mod]['lu_slope'].plot(ax=row_axes[0],
    #                                transform=ccrs.PlateCarree(),
    #                                cmap=cmap_list_lu,
    #                                cbar_ax=cbax_lu,
    #                                levels=levels_lu,
    #                                center=0,
    #                             #    norm=norm_lu,
    #                                add_labels=False)
    
    # stats_ds[mod]['treeFrac_slope'].plot(ax=row_axes[1],
    #                                      transform=ccrs.PlateCarree(),
    #                                      cmap=cmap_list_lc,
    #                                      cbar_ax=cbax_lc,
    #                                      levels=levels_lc,
    #                                      center=0,
    #                                     #  norm=norm_lc,
    #                                      add_labels=False)
    
    # stats_ds[mod]['cropFrac_slope'].plot(ax=row_axes[2],
    #                                      transform=ccrs.PlateCarree(),
    #                                      cmap=cmap_list_lc,
    #                                      cbar_ax=cbax_lc,
    #                                      levels=levels_lc,
    #                                      center=0,
    #                                     #  norm=norm_lc,
    #                                      add_labels=False)