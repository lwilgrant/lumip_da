#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

#%%=============================================================================
# import
#===============================================================================


import numpy as np
from pooch import test
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import pandas as pd
import os
import regionmask as rm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle as pk
import copy as cp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from random import shuffle
from matplotlib.lines import Line2D
import matplotlib as mpl


#==============================================================================
# functions
#==============================================================================

#%%==============================================================================    

def nc_read(file,
            y1,
            var,
            obs=False):
    
    """ Read in netcdfs based on variable and set time.
    
    Parameters
    ----------
    file : files in data directory
    
    Returns
    ------- 
    Xarray data array
    """
    
    ds = xr.open_dataset(file,decode_times=False)
    da = ds[var].squeeze()
    
    units, reference_date = da.time.attrs['units'].split('since')
    reference_date = reference_date.replace(reference_date[1:5],str(y1))[1:]
    new_date = pd.date_range(start=reference_date, periods=da.sizes['time'], freq='YS')
    da['time'] = new_date
    
    if 'height' in da.coords:
        da = da.drop('height')
        
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
    
    return da

#%%==============================================================================    

def ar6_mask(da,
             obs):
    

    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)
    ar6_regs = ar6_regs.where(landmask == 0)
    
    return ar6_regs

#%%==============================================================================    

def weighted_mean(continents,
                  da,
                  ar6_regs,
                  nt,
                  matrix):
    
    s = 0
    for c in continents.keys():
        for i in continents[c]:
            da_i = da.where(ar6_regs==i,
                            drop=True)                    
            for t in np.arange(nt):
                da_i_t = da_i.isel(time=t)
                weights = np.cos(np.deg2rad(da_i_t.lat))
                weights.name='weights'
                da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                matrix[t,s]= da_i_t_w
            s += 1
            
    return matrix

#%%==============================================================================    

def temp_center(ns,
                mod_ar6):
    
    for s in np.arange(ns):
        arr = mod_ar6[:,s]
        arr_mean = np.mean(arr)
        arr_center = arr - arr_mean
        mod_ar6[:,s] = arr_center
        
    return mod_ar6

#%%==============================================================================    

def ensembler(data_list,
              ax=False):
    if not ax:
        matrix = np.stack(data_list,axis=1)
        nx = np.shape(matrix)[1]
        ens_mean = np.mean(matrix,axis=1)
    
    if ax:
        matrix = np.stack(data_list,axis=ax)
        nx = np.shape(matrix)[ax]
        ens_mean = np.mean(matrix,axis=ax)
    
    return ens_mean,nx

#%%==============================================================================    

def da_ensembler(data):
    
    concat_dim = np.arange(len(data))
    aligned = xr.concat(data,dim=concat_dim)
    mean = aligned.mean(dim='concat_dim')
    
    return mean
    
    
#%%==============================================================================    

def c(x):
   col = plt.cm.BrBG(x)
   fig, ax = plt.subplots(figsize=(1,1))
   fig.set_facecolor(col)
   ax.axis("off")
   plt.show()
   
#%%==============================================================================   


def pca_plot(eof_dict,
             principal_components,
             pseudo_principal_components,
             exps_start,
             obs_types,
             outDIR):

    

    ############################### panels ##################################
    
    x=22
    y=10
    f = plt.figure(figsize=(x,y))
    
    # time series rect, rect=[left, bottom, right, top]
    t_left = 0.05
    t_bottom = 0.65
    t_right = 0.7
    t_top = 1.0
    t_rect = [t_left, t_bottom, t_right, t_top]
    
    # time series
    gs1 = gridspec.GridSpec(1,1)
    ax1 = f.add_subplot(gs1[0])    
       
    gs1.tight_layout(figure=f, rect=t_rect, h_pad=5)
    
    # maps of eof
    gs2 = gridspec.GridSpec(2,3)
        
    # map panel rect, rect=[left, bottom, right, top]
    c_left = 0
    c_bottom = 0.0
    c_right = 1.0
    c_top = 0.6
    c_rect = [c_left, c_bottom, c_right, c_top]
    
    ax2 = f.add_subplot(gs2[0],projection=ccrs.PlateCarree()) # obs eof (cru)
    ax3 = f.add_subplot(gs2[1],projection=ccrs.PlateCarree()) # hist eof
    ax4 = f.add_subplot(gs2[2],projection=ccrs.PlateCarree()) # hist-nolu bias
    
    ax5 = f.add_subplot(gs2[3],projection=ccrs.PlateCarree()) # obs eof (cru)
    ax6 = f.add_subplot(gs2[4],projection=ccrs.PlateCarree()) # delta eof
    ax7 = f.add_subplot(gs2[5],projection=ccrs.PlateCarree()) # eof bias
    
    map_axes = {}
    map_axes['cru'] = [ax2,ax3,ax4]
    map_axes['berkley_earth'] = [ax5,ax6,ax7]
    
    gs2.tight_layout(figure=f, rect=c_rect, h_pad=5)
    
    ############################### general ##################################
    
    letters = ['a','b','c','d','e','f','g','h','i']
    ls_types = ['-', # cru
                '--'] # berkley
    ls_types = {}
    ls_types['cru'] = '-'
    ls_types['berkley_earth'] = '--'
    ls_types['pc'] = ':'
    
    cmap_brbg = plt.cm.get_cmap('BrBG')
    cols={}
    cols['hist-noLu'] = cmap_brbg(0.95)
    cols['historical'] = cmap_brbg(0.05)
    col_zero = 'gray'   # zero change color
    
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    col_zero = 'gray'   # zero change color
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.2   # linewidth for coastlines
    
    title_font = 12
    cbtitle_font = 12
    tick_font = 12
    legend_font=12
    
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]
    
    
    ############################### legend ##################################
    
    # bbox
    le_x0 = 1.05
    le_y0 = 0.75
    le_xlen = 0.15
    le_ylen = 0.25
    
    # space between entries
    legend_entrypad = 0.5
    
    # length per entry
    legend_entrylen = 0.75
    
    # space between entries
    legend_spacing = 1.5
    
    ############################### colormaps ##################################
    
    # identify colors for obs eof maps
    cmap55 = cmap_brbg(0.01)
    cmap50 = cmap_brbg(0.05)   #blue
    cmap45 = cmap_brbg(0.1)
    cmap40 = cmap_brbg(0.15)
    cmap35 = cmap_brbg(0.2)
    cmap30 = cmap_brbg(0.25)
    cmap25 = cmap_brbg(0.3)
    cmap20 = cmap_brbg(0.325)
    cmap10 = cmap_brbg(0.4)
    cmap5 = cmap_brbg(0.475)
    cmap0 = col_zero
    cmap_5 = cmap_brbg(0.525)
    cmap_10 = cmap_brbg(0.6)
    cmap_20 = cmap_brbg(0.625)
    cmap_25 = cmap_brbg(0.7)
    cmap_30 = cmap_brbg(0.75)
    cmap_35 = cmap_brbg(0.8)
    cmap_40 = cmap_brbg(0.85)
    cmap_45 = cmap_brbg(0.9)
    cmap_50 = cmap_brbg(0.95)  #red
    cmap_55 = cmap_brbg(0.99)
    
    colors_brbg = [cmap_55,
                   cmap_45,
                   cmap_35,
                   cmap_25,
                   cmap_10,
                   cmap0,
                   cmap10,
                   cmap25,
                   cmap35,
                   cmap45,
                   cmap55]
    
    # declare list of colors for discrete colormap of colorbar
    cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
    
    # colorbar args
    start = 0.020
    inc = start/5
    values_eof = [-1*start,
                  -1*start+inc,
                  -1*start+inc*2,
                  -1*start+inc*3,
                  -1*start+inc*4,
                  -0.001,
                  0.001,
                  start-inc*4,
                  start-inc*3,
                  start-inc*2,
                  start-inc,
                  start]
    
    tick_locs_eof = [-1*start,
                     -1*start+inc,
                     -1*start+inc*2,
                     -1*start+inc*3,
                     -1*start+inc*4,
                     0,
                     start-inc*4,
                     start-inc*3,
                     start-inc*2,
                     start-inc,
                     start]
    
    tick_labels_eof = [str(-1*start),
                       str(-1*start+inc),
                       str(-1*start+inc*2),
                       str(-1*start+inc*3),
                       str(-1*start+inc*4),
                       str(0),
                       str(start-inc*4),
                       str(start-inc*3),
                       str(start-inc*2),
                       str(start-inc),
                       str(start)]
    
    norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)
    
    cb_eof_x0 = 0.36  
    cb_eof_y0 = -0.05   
    cb_eof_xlen = 0.5
    cb_eof_ylen = 0.015
    
    cbax_eof = f.add_axes([cb_eof_x0, 
                           cb_eof_y0, 
                           cb_eof_xlen, 
                           cb_eof_ylen])
    
    
    
    ############################### timeseries ##################################
    
    
    # plot time series
    for exp in exps_start:
        for obs in obs_types:
            data = pseudo_principal_components[obs][exp] - pseudo_principal_components[obs][exp].mean(dim='time')
            data.plot(ax=ax1,
                      add_legend=False,
                      color=cols[exp],
                      linestyle=ls_types[obs],
                      linewidth=3,
                      label=obs+' pseudo-PC ' +exp)
            
    ax1.legend(frameon=False,
               bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
               fontsize=legend_font,
               labelspacing=legend_spacing)
    ax1.set_title(letters[0],
                  fontweight='bold',
                  loc='left')
    
    ############################### eof maps ##################################
    
    # eof loading maps
    count = 0
    for obs in obs_types:
        # data = figure_data[obs]
        for i,ax in enumerate(map_axes[obs]):
            count += 1
            if i == 0:
                eof_dict[obs][obs].plot(ax=ax,
                             transform=ccrs.PlateCarree(),
                             cmap=cmap_list_eof,
                             cbar_ax=cbax_eof,
                             center=0,
                             norm=norm_eof,
                             add_labels=False)
                if obs == 'cru':
                    height = 0.5
                elif obs == 'berkley_earth':
                    height = 0.3
                ax.text(-0.07,
                        height,
                        obs,
                        fontsize=title_font,
                        fontweight='bold',
                        rotation='vertical',
                        transform=ax.transAxes)
            elif i > 0:
                for exp in exps_start:
                    eof_dict[obs][exp].plot(ax=ax,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap_list_eof,
                                    cbar_ax=cbax_eof,
                                    center=0,
                                    norm=norm_eof,
                                    add_labels=False)      
            if obs == 'cru':
                if i == 0:
                    ax.set_title('Obs EOF loading',
                                 fontweight='bold',
                                 loc='center',
                                 fontsize=title_font)
                if i == 1:
                    ax.set_title('hist EOF loading',
                                 fontweight='bold',
                                 loc='center',
                                 fontsize=title_font)
                if i == 2:
                    ax.set_title('hist-noLu EOF loading',
                                 fontweight='bold',
                                 loc='center',
                                 fontsize=title_font)
            ax.set_title(letters[count],
                         fontweight='bold',
                         loc='left')
            ax.set_extent(extent,
                          crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=cstlin_lw)
    
    # eof cb
    cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                       cmap=cmap_list_eof,
                                       norm=norm_eof,
                                       spacing='uniform',
                                       orientation='horizontal',
                                       extend='neither',
                                       ticks=tick_locs_eof,
                                       drawedges=False)
    cb_eof.ax.xaxis.set_label_position('top')
    cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                          labelsize=tick_font,
                          color=col_cbtic,
                          length=cb_ticlen,
                          width=cb_ticwid,
                          direction='out'); 
    cb_eof.ax.set_xticklabels(tick_labels_eof,
                              rotation=45)
    cb_eof.outline.set_edgecolor(col_cbedg)
    cb_eof.outline.set_linewidth(cb_edgthic)
    
    os.chdir(outDIR)
# =============================================================================
#     f.savefig(outDIR+'/pca_analysis.png',bbox_inches='tight',dpi=400)
# =============================================================================

#%%==============================================================================   


def pca_plot_continental(eof_dict,
                         principal_components,
                         pseudo_principal_components,
                         exps_start,
                         continents,
                         obs_types,
                         outDIR):

    
    ############################### general ##################################
    
    letters = ['a','b','c','d','e','f','g','h','i']
    ls_types = ['-', # cru
                '--'] # berkley
    ls_types = {}
    ls_types['cru'] = '-'
    ls_types['berkley_earth'] = '--'
    ls_types['pc'] = ':'
    
    cmap_brbg = plt.cm.get_cmap('BrBG')
    cols={}
    cols['hist-noLu'] = cmap_brbg(0.95)
    cols['historical'] = cmap_brbg(0.05)
    col_zero = 'gray'   # zero change color
    
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    col_zero = 'gray'   # zero change color
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.2   # linewidth for coastlines
    
    title_font = 12
    cbtitle_font = 12
    tick_font = 12
    legend_font=12
    
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]
    
    
    ############################### legend ##################################
    
    # bbox
    le_x0 = 1.05
    le_y0 = 0.75
    le_xlen = 0.15
    le_ylen = 0.25
    
    # space between entries
    legend_entrypad = 0.5
    
    # length per entry
    legend_entrylen = 0.75
    
    # space between entries
    legend_spacing = 1.5
    
    ############################### colormaps ##################################
    
    # identify colors for obs eof maps
    cmap55 = cmap_brbg(0.01)
    cmap50 = cmap_brbg(0.05)   #blue
    cmap45 = cmap_brbg(0.1)
    cmap40 = cmap_brbg(0.15)
    cmap35 = cmap_brbg(0.2)
    cmap30 = cmap_brbg(0.25)
    cmap25 = cmap_brbg(0.3)
    cmap20 = cmap_brbg(0.325)
    cmap10 = cmap_brbg(0.4)
    cmap5 = cmap_brbg(0.475)
    cmap0 = col_zero
    cmap_5 = cmap_brbg(0.525)
    cmap_10 = cmap_brbg(0.6)
    cmap_20 = cmap_brbg(0.625)
    cmap_25 = cmap_brbg(0.7)
    cmap_30 = cmap_brbg(0.75)
    cmap_35 = cmap_brbg(0.8)
    cmap_40 = cmap_brbg(0.85)
    cmap_45 = cmap_brbg(0.9)
    cmap_50 = cmap_brbg(0.95)  #red
    cmap_55 = cmap_brbg(0.99)
    
    colors_brbg = [cmap_55,
                   cmap_45,
                   cmap_35,
                   cmap_25,
                   cmap_10,
                   cmap0,
                   cmap10,
                   cmap25,
                   cmap35,
                   cmap45,
                   cmap55]
    
    # declare list of colors for discrete colormap of colorbar
    cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
    
    # colorbar args
    start = 0.020
    inc = start/5
    values_eof = [-1*start,
                  -1*start+inc,
                  -1*start+inc*2,
                  -1*start+inc*3,
                  -1*start+inc*4,
                  -0.001,
                  0.001,
                  start-inc*4,
                  start-inc*3,
                  start-inc*2,
                  start-inc,
                  start]
    
    tick_locs_eof = [-1*start,
                     -1*start+inc,
                     -1*start+inc*2,
                     -1*start+inc*3,
                     -1*start+inc*4,
                     0,
                     start-inc*4,
                     start-inc*3,
                     start-inc*2,
                     start-inc,
                     start]
    
    tick_labels_eof = [str(-1*start),
                       str(-1*start+inc),
                       str(-1*start+inc*2),
                       str(-1*start+inc*3),
                       str(-1*start+inc*4),
                       str(0),
                       str(start-inc*4),
                       str(start-inc*3),
                       str(start-inc*2),
                       str(start-inc),
                       str(start)]
    
    norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)
    
    cb_eof_x0 = 0.36  
    cb_eof_y0 = -0.05   
    cb_eof_xlen = 0.5
    cb_eof_ylen = 0.015
    
    for c in continents.keys():
    
        ############################### panels ##################################
        
        x=22
        y=10
        f = plt.figure(figsize=(x,y))
        
        # time series rect, rect=[left, bottom, right, top]
        t_left = 0.05
        t_bottom = 0.65
        t_right = 0.7
        t_top = 1.0
        t_rect = [t_left, t_bottom, t_right, t_top]
        
        # time series
        gs1 = gridspec.GridSpec(1,1)
        ax1 = f.add_subplot(gs1[0])    
        
        gs1.tight_layout(figure=f, rect=t_rect, h_pad=5)
        
        # maps of eof
        gs2 = gridspec.GridSpec(2,3)
            
        # map panel rect, rect=[left, bottom, right, top]
        c_left = 0
        c_bottom = 0.0
        c_right = 1.0
        c_top = 0.6
        c_rect = [c_left, c_bottom, c_right, c_top]
        
        ax2 = f.add_subplot(gs2[0],projection=ccrs.PlateCarree()) # obs eof (cru)
        ax3 = f.add_subplot(gs2[1],projection=ccrs.PlateCarree()) # hist eof
        ax4 = f.add_subplot(gs2[2],projection=ccrs.PlateCarree()) # hist-nolu bias
        
        ax5 = f.add_subplot(gs2[3],projection=ccrs.PlateCarree()) # obs eof (cru)
        ax6 = f.add_subplot(gs2[4],projection=ccrs.PlateCarree()) # delta eof
        ax7 = f.add_subplot(gs2[5],projection=ccrs.PlateCarree()) # eof bias
        
        map_axes = {}
        map_axes['cru'] = [ax2,ax3,ax4]
        map_axes['berkley_earth'] = [ax5,ax6,ax7]
        
        gs2.tight_layout(figure=f, rect=c_rect, h_pad=5)
        
        cbax_eof = f.add_axes([cb_eof_x0, 
                            cb_eof_y0, 
                            cb_eof_xlen, 
                            cb_eof_ylen])
        
        ############################### timeseries ##################################
        
        # plot time series
        for exp in exps_start:
            for obs in obs_types:
                data = pseudo_principal_components[obs][exp][c] - pseudo_principal_components[obs][exp][c].mean(dim='time')
                data.plot(ax=ax1,
                          add_legend=False,
                          color=cols[exp],
                          linestyle=ls_types[obs],
                          linewidth=3,
                          label=obs+' pseudo-PC ' +exp)
                
        ax1.legend(frameon=False,
                   bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                   fontsize=legend_font,
                   labelspacing=legend_spacing)
        ax1.set_title(letters[0],
                      fontweight='bold',
                      loc='left')
        
        ############################### eof maps ##################################
        
        # eof loading maps
        count = 0
        for obs in obs_types:
            # data = figure_data[obs]
            for i,ax in enumerate(map_axes[obs]):
                count += 1
                if i == 0:
                    eof_dict[obs][obs][c].plot(ax=ax,
                                               transform=ccrs.PlateCarree(),
                                               cmap=cmap_list_eof,
                                               cbar_ax=cbax_eof,
                                               center=0,
                                               norm=norm_eof,
                                               add_labels=False)
                    if obs == 'cru':
                        height = 0.5
                    elif obs == 'berkley_earth':
                        height = 0.3
                    ax.text(-0.07,
                            height,
                            obs,
                            fontsize=title_font,
                            fontweight='bold',
                            rotation='vertical',
                            transform=ax.transAxes)
                elif i > 0:
                    for exp in exps_start:
                        eof_dict[obs][exp][c].plot(ax=ax,
                                                   transform=ccrs.PlateCarree(),
                                                   cmap=cmap_list_eof,
                                                   cbar_ax=cbax_eof,
                                                   center=0,
                                                   norm=norm_eof,
                                                   add_labels=False)      
                if obs == 'cru':
                    if i == 0:
                        ax.set_title('Obs EOF loading',
                                     fontweight='bold',
                                     loc='center',
                                     fontsize=title_font)
                    if i == 1:
                        ax.set_title('hist EOF loading',
                                    fontweight='bold',
                                    loc='center',
                                    fontsize=title_font)
                    if i == 2:
                        ax.set_title('hist-noLu EOF loading',
                                    fontweight='bold',
                                    loc='center',
                                    fontsize=title_font)
                ax.set_title(letters[count],
                            fontweight='bold',
                            loc='left')
                ax.set_extent(extent,
                            crs=ccrs.PlateCarree())
                ax.coastlines(linewidth=cstlin_lw)
        
        # eof cb
        cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                        cmap=cmap_list_eof,
                                        norm=norm_eof,
                                        spacing='uniform',
                                        orientation='horizontal',
                                        extend='neither',
                                        ticks=tick_locs_eof,
                                        drawedges=False)
        cb_eof.ax.xaxis.set_label_position('top')
        cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                            labelsize=tick_font,
                            color=col_cbtic,
                            length=cb_ticlen,
                            width=cb_ticwid,
                            direction='out'); 
        cb_eof.ax.set_xticklabels(tick_labels_eof,
                                rotation=45)
        cb_eof.outline.set_edgecolor(col_cbedg)
        cb_eof.outline.set_linewidth(cb_edgthic)
        
        os.chdir(outDIR)
    # =============================================================================
    #     f.savefig(outDIR+'/pca_analysis_continental_{}.png'.format(c),bbox_inches='tight',dpi=400)
    # =============================================================================

#%%==============================================================================   

def pca_plot_luh2(eof_dict,
                  principal_components,
                  pseudo_principal_components,
                  models,
                  landcover_types,
                  obs_types,
                  outDIR):
    # 3 map panels on right side:
        # lu EOF
        # luh2 deforestation EOF
        # luh2 crops EOF

    # time series on right side:
        # lu psuedo pc on luh2 eofs (do pseudo pc's per model and on multi-model mean)
            # mmm could be band via leave 1 out cross val
        # 
    
    ############################### general ##################################
    
    letters = ['a','b','c','d','e','f','g','h','i']

    ls_types = {}
    ls_types['mmm'] = '-'
    ls_types['model_i'] = ':'
    
    cmap_brbg = plt.cm.get_cmap('BrBG')
    cols={}
    cols['forest'] = cmap_brbg(0.95)
    cols['crops'] = cmap_brbg(0.05)
    model_colors = ['maroon','darkorange','slategrey','slateblue']
    for i,mod in enumerate(models):
        cols[mod] = model_colors[i]

    col_zero = 'gray'   # zero change color
    
    col_cbticlbl = '0'   # colorbar color of tick labels
    col_cbtic = '0.5'   # colorbar color of ticks
    col_cbedg = '0.9'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors
    cblabel = 'corr'  # colorbar label
    col_zero = 'gray'   # zero change color
    sbplt_lw = 0.1   # linewidth on projection panels
    cstlin_lw = 0.2   # linewidth for coastlines
    
    title_font = 12
    cbtitle_font = 12
    tick_font = 12
    legend_font=12
    
    east = 180
    west = -180
    north = 80
    south = -60
    extent = [west,east,south,north]
    
    
    ############################### legend ##################################
    
    # bbox
    le_x0 = 1.0
    le_y0 = 0.75
    le_xlen = 0.15
    le_ylen = 0.25
    
    # space between entries
    legend_entrypad = 0.5
    
    # length per entry
    legend_entrylen = 0.75
    
    # space between entries
    legend_spacing = 1.5
    
    ############################### colormaps ##################################
    
    # identify colors for obs eof maps
    cmap55 = cmap_brbg(0.01)
    cmap50 = cmap_brbg(0.05)   #blue
    cmap45 = cmap_brbg(0.1)
    cmap40 = cmap_brbg(0.15)
    cmap35 = cmap_brbg(0.2)
    cmap30 = cmap_brbg(0.25)
    cmap25 = cmap_brbg(0.3)
    cmap20 = cmap_brbg(0.325)
    cmap10 = cmap_brbg(0.4)
    cmap5 = cmap_brbg(0.475)
    cmap0 = col_zero
    cmap_5 = cmap_brbg(0.525)
    cmap_10 = cmap_brbg(0.6)
    cmap_20 = cmap_brbg(0.625)
    cmap_25 = cmap_brbg(0.7)
    cmap_30 = cmap_brbg(0.75)
    cmap_35 = cmap_brbg(0.8)
    cmap_40 = cmap_brbg(0.85)
    cmap_45 = cmap_brbg(0.9)
    cmap_50 = cmap_brbg(0.95)  #red
    cmap_55 = cmap_brbg(0.99)
    
    colors_brbg = [cmap_45,
                   cmap_35,
                   cmap_25,
                   cmap_10,
                   cmap0,
                   cmap10,
                   cmap25,
                   cmap35,
                   cmap45]
    
    # declare list of colors for discrete colormap of colorbar
    cmap_list_eof = mpl.colors.ListedColormap(colors_brbg,N=len(colors_brbg))
    
    # colorbar args
    start = 0.015
    inc = start/4
    values_eof = [-1*start,
                  -1*start+inc,
                  -1*start+inc*2,
                  -1*start+inc*3,
                  -0.001,
                  0.001,
                  start-inc*3,
                  start-inc*2,
                  start-inc,
                  start]
    
    tick_locs_eof = [-1*start,
                     -1*start+inc,
                     -1*start+inc*2,
                     -1*start+inc*3,
                     0,
                     start-inc*3,
                     start-inc*2,
                     start-inc,
                     start]
    
    tick_labels_eof = [str(-1*start),
                       str(-1*start+inc),
                       str(-1*start+inc*2),
                       str(-1*start+inc*3),
                       str(0),
                       str(start-inc*3),
                       str(start-inc*2),
                       str(start-inc),
                       str(start)]
    
    norm_eof = mpl.colors.BoundaryNorm(values_eof,cmap_list_eof.N)
    
    cb_eof_x0 = 0.66  
    cb_eof_y0 = -0.05   
    cb_eof_xlen = 0.25
    cb_eof_ylen = 0.015
    
    ############################### timeseries ##################################

    # one fig per obs type:
    for obs in obs_types:

        x=22
        y=10
        f = plt.figure(figsize=(x,y))
        
        # time series rect, rect=[left, bottom, right, top]
        t_left = 0.0
        t_bottom = 0.0
        t_right = 0.45
        t_top = 1.0
        t_rect = [t_left, t_bottom, t_right, t_top]
        
        # time series
        gs1 = gridspec.GridSpec(2,1)
        ax1 = f.add_subplot(gs1[0])    
        ax2 = f.add_subplot(gs1[1])    
        
        gs1.tight_layout(figure=f, rect=t_rect, h_pad=5)

        ts_axes = {}
        ts_axes['forest'] = ax1
        ts_axes['crops'] = ax2
        
        # maps of eof
        gs2 = gridspec.GridSpec(2,1)
        ax3 = f.add_subplot(gs2[0],projection=ccrs.PlateCarree()) # luh2 defor eof
        ax4 = f.add_subplot(gs2[1],projection=ccrs.PlateCarree()) # luh2 crops eof
            
        c_left = 0.6
        c_bottom = 0.0
        c_right = 1.0
        c_top = 1.0
        c_rect = [c_left, c_bottom, c_right, c_top]
        
        map_axes = {}
        map_axes['forest'] = ax3
        map_axes['crops'] = ax4
        
        gs2.tight_layout(figure=f, rect=c_rect, h_pad=5)

        cbax_eof = f.add_axes([cb_eof_x0, 
                               cb_eof_y0, 
                               cb_eof_xlen, 
                               cb_eof_ylen])

        # plot time series
        for lc in landcover_types:
            ax = ts_axes[lc]
            # data = principal_components[obs][lc] - principal_components[obs][lc].mean(dim='time')
            # ax = ts_axes[lc]
            # data.plot(ax=ax,
            #           add_legend=False,
            #           color='k',
            #           linestyle=ls_types['mmm'],
            #           linewidth=5,
            #           label='PC1')
            # plot mmm
            data = pseudo_principal_components[obs][lc]['mmm'] - pseudo_principal_components[obs][lc]['mmm'].mean(dim='time')
            data.plot(ax=ax,
                      add_legend=False,
                      color=cols[lc],
                      linestyle=ls_types['mmm'],
                      linewidth=3,
                      label='MMM pseudo-PC ({})'.format(lc))
            # plot alternative mmm
            for mod in models:
                data = pseudo_principal_components[obs][lc]['mmm_no_{}'.format(mod)] - pseudo_principal_components[obs][lc]['mmm_no_{}'.format(mod)].mean(dim='time')
                data.plot(ax=ax,
                          add_legend=False,
                          color=cols[mod],
                          linestyle=ls_types['mmm'],
                          linewidth=3,
                          label='MMM except {}'.format(mod))
                # data = pseudo_principal_components[obs][lc][mod] - pseudo_principal_components[obs][lc][mod].mean(dim='time')
                # data.plot(ax=ts_axes[lc],
                #           add_legend=False,
                #           color=cols[mod],
                #           linestyle=ls_types['model_i'],
                #           linewidth=3,
                #           label=mod)                      
            
                
        ax1.legend(frameon=False,
                bbox_to_anchor=(le_x0, le_y0, le_xlen, le_ylen),
                fontsize=legend_font,
                labelspacing=legend_spacing)
        ax1.set_title(letters[0],
                    fontweight='bold',
                    loc='left')
        
        ############################### eof maps ##################################
        
        # eof loading maps
        for i,lc in enumerate(landcover_types):
            data = eof_dict[obs][lc]
            ax = map_axes[lc]
            data.plot(ax=ax,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap_list_eof,
                        cbar_ax=cbax_eof,
                        center=0,
                        norm=norm_eof,
                        add_labels=False)
            ax.set_title(lc,
                        fontweight='bold',
                        loc='center',
                        fontsize=title_font)
            ax.set_title(letters[i+1],
                        fontweight='bold',
                        loc='left')
            ax.set_extent(extent,
                            crs=ccrs.PlateCarree())
            ax.coastlines(linewidth=cstlin_lw)
        
        # eof cb
        cb_eof = mpl.colorbar.ColorbarBase(ax=cbax_eof, 
                                            cmap=cmap_list_eof,
                                            norm=norm_eof,
                                            spacing='uniform',
                                            orientation='horizontal',
                                            extend='neither',
                                            ticks=tick_locs_eof,
                                            drawedges=False)
        cb_eof.ax.xaxis.set_label_position('top')
        cb_eof.ax.tick_params(labelcolor=col_cbticlbl,
                            labelsize=tick_font,
                            color=col_cbtic,
                            length=cb_ticlen,
                            width=cb_ticwid,
                            direction='out'); 
        cb_eof.ax.set_xticklabels(tick_labels_eof,
                                rotation=45)
        cb_eof.outline.set_edgecolor(col_cbedg)
        cb_eof.outline.set_linewidth(cb_edgthic)
        
        os.chdir(outDIR)
        plt.show()

        # f.savefig(outDIR+'/pca_analysis_luh2_{}.png'.format(obs),bbox_inches='tight',dpi=400)

#%%==============================================================================   

def standard_data(da):
    
    climatology_mean = da.mean("time")
    climatology_std = da.std("time")
    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        da,
        climatology_mean,
        climatology_std,
    )
    return stand_anomalies
        
# #data ensembler
# def ensembler(data,roller_dim):
#     concat_dim = np.arange(len(data))
#     aligned = xr.concat(data,dim=concat_dim)    
#     mean = aligned.mean(dim='concat_dim').rolling(dim={roller_dim:21},center=True).mean()
#     scen_max = aligned.max(dim='concat_dim').rolling(dim={roller_dim:21},center=True).mean()
#     scen_min = aligned.min(dim='concat_dim').rolling(dim={roller_dim:21},center=True).mean()
#     return [mean,scen_max,scen_min]

#%%==============================================================================           

def lengther(mask):
    mask = mask.values.flatten()
    mask_length = np.count_nonzero(mask)
    return(mask_length)