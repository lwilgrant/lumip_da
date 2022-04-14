#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

# =============================================================================
# import
# =============================================================================


import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import pandas as pd
import os
import regionmask as rm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import pickle as pk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from copy import deepcopy
import pickle as pk
import geopandas as gp
import mapclassify as mc


# =============================================================================
# functions
# =============================================================================

#%%============================================================================

def classifier(value):
    
    one_pc = 1/100
    five_pc = 5/100
    ten_pc = 10/100
    if value < one_pc:
        value = 1
    elif (value >= one_pc) & (value < five_pc):
        value = 2
    elif (value >= five_pc) & (value < ten_pc):
        value = 3
    elif value >= ten_pc:
        value = 4
    return value

