#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:49:56 2022

@author: apple
"""


import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def run(FILE_NAME):
    
    with h5py.File(FILE_NAME, mode='r') as f:

        name = '/ZLE'
        data = f[name][0,0,:,:]
        units = f[name].attrs['units']
        long_name = f[name].attrs['long_name']
        _FillValue = f[name].attrs['_FillValue']
        data[data == _FillValue] = np.nan
        data = np.ma.masked_where(np.isnan(data), data)

        
        # Get the geolocation data.
        latitude = f['/lat'][:]
        longitude = f['/lon'][:]

        # Get the time & lev data.
        lev = f['/lev'][:]
        lev_units = f['/lev'].attrs['units']
        lev_long_name = f['/lev'].attrs['long_name']
        time = f['/time'][:]
        time_units = f['/time'].attrs['units']
        time_long_name = f['/time'].attrs['long_name']        
        
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180)
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(np.arange(-90, 91, 45))
    m.drawmeridians(np.arange(-180, 180, 45), labels=[True,False,False,True])
    m.pcolormesh(longitude, latitude, data, latlon=True)
    cb = m.colorbar()    
    cb.set_label(units)


    basename = os.path.basename(FILE_NAME)
    tstr = time_long_name+' = '+str(time[0])+' '+time_units
    lstr = lev_long_name+' = '+str(lev[0])+' '+lev_units
    plt.title('{0}\n{1}\n{2}\n{3}'.format(basename, long_name, tstr, lstr))
    fig = plt.gcf()
    # plt.show()
    pngfile = "{0}.py.png".format(basename)
    fig.savefig(pngfile)

if __name__ == "__main__":

    # If a certain environment variable is set, look there for the input
    # file, otherwise look in the current directory.
    hdffile = '/users/apple/Desktop/UC_Berkeley/UCB_2022/'\
        'w210/data/3B-DAY.MS.MRG.3IMERG.20200602-S000000-E235959.V06.nc4'

    try:
        hdffile = os.path.join(os.environ['HDFEOS_ZOO_DIR'], hdffile)
    except KeyError:
        pass

    run(hdffile)