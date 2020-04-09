#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:50:22 2019

@author: hebert
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Theorical curves
def vrain(x):
    return 3.78*(x**0.67)      
def vlumpg(x):
    return 1.3*x**0.66 
def vdry(x):
    return 1.07*x**0.2
def vwet(x):
    return 2.14*x**0.2

# Read the file
F = open("/home/hebert/Bureau/python/SPFP_Parsivel_RAW_190403.txt","r") 

# Extract data from txt file
table = []
time = []
Cumul = []
RR = []
for line in F: 
    ligne = line.split(',')
    if len(ligne) == 1104:
        time = np.append(time, datetime.datetime.strptime(ligne[0][0:19], '%Y/%m/%d %H:%M:%S'))
        RR = np.append(RR, float(ligne[0][21:]))
        Cumul = np.append(Cumul, float(ligne[1]))
        table = np.append(table, np.array(map(int,ligne[79:1103])))

### Time
## Option 1: Set time with the last minute of the file
#end = time[-1]
#start = end + datetime.timedelta(minutes=-1)

## Option 2: Set the time interval you want
start = datetime.datetime.strptime('2019/04/03 14:40:00', '%Y/%m/%d %H:%M:%S')
end = datetime.datetime.strptime('2019/04/03 14:41:00', '%Y/%m/%d %H:%M:%S')

## Option 3: Set time automatically for the last minute from now (or UTC now with datetime.datetime.utcnow()) 
#start = datetime.datetime.now() + datetime.timedelta(minutes=-1)
#end = datetime.datetime.now()

### Tables and matrix
Cumul = Cumul - Cumul[0]*np.ones(len(Cumul))
table0 = pd.DataFrame(table.reshape(len(time),1024), index = time).loc[start:end]
table1 = np.ma.masked_where(np.isnan(table0.replace(0, np.nan)),table0.replace(0, np.nan))
mat = table1.sum(axis=0).reshape(32,32)

D=[0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,1.187,1.375,1.625,1.875,2.125,2.375,2.750,3.25,3.75,4.25,4.75,5.5,6.5,7.5,8.5,9.5,11,13,15,17,19,21.5,24.5]     
V=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.10,1.30,1.5,1.7,1.9,2.2,2.6,3,3.4,3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,17.6,20.8]

### Average speed and size of particles for this interval
Dmoy = np.sum(mat.sum(axis=0)*D)/mat.sum()
Vmoy = np.sum(mat.sum(axis=1)*V)/mat.sum()

### Graphic
dfit = np.arange(0,len(D),0.1)
fig, ax = plt.subplots(1,1, figsize = (12,7))
[x,y]=np.meshgrid(D,V,indexing='xy')
pc = ax.pcolormesh(x,y,mat, alpha=0.5, cmap="winter")
cb = fig.colorbar(pc, fraction=0.046, pad=0.04)
cb.ax.set_ylabel('Number of particles', size = 11)
ax.scatter(x=Dmoy,y=Vmoy, s=200, marker='+', color="red" )
ax.plot(dfit,vrain(dfit),'g',label=u"Rain")
ax.plot(dfit,vlumpg(dfit),'r',label=u"Graupel")
ax.plot(dfit,vdry(dfit),'b',label=u"Dry snow")
ax.plot(dfit,vwet(dfit),'y',label=u"Wet snow")
ax.set_xlim(0,10)
ax.set_ylim(0,10)
for i in V:    plt.axhline(y=i, linewidth=0.2, color = 'k')
for j in D:    plt.axvline(x=j, linewidth=0.2, color='k')
ax.set_xticks(np.arange(0, 16))
ax.set_yticks(np.arange(0, 13))
ax.tick_params(labelsize=11)
ax.tick_params(labelsize=11)
plt.xlabel('Diameter (mm)', size = 12)
plt.ylabel('Fallspeed (m/s)', size = 12)
plt.legend(loc= 'upper right')
fig.suptitle('Parsivel observation: ' + str(start) +' UTC', size=20)