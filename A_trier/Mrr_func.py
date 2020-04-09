import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, date2num
import matplotlib
from datetime import datetime, timedelta
# from rpn_functions import hrstr
import xarray as xray
import sys, os
def hrstr(h):
    if h>=10:
        return str(h)
    else:
        return '0'+str(h)
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def merge_nc_files(filemerge, list_dates, output_name):
    nco = Nco()
    list_nc = []

    for i in range(len(heures)):
        list_nc.append(filemerge+date+'_'+heures[i]+'0000.nc')

    nco.ncrcat(input = list_nc , output = output_name +'.nc')
    

def classic_time_serie(start, end, file):
    #=============================================================================
    #... create the figure
    #....................................
    fig=plt.figure(figsize=(10,10))
    #... reflectivity 
    ax1=fig.add_axes([0.13,0.75,0.85,0.2])
    ax1.set_title('(a) Reflectivity', fontsize = 10) 
    #... Doppler velocity
    ax2=fig.add_axes([0.13,0.50,0.85,0.2])
    ax2.set_title('(b) Doppler velocity', fontsize = 10) 
    #... Contenu en eau liquide
    ax3=fig.add_axes([0.13,0.25,0.85,0.2])
    ax3.set_title('(c) Liquid water content', fontsize = 10) 

    #=============================================================================
    #=============================================================================
    #...Read the MRR data
    #... read the netcdf file after QC

#     start = '2019-02-24 15:40:00'
#     end = '2019-02-24 16:20:00'
    
    current  = datetime(year=start.year, month=start.month, day=start.day, hour=start.hour)
    end_hour = datetime(year=end.year, month=end.month, day=end.day, hour=end.hour)
    filename_list = []
    while current<=end_hour:
        filename_list.append(file+str(current.year)\
                             +hrstr(current.month)+hrstr(current.day)+'_'+hrstr(current.hour)+'0000.nc')
        current += timedelta(hours=1)
    
    nc = []
    for i in range(len(filename_list)):
        nc.append(xray.open_dataset(filename_list[i]))

    nc = xray.concat(nc, dim='time')


    bool_intervalle =  (nc.time.data >= np.datetime64(start)) & (nc.time.data<=np.datetime64(end)) 
    
    #... To plot the MRR timeseries
    height       = nc.variables['range'][:]
    #height_2D = np.rot90(height)

    readtime     = nc.variables['time'][bool_intervalle] 
    x_unstag = np.tile(readtime, (128, 1))

    #... colors number for the colormap
    nb_couleur = 256

    #=============================================================================
    #... Reflectivity
    reflectivity = nc.variables['Ze'][bool_intervalle]
    ones = np.ones(reflectivity.shape)
    reflectivity_2D = np.rot90(reflectivity)

    for i in range(x_unstag.shape[0]):
        t = x_unstag[i,:]
        t = pd.to_datetime(t,unit='s')
        t = np.array(t)
        if i == 0:
            temps = t.reshape(1,t.shape[0])
        else:
            temps = np.append(temps,t.reshape(1,t.shape[0]),axis=0)

    # cree une matrice de niveau de hauteur de la meme dimension que temps
    height_2D = np.zeros((len(height),len(temps[1])))
    for i in range(len(height)):
        height_2D[i,:]=height[len(height)-i-1]

    # tracer la reflectivite en fonction du temps et du range
    plotCF1 = ax1.contourf(temps,height_2D,reflectivity_2D)
    level = np.linspace(-15, 46, nb_couleur)
    label_colormap = 'Ze [dBz]'
    cbZe=plt.colorbar(plotCF1,ax=ax1)
    cbZe.set_label(label_colormap,size=10)
    cbZe.set_ticks([-15, 0, 15, 30, 45])
    cbZe.ax.tick_params(labelsize=10)

    ax1.set_ylim(0,4000)
    ax1.set_ylabel("Height \n AGL [m]",size=10)
    ax1.tick_params(axis='both',labelsize=10)
    ax1.set_xticklabels([])
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) #to plot with the date with day/time
    #ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))


    ##... Doppler velocity
    Dopplervel = nc.variables['VEL'][bool_intervalle]
    Dopplervel_2D = np.rot90(Dopplervel)

    for i in range(x_unstag.shape[0]):
        t = x_unstag[i,:]
        t_mrr = pd.to_datetime(t,unit='s')
        t = np.array(t_mrr)
        if i == 0:
            temps = t.reshape(1,t.shape[0])
        else:
            temps = np.append(temps,t.reshape(1,t.shape[0]),axis=0)

    str_time = str(t_mrr[0].year)+'-'+str(t_mrr[0].month)+'-'+str(t_mrr[0].day)
    str_time_1 = str_time
    t1 = pd.Timestamp(str_time)
    str_time = str(t_mrr[-1].year)+'-'+str(t_mrr[-1].month)+'-'+str(t_mrr[-1].day)
    t2 = pd.Timestamp(str_time)

    level = np.linspace(0, 10, nb_couleur)
    cmap = plt.get_cmap('brg')    
    plotCF2 = ax2.contourf(temps,height_2D,Dopplervel_2D,level,cmap=cmap,extend="both")

    #tracer la vitesse doppler en fonction de temps et du range
    label_colormap = 'W [m/s]'
    cbZe=plt.colorbar(plotCF2,ax=ax2)
    cbZe.set_label(label_colormap,size=10)
    cbZe.set_ticks([ 0, 1, 2, 3,4,5,6,7,8,9,10])
    cbZe.ax.tick_params(labelsize=10)

    ax2.set_ylim(0,4000)
    ax2.set_ylabel("Height \n AGL [m]",size=10)
    ax2.tick_params(axis='both',labelsize=10)
    ax2.set_xticklabels([])
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) #to plot with the date with day/time
    #ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    ##... Taille des hydrometeores
    LWC = nc.variables['LWC'][bool_intervalle]
    LWC_2D = np.rot90(LWC)

    level = np.linspace(0, 2.5, nb_couleur)
    cmap = plt.get_cmap('bwr')    
    plotCF3 = ax3.contourf(temps,height_2D,LWC_2D,level,cmap=cmap,extend="both")

    # tracer le LWC en fonction du temps et du range
    label_colormap = 'g/m3'
    cbZe=plt.colorbar(plotCF3,ax=ax3)
    cbZe.set_label(label_colormap,size=10)
    cbZe.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])
    cbZe.ax.tick_params(labelsize=10)

    ax3.set_ylim(0,4000)
    ax3.set_ylabel("Height \n AGL [m]",size=10)
    ax3.tick_params(axis='both',labelsize=10)
    ax3.set_xticklabels([])
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) #to plot with the date with day/time
    #ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    # No lines
    cnt = [plotCF1,plotCF2,plotCF3]
    for iii in range(len(cnt)):
        for c in cnt[iii].collections:
            c.set_edgecolor("face")
    #plt.savefig('gresil24fev2019.pdf', format='pdf')
    
def Doppler_Reflectivity_time_serie(start, end, file, MRR_pro=True):
    #=============================================================================
    #... create the figure
    #....................................
    fig=plt.figure(figsize=(10,10))
    #... reflectivity 
    ax1=fig.add_axes([0.13,0.75,0.85,0.2])
    ax1.set_title('(a) Reflectivity', fontsize = 10) 
    #... Doppler velocity
    ax2=fig.add_axes([0.13,0.50,0.85,0.2])
    ax2.set_title('(b) Doppler velocity', fontsize = 10) 

    #=============================================================================
    #=============================================================================
    #...Read the MRR data
    #... read the netcdf file after QC

#     start = '2019-02-24 15:40:00'
#     end = '2019-02-24 16:20:00'
    blockPrint()
    if MRR_pro == True:
        current  = datetime(year=start.year, month=start.month, day=start.day, hour=start.hour)
        end_hour = datetime(year=end.year, month=end.month, day=end.day, hour=end.hour)
        filename_list = []
        while current<=end_hour:
            filename_list.append(file+str(current.year)\
                                 +hrstr(current.month)+hrstr(current.day)+'_'+hrstr(current.hour)+'0000.nc')
            current += timedelta(hours=1)

        nc = []
        for i in range(len(filename_list)):
            nc.append(xray.open_dataset(filename_list[i]))

        nc = xray.concat(nc, dim='time')
    else:
        filename_list = []
        current  = datetime(year=start.year, month=start.month, day=start.day, hour=start.hour)
        end_hour = datetime(year=end.year, month=end.month, day=end.day, hour=end.hour)
        filename_list = []
        while current<=end_hour:
#             filename_list.append(file+str(current.year)+hrstr(current.month)+'/'
#             +hrstr(current.month)+hrstr(current.day)+'_ipt.nc')
            filename_list.append(file+str(current.year)+hrstr(current.month)+'/UQAM_MRR2_'+\
            str(current.year)+hrstr(current.month)+hrstr(current.day)+'.nc')


            current += timedelta(days=1)
        nc = []
        for i in range(len(filename_list)):
            nc.append(xray.open_dataset(filename_list[i]))

        
        nc = xray.concat(nc, dim='time')
    enablePrint()
    bool_intervalle =  (nc.time.data >= np.datetime64(start)) & (nc.time.data<=np.datetime64(end)) 
    
    #... To plot the MRR timeseries
    if MRR_pro == True:
        height       = nc.variables['range'][:]
    else:
        height    =  nc.height[0]
    #height_2D = np.rot90(height)

    readtime     = nc.variables['time'][bool_intervalle] 
    x_unstag = np.tile(readtime, (len(height), 1))

    #... colors number for the colormap
    nb_couleur = 256

    #=============================================================================
    #... Reflectivity
    reflectivity = nc.variables['Ze'][bool_intervalle]
    ones = np.ones(reflectivity.shape)
    reflectivity_2D = np.rot90(reflectivity)

    for i in range(x_unstag.shape[0]):
        t = x_unstag[i,:]
        t = pd.to_datetime(t,unit='s')
        t = np.array(t)
        if i == 0:
            temps = t.reshape(1,t.shape[0])
        else:
            temps = np.append(temps,t.reshape(1,t.shape[0]),axis=0)

    # cree une matrice de niveau de hauteur de la meme dimension que temps
    height_2D = np.zeros((len(height),len(temps[1])))
    for i in range(len(height)):
        height_2D[i,:]=height[len(height)-i-1]

    # tracer la reflectivite en fonction du temps et du range
    level = np.linspace(-15, 46, nb_couleur)
    plotCF1 = ax1.contourf(temps,height_2D,reflectivity_2D, level)
    label_colormap = 'Ze [dBz]'
    cbZe=plt.colorbar(plotCF1,ax=ax1)
    cbZe.set_label(label_colormap,size=10)
    cbZe.set_ticks([-15, 0, 15, 30, 45])
    cbZe.ax.tick_params(labelsize=10)

    ax1.set_ylim(0,np.max(height))
    ax1.set_ylabel("Height \n AGL [m]",size=10)
    ax1.tick_params(axis='both',labelsize=10)
    ax1.set_xticklabels([])
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) #to plot with the date with day/time
    #ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))


    ##... Doppler velocity
    
    if MRR_pro == True:
        str_vel = 'VEL'
    else:
        str_vel = 'W'
        
    Dopplervel = nc.variables[str_vel][bool_intervalle]
    Dopplervel_2D = np.rot90(Dopplervel)

    for i in range(x_unstag.shape[0]):
        t = x_unstag[i,:]
        t_mrr = pd.to_datetime(t,unit='s')
        t = np.array(t_mrr)
        if i == 0:
            temps = t.reshape(1,t.shape[0])
        else:
            temps = np.append(temps,t.reshape(1,t.shape[0]),axis=0)

    str_time = str(t_mrr[0].year)+'-'+str(t_mrr[0].month)+'-'+str(t_mrr[0].day)
    str_time_1 = str_time
    t1 = pd.Timestamp(str_time)
    str_time = str(t_mrr[-1].year)+'-'+str(t_mrr[-1].month)+'-'+str(t_mrr[-1].day)
    t2 = pd.Timestamp(str_time)

    level = np.linspace(0, 10, nb_couleur)
    cmap = plt.get_cmap('brg')    
    plotCF2 = ax2.contourf(temps,height_2D,Dopplervel_2D,level,cmap=cmap,extend="both")

    #tracer la vitesse doppler en fonction de temps et du range
    label_colormap = 'W [m/s]'
    cbZe=plt.colorbar(plotCF2,ax=ax2)
    cbZe.set_label(label_colormap,size=10)
    cbZe.set_ticks([ 0, 1, 2, 3,4,5,6,7,8,9,10])
    cbZe.ax.tick_params(labelsize=10)

    ax2.set_ylim(0,np.max(height))
    ax2.set_ylabel("Height \n AGL [m]",size=10)
    ax2.tick_params(axis='both',labelsize=10)
    ax2.set_xticklabels([])
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M')) #to plot with the date with day/time
    #ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    # No lines
    cnt = [plotCF1,plotCF2]
    for iii in range(len(cnt)):
        for c in cnt[iii].collections:
            c.set_edgecolor("face")
    #plt.savefig('gresil24fev2019.pdf', format='pdf')
    
    return nc,bool_intervalle
    
    
def classic_spectrum(start, filename, end=datetime(year=2000,month=1,day=1, second=17)):
    
    fig, ax = plt.subplots(1, figsize = (7,7))

    if end==datetime(year=2000,month=1,day=1,second=17):
        end = start + timedelta(seconds=10)
        
    if (end - start) <= timedelta(seconds=10):
        end = start + timedelta(seconds=10)

    current  = datetime(year=start.year, month=start.month, day=start.day, hour=start.hour)

    nc = xray.open_dataset(filename+str(current.year)\
                             +hrstr(current.month)+hrstr(current.day)+'_'+hrstr(current.hour)+'0000.nc')

    
    time_steps =  (nc.time.data >=np.datetime64(start)) & (nc.time.data <=np.datetime64(end))
    

    
    x,y = np.meshgrid( nc.spectrum_n_samples.data*12.0/nc.spectrum_n_samples.data[-1], nc.range.data )

    signal = np.mean( nc.variables['spectrum_raw'][time_steps],0)
    noise = np.min(signal[100,30])
    for i in range(np.shape( signal )[0] ):
        
        signal[i,:] =  signal[i,:] - noise
                
    cntf = ax.pcolormesh(x,y, signal  ,  cmap = 'Blues')   

#     ax.plot(x[100,55],y[100,55],'ko')
    
    cbZe=plt.colorbar(cntf,ax=ax)
    
    #ax.set_ylim([np.min(nc.variables['range'])-100,np.max(nc.variables['range']) ])

    #ax.plot(nc.variables['VEL'][time_step]*64/12, nc.variables['range'], 'r')
    ax.set_xlabel('Speed (m/s)')

    ax.set_ylabel('Height (m)')
    #ax[0].plot(np.max(nc.variables['spectrum_raw'][time_step],0), nc.variables['range'], 'r--')

#     for c in cntf.collections:
#         c.set_edgecolor("face")


    # plt.savefig('Raw_spectra_neige_24_1750UTC.pdf', format='pdf', bbox_inches='tight')


def Fall_Streak_Tracking(recgz,recu,recv,lon, lat, nc,b_i, lonlat_Mtl, time_0,height_gen):
        
    dist = np.sqrt((lon-lonlat_Mtl[0])**2 + (lat-lonlat_Mtl[1])**2 )
    ind_Mtl = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    u_Mtl = recu[:,ind_Mtl[0],ind_Mtl[1]]
    v_Mtl = recv[:,ind_Mtl[0],ind_Mtl[1]]
    heights_model = recgz[:,ind_Mtl[0],ind_Mtl[1]]

    heights_mrr = nc.variables['range'][:].data + 90
    times       = nc.time.data[b_i]
    from scipy.interpolate import interp1d

    f_u = interp1d(heights_model, u_Mtl)
    f_v = interp1d(heights_model, v_Mtl)

    u_inter = f_u(heights_mrr) * 0.514444 # m/s
    v_inter = f_v(heights_mrr) * 0.514444 # m/s

    delta_z = -30
    t_rad = np.ones(len(heights_mrr))*-9999


    time_en_sec = np.array((times-times[0]),dtype=float)*10**-9
    t_ini_sec   = np.array((time_0-times[0]),dtype=float)*10**-9

    tester_nan = np.nan
    cnt =-1
    while np.isnan(tester_nan)==True:
        Vd  = nc.variables['VEL'][:,cnt][b_i]
        f_Vd = interp1d(time_en_sec, Vd)
        tester_nan = -f_Vd(t_ini_sec)

        cnt += -1
#     cnt += -cnt_delta
#     height_gen = heights_mrr[cnt]
    uv_gen = np.array([f_u(height_gen)* 0.514444,f_v(height_gen)* 0.514444])

    u_gen = np.linalg.norm(uv_gen)


    t_rad[cnt+1] = t_ini_sec
    Vd_t = np.empty(len(heights_mrr), dtype=object)
    u_z  = np.empty(len(heights_mrr), dtype=object)
    z_gen = heights_mrr[cnt] 
    z = z_gen + delta_z
    error_cnt = 0
    lastVd_t = -2
    while z > heights_mrr[0]:

        u_z[cnt] = np.dot( np.array([u_inter[cnt], v_inter[cnt]]), uv_gen) / u_gen
        Vd  = nc.variables['VEL'][:,cnt][b_i]
        f_Vd = interp1d(time_en_sec, Vd)

        Vd_t[cnt] = f_Vd( t_rad[cnt+1] )

        if np.isnan(Vd_t[cnt])==False: 
            delta_t_rad = delta_z*(u_z[cnt] - u_gen)/Vd_t[cnt]/u_gen
            t_rad[cnt] = t_rad[cnt+1] + delta_t_rad
            lastVd_t = Vd_t[cnt]
        else:
            delta_t_rad = delta_z*(u_z[cnt] - u_gen)/lastVd_t/u_gen
            t_rad[cnt] = t_rad[cnt+1] + delta_t_rad
            error_cnt +=1
        cnt += -1
        z   += delta_z

    t_rad = np.ma.masked_object(t_rad,-9999)
    # datetime_trad = np.ones()

    datetime_trad = np.empty(len(t_rad), dtype=object)
    for i in np.arange(len(t_rad))[t_rad.mask==False]:
        datetime_trad[i] = times[0] + np.timedelta64(int(round(t_rad[i]*10**9)),'ns' )
    datetime_trad = np.ma.masked_object(datetime_trad,None)


    return datetime_trad, times, heights_mrr


def spectrum_from_fs(datetime_trad, nc, avg_time=10):
    
    fig, ax = plt.subplots(1, figsize = (7,7))

    signal = np.zeros([ len(datetime_trad) ,len(nc.spectrum_n_samples.data)])
    for i in range(len(datetime_trad)):
        if datetime_trad.mask[i]==False:
            time_steps=((nc.time.data>=datetime_trad[i]) & (nc.time.data<=datetime_trad[i]+np.timedelta64(10**9*avg_time) ))
            signal[i,:] = np.mean( nc.variables['spectrum_raw'][time_steps,i,:],0)
            
    x,y = np.meshgrid( nc.spectrum_n_samples.data*12.0/nc.spectrum_n_samples.data[-1], nc.range.data )

    cntf = ax.pcolormesh(x,y, signal,  cmap = 'Blues')   

    cbZe=plt.colorbar(cntf,ax=ax)
    
    ax.set_xlabel('Speed (m/s)')

    ax.set_ylabel('Height (m)')
    return signal


    
# def noise_removal(signal_vec):
#     mean_signal = np.mean(signal_vec)
#     mask_noise  = (signal_vec==np.max(mean_signal) ) 
#     mean_signal = np.ma(mean_signal, mask=mask_noise)
#     while mean_signal
    
    
    
    
def dB_to_normal(nc):
    new_nc = 10.**(nc.spectrum_raw/10.)
    nc['spectrum_raw_ML'] = (('time', 'n_spectra', 'spectrum_n_samples'), new_nc)
    nc['TimeStamps']      = (('time'), np.array(date2unix(dt64_to_datetime(nc.time) ) )) 
    nc['mrrRawNoSpec']=  int(10./(2.*nc.dims['spectrum_n_samples']*nc.dims['range']/500000.)) #From Metek notes
    return nc

def simple_hildebrand(nc):         
    raw_data_masked = np.ma.zeros(np.shape(nc.spectrum_raw.data))
    Variance_mat    = np.ma.zeros(np.shape(nc.spectrum_raw.data))
    Mask3D  = np.ma.zeros(np.shape(nc.spectrum_raw.data))
    n_spectra = 10./(2.*nc.dims['spectrum_n_samples']*nc.dims['range']/500000.) #From Metek notes
    import copy
    for i in range(nc.dims['time']):
        for ii in range(nc.dims['range']):
            range_spectrum = nc.spectrum_raw_ML.data[i,ii,3:61]
            range_spectrum_mod = copy.copy(range_spectrum)
            Variance = np.var(range_spectrum)
            Esquare  = (np.mean(range_spectrum))**2

            while Esquare < n_spectra*Variance:
                range_spectrum_mod = range_spectrum_mod[range_spectrum_mod<np.max(range_spectrum_mod)]
                Variance = np.var(range_spectrum_mod)
                Esquare  = (np.mean(range_spectrum_mod))**2

            noise = np.mean(range_spectrum_mod)
            raw_data_masked[i,ii,3:61] = range_spectrum-noise
            Mask3D[i,ii] = (raw_data_masked[i,ii,:]<10*np.sqrt(Variance))
            
            
            
    raw_data_masked.mask = Mask3D
    
    nc['spectrum_raw_ML'] = (('time', 'n_spectra', 'spectrum_n_samples'), raw_data_masked)
    return nc

def Calc_Ze_Eta(nc):

    CalibConst = nc.calibration_constant.data[0]
    rien,rangei = np.meshgrid(nc.spectrum_n_samples.data,np.arange(128)+1)
    deltaH = nc.range.data[10]-nc.range.data[9]
    
    eta_clear_sky = np.load('eta_clear_sky.npy')

    [eta_clear_sky_3D,rien,rien2] = np.meshgrid(eta_clear_sky, np.arange(len(nc.time.data)), nc.spectrum_n_samples.data)      
    [transfer3D,rien,rien2] = np.meshgrid(nc.transfer_function.data[0],np.arange(len(nc.time.data)), nc.spectrum_n_samples.data)      

    eta = nc.spectrum_raw_ML*CalibConst*rangei**2*deltaH/1e20/transfer3D \
    - eta_clear_sky_3D*0

    K2   = 0.92 #Maanh Kollias
#     K2   = 1 #Metek
    lamb = 0.01238
    Ze = 1e18*lamb**4*np.ma.sum(np.abs(eta),axis=-1)/np.pi**5/K2
    Ze = 10*np.ma.log10(Ze)
    eta_Z_ML = 1e18*lamb**4*np.abs(eta)/np.pi**5/K2   *  64/12
    eta_Z_ML = 10*np.ma.log10(eta_Z_ML)
    nc['Ze_ML'] = (('time', 'n_spectra'), Ze)
    nc['eta_ML'] = (('time', 'n_spectra', 'spectrum_n_samples'), eta)
    nc['eta_Z_ML'] = (('time', 'n_spectra', 'spectrum_n_samples'), eta_Z_ML)
    return nc


import calendar
def date2unix(dt_list):
    '''
    converts datetime object to seconds since 01-01-1970
    '''
    dt_list2 = list()
    for i in range(len(dt_list)):
        dt_list2.append( int(calendar.timegm(dt_list[i].timetuple())))
        
    return dt_list2

def dt64_to_datetime(datetime_obj_list):
    dt_list = list()
    for i in range(len(datetime_obj_list)):
        ts = (datetime_obj_list[i] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        dt_list.append(datetime.utcfromtimestamp(ts))
    return dt_list



def ImMrrProToo(file_name_list):
    for i in range(len(file_name_list)):
        nc = xray.open_dataset(filename_list[i])
        
        nc = Mrr_func.dB_to_normal(nc)
        processedSpec0 = corePro.MrrZe(nc)
        processedSpec0.rawToSnow()
        
        processedSpec0.writeNetCDF(file_name_list[i]+"_Dea.nc",ncForm="NETCDF3_CLASSIC")
    
    

    
    
    
    
    
    