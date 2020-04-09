# import os
import numpy as np
# import xml.etree.ElementTree as ET
import pandas as pd
import wget
from datetime import datetime, timedelta
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from rpn_functions import *
import xml.etree.ElementTree as ET
import os


# 51157 : MONTREAL INTL A
# 49608 : MONTREAL MIRABEL INTL A
# 48374 : MONTREAL/ST-HUBERT
# 49568 : OTTAWA INTL A
# 51457 : QUEBEC INTL A
# 48371 : SHERBROOKE
# 51698 : TROIS RIVIERES A
# 50719 : OTTAWA GATINEAU A
# 50428 : KINGSTON A
# resulti, stinf = import_ECCC_obs_climweather(51157,'hourly', start, end)



def load_event_ECCC(start, end):
    
    startUTC = start+timedelta(hours=5)
    datestr = str(startUTC.year)+hrstr(startUTC.month)+hrstr(startUTC.day)

    #### Hourly Man Obs ######
    
#     St-Lawrence Valley Airports
#     list_id = [51157, 49608, 48374, 49568, 51457, 48371, 51698, 50719, 50428] 
    
#     station_info = list()
#     list_result = list()
#     for i in range(len(list_id)):
#         try:
#             resulti, stinf = import_ECCC_obs_climweather(list_id[i],'hourly', start, end)
#             list_result.append(resulti)
#             station_info.append(pd.DataFrame(dict({'Name': stinf.find('name').text, \
#             'latitude': stinf.find('latitude').text, 'longitude': stinf.find('longitude').text}) \
#             ,index =[1]) )
#         except:
#             print('No hourly at: '+ str(list_id[i]))

#     pd.concat(station_info).to_csv('Obs/StationInfo_Hourly'+datestr )
#     pd.concat(list_result).to_csv('Obs/'+datestr+'_Ec_Obs')

# #     ##### Hourly Temp Obs ######
# #     #Loading meteo stations list
    df = pd.read_csv('~/Scripts/repertoire_stations_2018.csv', sep=';', parse_dates=[0], header=None,\
                      names=['Name', 'Climate_ID', 'Station_ID','lat', 'lng', 'Elevation',\
                             'yini', 'yfin','Duration'])

#     # Stations to get (Coords + year)
    df_station = df[(df.lat>43.38847) & (df.lat<47.34) &\
                    (df.lng > -77.3 ) & (df.lng < -70.5) & (df.yfin>2016)]

    list_result = list()
    stations_info = list()
    for i in range(len(df_station)):
        station_id = df_station.iloc[i].Station_ID
        try:
            resulti, stinf = import_ECCC_obs_climweather(station_id,\
            'hourly', start, end)
            list_result.append(resulti)
            stations_info.append(pd.DataFrame(dict({'Name': stinf.find('name').text,\
            'latitude': stinf.find('latitude').text,'longitude': stinf.find('longitude').text})\
             ,index =[1]) )
        except:
            print('No hourly')

    pd.concat(list_result).to_csv('Obs/'+datestr+'_Ec_Temp')
    pd.concat(stations_info).to_csv('Obs/StationInfo_HourlyTemp'+datestr)
    

#     ##### Daily ######
#     list_result = list()
#     stations_info = list()
#     for i in range(len(df_station)):
#         station_id = df_station.iloc[i].Station_ID
#         try:
#             resulti, stinf = import_ECCC_obs_climweather(station_id, 'daily',\
#             start, end)   
#             list_result.append(resulti)
#             stations_info.append( pd.DataFrame(dict({'Name': stinf.find('name').text,\
#         'latitude': stinf.find('latitude').text,'longitude': stinf.find('longitude').text})\
#          ,index =[1]) )
#         except:
#             print('No daily')

#     pd.concat(list_result).to_csv('Obs/'+datestr+'_Ec_Daily')
#     pd.concat(stations_info).to_csv('Obs/StationInfo_Daily'+datestr)

    
#     ##### Daily Airports######
#     list_id = [51157, 49608, 48374, 49568, 51457, 48371, 51698, 50719, 50428] 
#     list_result = list()
#     stations_info = list()
#     for i in range(len(list_id)):
#         try:
#             resulti, stinf = import_ECCC_obs_climweather(list_id[i], 'daily',\
#             start, end)   
#             list_result.append(resulti)
#             stations_info.append( pd.DataFrame(dict({'Name': stinf.find('name').text,\
#         'latitude': stinf.find('latitude').text,'longitude': stinf.find('longitude').text})\
#          ,index =[1]) )
#         except:
#             print('No daily')

#     pd.concat(list_result).to_csv('Obs/'+datestr+'_Ec_DailyAirports')
#     pd.concat(stations_info).to_csv('Obs/StationInfo_DailyAirports'+datestr)

    
    
    
def import_ECCC_obs_climweather(ID_to_find, hourly_or_daily, start, end):

    if hourly_or_daily=='daily':
        
        start = datetime(year = start.year, month = start.month, day = start.day)
        end   = datetime(year = end.year, month = end.month, day = end.day+1)
        
        days = pd.date_range(start,end)

        yi    = start.year
        yf    = end.year

        tmpmin    = []
        tmpmax    = []
        tmpmoy    = []
        preciptot = []
        rain      = []
        snow      = []
        groundsnow = []  

        current = datetime(year=start.year, month=1, day=1)
        while current <= end :     ### Boucle sur les annees
            year  = current.year
            if current.year == start.year:
                day_since_year = start - datetime(year=start.year, month=1, day= 1, hour=0)
                day_index_start = day_since_year.days
            else:
                day_index_start = 0

            if current.year == end.year:
                day_since_year = end - datetime(year=end.year, month=1, day= 1, hour=0)
                day_index_end   = day_since_year.days
            else :
                day_index_end = 366 - np.ceil(np.mod(2003,4)/4) 

            current += timedelta(days=366 - np.ceil(np.mod(2003,4)/4))


            wget.download('http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=xml&stationID='+str(ID_to_find)+'&Year='+str(year)+'&Month=01&Day=14&timeframe=2')
            tmp_file ='eng-daily-0101'+str(year)+'-1231'+str(year)+'.xml' 
          #  #  # le fichier d entree est au format xml: besoin de connaitre l arborescence du fichier pour extraire la variable souhaitee: ici la precipitation
            tree=ET.parse(tmp_file)
            os.remove(tmp_file)
            root=tree.getroot()
                   ######### lecture du header de la station    #########
            stationsinfo = root.find('stationinformation')
#             print('nom station', stationsinfo[0].text, 'latitude', stationsinfo[2].text, 'longitude', stationsinfo[3].text, 'ANNEE', year)

                   ######### lecture des donnees brutes de la station    #########
            day_index = 0
            stationsdata = root.findall('.//stationdata')
            for  stationdata in stationsdata:
                if (day_index >= day_index_start) & (day_index <= day_index_end):
                    champs=stationdata.find("maxtemp")              
                    tmpmax.append(champs.text)   #  on fait une boucle sur les jours de l annee
                    ######### Travail sur les tasmin    #########
                    champs=stationdata.find("mintemp")              
                    tmpmin.append(champs.text)   #  on fait une boucle sur les jours de l annee
                    ######### Travail sur les tasmean    #########
                    champs=stationdata.find("meantemp")              
                    tmpmoy.append(champs.text)   #  on fait une boucle sur les jours de l annee

                    ######### Travail sur les precTOT   #########
                    champs=stationdata.find("totalprecipitation")              
                    preciptot.append(champs.text)   #  on fait une boucle sur les jours de l annee
                    ######### Travail sur la pluie       #########
                    champs=stationdata.find("totalrain")              
                    rain.append(champs.text)   #  on fait une boucle sur les jours de l annee
                    ######### Travail sur la neige       #########
                    champs=stationdata.find("totalsnow")              
                    snow.append(champs.text)   #  on fait une boucle sur les jours de l annee
                    ######### Travail sur la neige au sol   #########
                    champs=stationdata.find("snowonground")  
                    groundsnow.append(champs.text)   #  on fait une boucle sur les jours de l annee
                day_index += 1

        if not any(preciptot):
            raise ValueError('Only None type in preciptot')

        
        result = pd.DataFrame({'maxtemp': tmpmax, 'mintemp': tmpmin, 'meantemp': tmpmoy, 'totalprecipitation':  preciptot, 'totalrain': rain, 'totalsnow': snow, 'snowonground': groundsnow}, index=days)
        #result.to_csv(filepath, index=False)
        return result, stationsinfo


    elif hourly_or_daily=='hourly':

        hours = pd.date_range(start, end, freq='1h')

        yi    = start.year
        yf    = end.year

        mi   = start.month
        mf   = end.month

        temp      = []
        dptemp    = []
        relhum    = []
        winddir   = []
        windspd   = []
        visibility = []
        stnpress  = []  
        humidex   = []
        windchill = []
        weather   = []

        current = datetime(year=start.year, month=start.month, day=1)

        while current <= end :    ### Boucle sur les annees
            year  = current.year
            month = current.month
            if current.month == start.month:
                hour_since_month = start - datetime(year=start.year, month=start.month, day= 1, hour=0)
                hour_index_start = hour_since_month.days*24 + int(hour_since_month.seconds/3600)
            else:
                hour_index_start = 0

            if end.month == current.month:
                hour_since_month = end - datetime(year=end.year, month=end.month, day= 1, hour=0)
                hour_index_end   = hour_since_month.days*24 + int(hour_since_month.seconds/3600)
            else :
                hour_index_end = 24*monthrange(year, month)[1]


            current += timedelta(hours=hour_index_end+1)

            wget.download('http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=xml&stationID='\
            +str(ID_to_find)+'&Year='+str(year)+'&Month='+str(month)+'&Day=14&timeframe=1')

            tmp_file ='eng-hourly-'+hrstr(month)+'01'+str(year)+'-'+hrstr(month)+hrstr(monthrange(year, month)[1])+str(year)+'.xml' 

          #  #  # le fichier d entree est au format xml: besoin de connaitre l arborescence du fichier pour extraire la variable souhaitee: ici la precipitation
            tree=ET.parse(tmp_file)
            os.remove(tmp_file)
            root=tree.getroot()
                   ######### lecture du header de la station    #########
            stationsinfo=root.find('stationinformation')
#             print('nom station', stationsinfo[0].text, 'latitude', stationsinfo[2].text, 'longitude', stationsinfo[3].text, 'ANNEE', year)
            # print(root.findall('.//stationdata')[3].getchildren())
                   ######### lecture des donnees brutes de la station    #########

#             list_data_obs = ["temp","dptemp","relhum","winddir","stnpress"]
            stationsdata = root.findall('.//stationdata')
            hour_index = 0
            for  stationdata in stationsdata:
                if (hour_index >= hour_index_start) & (hour_index <= hour_index_end):
                    champs1=stationdata.find("temp")
                    temp.append(champs1.text)   #  on fait une boucle sur les jours du mois
                    ######### Travail sur les tasmin    #########
                    champs2=stationdata.find("dptemp")              
                    dptemp.append(champs2.text)   #  on fait une boucle sur les jours du mois
                    ######### Travail sur les tasmean    #########
                    champs3=stationdata.find("relhum")              
                    relhum.append(champs3.text)   #  on fait une boucle sur les jours du mois                     
                    ######### Travail sur les precTOT   #########
                    champs4=stationdata.find("winddir")              
                    winddir.append(champs4.text)   #  on fait une boucle sur les jours du mois
                    ######### Travail sur la pluie       #########
                    champs5=stationdata.find("visibility")              
                    visibility.append(champs5.text)   #  on fait une boucle sur les jours du mois
                    ######### Travail sur la neige       #########
                    champs6=stationdata.find("stnpress")              
                    stnpress.append(champs6.text)   #  on fait une boucle sur les jours du mois
                    ######### Travail sur la neige au sol   #########
                    champs7=stationdata.find("humidex")  
                    humidex.append(champs7.text)   #  on fait une boucle sur les jours du mois

                    champs8=stationdata.find("windchill")  
                    windchill.append(champs8.text)   #  on fait une boucle sur les jours du mois

                    champs9=stationdata.find("weather")  
                    if champs9.text != None:
                        # replaced = (champs.text).replace('Freezing Rain', 'FR')
                        # replaced = replaced.replace('Snow Grains', 'Gr')
                        # replaced = replaced.replace('Blowing Snow', 'BS')
                        # replaced = replaced.replace('Snow', 'S')
                        # replaced = replaced.replace('Cloudy', 'C')
                        # replaced = replaced.replace('Ice Pellets', 'IP')
                        # replaced = replaced.replace('NA', '-')
                        # replaced = replaced.replace('Moderate', '')
                        # replaced = replaced.replace('Mostly', '')
                        # replaced = replaced.replace('Mainly', '')
                        # replaced = replaced.replace('Clear', '')
                        # replaced = replaced.replace('Freezing Drizzle', 'FDz')
                        # replaced = replaced.replace('Drizzle', 'Dz')
                        weather.append(champs9.text)   #  on fait une boucle sur les jours du mois
                    else:
                        weather.append('-')

                hour_index += 1
        
        

        if not any(temp):
            print temp
            raise ValueError('Only None type in Temp')
        

        result = pd.DataFrame({"temp":temp,'dptemp':dptemp,'relhum':relhum,'winddir':winddir,'visibility':visibility,'stnpress':stnpress, 'humidex':humidex,'windchill':windchill,'weather':weather }, index=hours)

        return result, stationsinfo


# start = datetime(year=2017,month=1,day=23,hour=19)
# end   = datetime(year=2017,month=1,day=27,hour=0)
    
# start = datetime(year=2019,month=4,day=7,hour=15)
# end   = datetime(year=2019,month=4,day=10,hour=6)
    
# start = datetime(year=2019,month=2,day=23,hour=14)
# end   = datetime(year=2019,month=2,day=27,hour=0)

start = datetime(year=2019,month=2,day=2,hour=14)
end   = datetime(year=2019,month=2,day=5,hour=19)

load_event_ECCC(start, end)
    











