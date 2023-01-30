import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import numpy as np
import random

SEED = 0

#################################### 5 best stations ##############################################
def st_kimathi():
    station_kimathi = pd.read_csv('full_rainset/TA00283.csv')
    station_kimathi[['timestamp (UTC)']] = station_kimathi[['timestamp (UTC)']].apply(pd.to_datetime)
    kimathi_arr = np.array(station_kimathi['pre'])
    kimathi_list = list(station_kimathi['pre'])
    kimathi = kimathi_arr.reshape((len(kimathi_list), 1))
    return kimathi

def st_tetu():
    station_tetu = pd.read_csv('full_rainset/TA00074.csv')
    station_tetu[['timestamp (UTC)']] = station_tetu[['timestamp (UTC)']].apply(pd.to_datetime)
    tetu_arr = np.array(station_tetu['pre'])
    tetu_list = list(station_tetu['pre'])
    tetu = tetu_arr.reshape((len(tetu_list), 1))
    return tetu

def st_chuka():
    station_chuka = pd.read_csv('full_rainset/TA00166.csv')
    station_chuka[['timestamp (UTC)']] = station_chuka[['timestamp (UTC)']].apply(pd.to_datetime)
    chuka_arr = np.array(station_chuka['pre'])
    chuka_list = list(station_chuka['pre'])
    chuka = chuka_arr.reshape((len(chuka_list), 1))
    return chuka

def st_embu():
    station_embu = pd.read_csv('full_rainset/TA00190.csv')
    station_embu[['timestamp (UTC)']] = station_embu[['timestamp (UTC)']].apply(pd.to_datetime)
    embu_arr = np.array(station_embu['pre'])
    embu_list = list(station_embu['pre'])
    embu = embu_arr.reshape((len(embu_list), 1))
    return embu

def st_kibugu():
    station_kibugu = pd.read_csv('full_rainset/TA00719.csv')
    station_kibugu[['timestamp (UTC)']] = station_kibugu[['timestamp (UTC)']].apply(pd.to_datetime)
    kibugu_arr = np.array(station_kibugu['pre'])
    kibugu_list =list(station_kibugu['pre'])
    kibugu= kibugu_arr.reshape((len(kibugu_list), 1))
    return kibugu
###################################################################################################

############################################# 4 worst stations ####################################

def st_thome():
    station_thome = pd.read_csv('full_rainset/TA00073.csv')
    station_thome[['timestamp (UTC)']] = station_thome[['timestamp (UTC)']].apply(pd.to_datetime)
    thome_arr = np.array(station_thome['pre'])
    thome_list = list(station_thome['pre'])
    thome = thome_arr.reshape((len(thome_list), 1))
    return thome

def st_muranga():
    station_muranga = pd.read_csv('full_rainset/TA00056.csv')
    station_muranga[['timestamp (UTC)']] = station_muranga[['timestamp (UTC)']].apply(pd.to_datetime)
    muranga_arr = np.array(station_muranga['pre'])
    muranga_list =list(station_muranga['pre'])
    muranga = muranga_arr.reshape((len(muranga_list), 1))
    return muranga

def st_murungaru():
    station_murungaru = pd.read_csv('full_rainset/TA00414.csv')
    station_murungaru[['timestamp (UTC)']] = station_murungaru[['timestamp (UTC)']].apply(pd.to_datetime)
    murungaru_arr = np.array(station_murungaru['pre'])
    murungaru_list = list(station_murungaru['pre'])
    murungaru = murungaru_arr.reshape((len(murungaru_list), 1))
    return murungaru

def st_karima():
    station_karima = pd.read_csv('full_rainset/TA00029.csv')
    station_karima[['timestamp (UTC)']] = station_karima[['timestamp (UTC)']].apply(pd.to_datetime)
    karima_arr = np.array(station_karima['pre'])
    karima_list = list(station_karima['pre'])
    karima = karima_arr.reshape((len(karima_list), 1))
    return karima
#######################################################################################################
def waterlev():
    df_water = pd.read_csv('muringato-sensor6.csv')
    df_water[['time']] = df_water[['time']].apply(pd.to_datetime,dayfirst = True)
    stage_list = list(df_water["Data"])
    data_list = []
    for i in stage_list:
        y = (i - min(stage_list)) / (max(stage_list) - min(stage_list))
        data_list.append(y)  
    lev = np.array(data_list)
    waterlevel = lev.reshape((len(stage_list), 1))
    return waterlevel


def waterlev2():
    df_water = pd.read_csv('muringato-sensor2.csv')
    df_water[['time']] = df_water[['time']].apply(pd.to_datetime,dayfirst = True)
    stage_list = list(df_water["Data"])
    data_list = []
    for i in stage_list:
        y = (i - min(stage_list)) / (max(stage_list) - min(stage_list))
        data_list.append(y)  
    lev = np.array(data_list)
    waterlevel = lev.reshape((len(stage_list), 1))
    return waterlevel
####################################################################################################################
def window_stage(series, sample_step):
    X, y = list(), list()
    for i in range(len(series)):
        end_ix = i + sample_step
        if end_ix > len(series)-1:
            break
        series_x, series_y = series[i:end_ix, :-1], series[end_ix-1, -1]
        X.append(series_x)
        y.append(series_y)
    return np.array(X), np.array(y)
####################################################################################################################
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
#################################################################################################################################
#######################################################differenced best stations#################################################
def s_kimathi():
    station_kimathi = pd.read_csv('full_rainset_diff/TA00283_diff.csv')
    station_kimathi[['timestamp (UTC)']] = station_kimathi[['timestamp (UTC)']].apply(pd.to_datetime)
    kimathi_arr = np.array(station_kimathi['difference'])
    kimathi_list = list(station_kimathi['difference'])
    kimathi = kimathi_arr.reshape((len(kimathi_list), 1))
    return kimathi

def s_tetu():
    station_tetu = pd.read_csv('full_rainset_diff/TA00074_diff.csv')
    station_tetu[['timestamp (UTC)']] = station_tetu[['timestamp (UTC)']].apply(pd.to_datetime)
    tetu_arr = np.array(station_tetu['difference'])
    tetu_list = list(station_tetu['difference'])
    tetu = tetu_arr.reshape((len(tetu_list), 1))
    return tetu

def s_chuka():
    station_chuka = pd.read_csv('full_rainset_diff/TA00166_diff.csv')
    station_chuka[['timestamp (UTC)']] = station_chuka[['timestamp (UTC)']].apply(pd.to_datetime)
    chuka_arr = np.array(station_chuka['difference'])
    chuka_list = list(station_chuka['difference'])
    chuka = chuka_arr.reshape((len(chuka_list), 1))
    return chuka

def s_embu():
    station_embu = pd.read_csv('full_rainset_diff/TA00190_diff.csv')
    station_embu[['timestamp (UTC)']] = station_embu[['timestamp (UTC)']].apply(pd.to_datetime)
    embu_arr = np.array(station_embu['difference'])
    embu_list = list(station_embu['difference'])
    embu = embu_arr.reshape((len(embu_list), 1))
    return embu

def s_kibugu():
    station_kibugu = pd.read_csv('full_rainset_diff/TA00719_diff.csv')
    station_kibugu[['timestamp (UTC)']] = station_kibugu[['timestamp (UTC)']].apply(pd.to_datetime)
    kibugu_arr = np.array(station_kibugu['difference'])
    kibugu_list =list(station_kibugu['difference'])
    kibugu= kibugu_arr.reshape((len(kibugu_list), 1))
    return kibugu
###############################################################################################################################
##################################################difference worst stations####################################################
def s_thome():
    station_thome = pd.read_csv('full_rainset_diff/TA00073_diff.csv')
    station_thome[['timestamp (UTC)']] = station_thome[['timestamp (UTC)']].apply(pd.to_datetime)
    thome_arr = np.array(station_thome['difference'])
    thome_list = list(station_thome['difference'])
    thome = thome_arr.reshape((len(thome_list), 1))
    return thome

def s_muranga():
    station_muranga = pd.read_csv('full_rainset_diff/TA00056_diff.csv')
    station_muranga[['timestamp (UTC)']] = station_muranga[['timestamp (UTC)']].apply(pd.to_datetime)
    muranga_arr = np.array(station_muranga['difference'])
    muranga_list =list(station_muranga['difference'])
    muranga = muranga_arr.reshape((len(muranga_list), 1))
    return muranga

def s_murungaru():
    station_murungaru = pd.read_csv('full_rainset_diff/TA00414_diff.csv')
    station_murungaru[['timestamp (UTC)']] = station_murungaru[['timestamp (UTC)']].apply(pd.to_datetime)
    murungaru_arr = np.array(station_murungaru['difference'])
    murungaru_list = list(station_murungaru['difference'])
    murungaru = murungaru_arr.reshape((len(murungaru_list), 1))
    return murungaru

def s_karima():
    station_karima = pd.read_csv('full_rainset_diff/TA00029_diff.csv')
    station_karima[['timestamp (UTC)']] = station_karima[['timestamp (UTC)']].apply(pd.to_datetime)
    karima_arr = np.array(station_karima['difference'])
    karima_list = list(station_karima['difference'])
    karima = karima_arr.reshape((len(karima_list), 1))
    return karima
#############################################################################################################################
def waterlev_diff():
    df_water = pd.read_csv('muringato-sensor6_diff.csv')
    df_water[['time']] = df_water[['time']].apply(pd.to_datetime,dayfirst = True)
    stage_list = list(df_water["difference"])
    data_list = []
    for i in stage_list:
        y = (i - min(stage_list)) / (max(stage_list) - min(stage_list))
        data_list.append(y)  
    lev = np.array(data_list)
    waterlevel = lev.reshape((len(stage_list), 1))
    return waterlevel