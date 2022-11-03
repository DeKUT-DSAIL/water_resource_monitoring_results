import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.rcParams['figure.figsize'] = (7,5)
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import numpy as np
import random
import warnings
warnings. filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM, Bidirectional
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

SEED = 0
############################################################################################
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
#############################################################################################
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
#############################################################################################
def window_historical_stage(series, future_step, window_step):
    X, y = list(), list()
    for i in range(len(series)):
        end_ix = i + window_step
        if end_ix > len(series)-1:
            break
        series_x, series_y = series[i:end_ix - (future_step - 1), :-1], series[end_ix, -1]
        X.append(series_x)
        y.append(series_y)
    return np.array(X), np.array(y)
#########################################################################################################

set_global_determinism()

##########################################################################################################
###############################################STATIONS ONLY###########################################################
##########################################################################################################
def optimal_window_stations(future_step):
    warnings. filterwarnings('ignore')
    date_df = pd.read_csv('muringato-sensor6.csv')
    df = pd.read_csv('original_set_raw.csv')
    df_normalized = normalize(df)
    dataset= df_normalized[df_normalized.columns[0:9]].to_numpy()
    histo_step = np.arange(future_step,21)
    r_squared_list = []
    for i in histo_step:
        x,y = window_historical_stage(dataset, future_step, i)
        X = x.reshape((x.shape[0], (x.shape[1]*x.shape[2])))
        data_size = int(X.shape[0] * .8)
        x_train, y_train = X[:data_size], y[:data_size] 
        x_test, y_test = X[data_size:], y[data_size:]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        #### model####
        model = Sequential()
        model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(x_train, y_train, epochs=6, batch_size=35, validation_data=(x_test, y_test), verbose=0, shuffle=False)
        y_pred = model(x_test)
        warnings. filterwarnings('ignore')
        #y_pred = y_pred.flatten()
        r_squared = r2_score(y_test, y_pred)
        r_squared_list.append(r_squared)
    x_plot = list(histo_step)
    y_plot = list(r_squared_list)
    plt.plot(x_plot, y_plot,linewidth=3)
    plt.title('R2 score vs Historical window size',fontsize=15,weight ='bold')
    plt.xlabel('Historical window size',fontsize =10,weight ='bold')
    plt.ylabel('R2 score',fontsize =10,weight ='bold')
    plt.grid(True)        
##########################################################################################################
def stations_main(future_step):
    warnings. filterwarnings('ignore')
    date_df = pd.read_csv('muringato-sensor6.csv')
    df = pd.read_csv('original_set_raw.csv')
    df_normalized = normalize(df)
    dataset= df_normalized[df_normalized.columns[0:9]].to_numpy()
    histo_step = np.arange(future_step,21)
    r_squared_list = []
    for i in histo_step:
        x,y = window_historical_stage(dataset, future_step, i)
        X = x.reshape((x.shape[0], (x.shape[1]*x.shape[2])))
        data_size = int(X.shape[0] * .8)
        x_train, y_train = X[:data_size], y[:data_size] 
        x_test, y_test = X[data_size:], y[data_size:]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        #### model####
        model = Sequential()
        model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(x_train, y_train, epochs=6, batch_size=35, validation_data=(x_test, y_test), verbose=0, shuffle=False)
        y_pred = model(x_test)
        warnings. filterwarnings('ignore')
        #y_pred = y_pred.flatten()
        r_squared = r2_score(y_test, y_pred)
        r_squared_list.append(r_squared)
        t = np.argmax(r_squared_list)
        optimal_window = histo_step[t]
        ####################################################
        u,v = window_historical_stage(dataset, future_step, optimal_window)
        U = u.reshape((u.shape[0], (u.shape[1]*u.shape[2])))
        data_size2 = int(U.shape[0] * .8)
        u_train, v_train = U[:data_size2], v[:data_size2] 
        u_test, v_test = U[data_size2:], v[data_size2:]
        u_train = u_train.reshape((u_train.shape[0], 1, u_train.shape[1]))
        u_test = u_test.reshape((u_test.shape[0], 1, u_test.shape[1]))
        #### model####
        optimodel = Sequential()
        optimodel.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(u_train.shape[1], u_train.shape[2])))
        optimodel.add(Dense(1))
        optimodel.compile(loss='mae', optimizer='adam')
        history = optimodel.fit(u_train, v_train, epochs=6, batch_size=35, validation_data=(u_test, v_test), verbose=0, shuffle=False)
        v_pred = optimodel(u_test)
        warnings. filterwarnings('ignore')
        #v_pred = v_pred.flatten()
        rmse = np.sqrt(mean_squared_error(v_test, v_pred))
######################################################
    print('RMSE: ', rmse )
    df3 = date_df.tail(v_test.shape[0])
    df4 = df3.drop(['Data'], axis = 1)
    df4['v_test'] = v_test
    df4['v_pred'] = v_pred
    df4[['time']] = df4[['time']].apply(pd.to_datetime,dayfirst=True)
    #Plot the output
    fig, ax = plt.subplots(1,1)
    fig.patch.set_facecolor('white')
    Test, = plt.plot(df4['time'],list(v_test),linewidth=3, label='label1')
    Prediction, = plt.plot(df4['time'], list(v_pred), linewidth=3, label='label1')
    #ax.grid(color = 'gray', linestyle = '--', linewidth = 1.5)
    ax.set_title('Test vs. Prediction',fontsize=15,weight = 'bold')
    ax.set_xlabel('Time',fontsize=15,weight = 'bold')
    ax.set_ylabel('Waterlevel',fontsize=15, weight = 'bold')
    ax.set_ylim(0,0.5)
    ax.tick_params(axis='both',labelsize=10)
    ax.tick_params(axis = 'x', labelsize = 10)
    ax.grid(True)
    date_form = DateFormatter("%d-%m")
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(["Test", "Prediction"], loc ="upper left", fancybox=True,facecolor='#01FFFF',prop={'size': 15,  'style': 'normal'})
    ax.set(facecolor = "white")
    #plt.savefig('rain_water.png', dpi=450, orientation='portrait', bbox_inches='tight', facecolor='w',edgecolor='b',)
    plt.show()
###############################################################################################################
###############################################################################################################

################################################################################################################
######################################STATION + WATER ONLY #####################################################
################################################################################################################
def stations_water_main(future_step):
    date_df = pd.read_csv('muringato-sensor6.csv')
    df = pd.read_csv('original_set_raw+water.csv')
    df_normalized = normalize(df)
    dataset= df_normalized[df_normalized.columns[0:10]].to_numpy()
    histo_step = np.arange(future_step,21)
    r_squared_list = []
    for i in histo_step:
        x,y = window_historical_stage(dataset, future_step, i)
        X = x.reshape((x.shape[0], (x.shape[1]*x.shape[2])))
        data_size = int(X.shape[0] * .8)
        x_train, y_train = X[:data_size], y[:data_size] 
        x_test, y_test = X[data_size:], y[data_size:]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        #### model####
        model = Sequential()
        model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(x_train, y_train, epochs=6, batch_size=35, validation_data=(x_test, y_test), verbose=0, shuffle=False)
        y_pred = model(x_test)
        warnings. filterwarnings('ignore')
        #y_pred = y_pred.flatten()
        r_squared = r2_score(y_test, y_pred)
        r_squared_list.append(r_squared)
        t = np.argmax(r_squared_list)
        optimal_window = histo_step[t]
        ####################################################
        u,v = window_historical_stage(dataset, future_step, optimal_window)
        U = u.reshape((u.shape[0], (u.shape[1]*u.shape[2])))
        data_size2 = int(U.shape[0] * .8)
        u_train, v_train = U[:data_size2], v[:data_size2] 
        u_test, v_test = U[data_size2:], v[data_size2:]
        u_train = u_train.reshape((u_train.shape[0], 1, u_train.shape[1]))
        u_test = u_test.reshape((u_test.shape[0], 1, u_test.shape[1]))
        #### model####
        optimodel = Sequential()
        optimodel.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(u_train.shape[1], u_train.shape[2])))
        optimodel.add(Dense(1))
        optimodel.compile(loss='mae', optimizer='adam')
        history = optimodel.fit(u_train, v_train, epochs=6, batch_size=35, validation_data=(u_test, v_test), verbose=0, shuffle=False)
        v_pred = optimodel(u_test)
        warnings. filterwarnings('ignore')
        #v_pred = v_pred.flatten()
        rmse = np.sqrt(mean_squared_error(v_test, v_pred))
######################################################
    print('RMSE: ', rmse )
    df3 = date_df.tail(v_test.shape[0])
    df4 = df3.drop(['Data'], axis = 1)
    df4['v_test'] = v_test
    df4['v_pred'] = v_pred
    df4[['time']] = df4[['time']].apply(pd.to_datetime,dayfirst=True)
    #Plot the output
    fig, ax = plt.subplots(1,1)
    fig.patch.set_facecolor('white')
    Test, = plt.plot(df4['time'],list(v_test),linewidth=3, label='label1')
    Prediction, = plt.plot(df4['time'], list(v_pred), linewidth=3, label='label1')
    #ax.grid(color = 'gray', linestyle = '--', linewidth = 1.5)
    ax.set_title('Test vs. Prediction',fontsize=15,weight = 'bold')
    ax.set_xlabel('Time',fontsize=15,weight = 'bold')
    ax.set_ylabel('Waterlevel',fontsize=15, weight = 'bold')
    ax.set_ylim(0,0.46)
    ax.tick_params(axis='both',labelsize=10)
    ax.tick_params(axis = 'x', labelsize = 10)
    ax.grid(True)
    date_form = DateFormatter("%d-%m")
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(["Test", "Prediction"], loc ="upper left", fancybox=True,facecolor='#01FFFF',prop={'size': 15,  'style': 'normal'})
    ax.set(facecolor = "white")
    #plt.savefig('rain_water.png', dpi=450, orientation='portrait', bbox_inches='tight', facecolor='w',edgecolor='b',)
    plt.show()
###################################################################################################################
def optimal_window_stations_water(future_step):
    date_df = pd.read_csv('muringato-sensor6.csv')
    df = pd.read_csv('original_set_raw+water.csv')
    df_normalized = normalize(df)
    dataset= df_normalized[df_normalized.columns[0:10]].to_numpy()
    histo_step = np.arange(future_step,21)
    r_squared_list = []
    for i in histo_step:
        x,y = window_historical_stage(dataset, future_step, i)
        X = x.reshape((x.shape[0], (x.shape[1]*x.shape[2])))
        data_size = int(X.shape[0] * .8)
        x_train, y_train = X[:data_size], y[:data_size] 
        x_test, y_test = X[data_size:], y[data_size:]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        #### model####
        model = Sequential()
        model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(x_train, y_train, epochs=6, batch_size=35, validation_data=(x_test, y_test), verbose=0, shuffle=False)
        y_pred = model(x_test)
        warnings. filterwarnings('ignore')
        #y_pred = y_pred.flatten()
        r_squared = r2_score(y_test, y_pred)
        r_squared_list.append(r_squared)
    x_plot = list(histo_step)
    y_plot = list(r_squared_list)
    plt.plot(x_plot, y_plot,linewidth=3)
    plt.title('R2 score vs Historical window size',fontsize=15,weight ='bold')
    plt.xlabel('Historical window size',fontsize =10,weight ='bold')
    plt.ylabel('R2 score',fontsize =10,weight ='bold')
    plt.grid(True)   
####################################################################################################################        
####################################################################################################################
####################################################################################################################
    
    
    
    