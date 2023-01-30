import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.rcParams['figure.figsize'] = (7,5)
import pandas as pd
import pandas as pd 
import numpy as np
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib.dates import DateFormatter
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM, Bidirectional
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBRegressor
from numpy.random import seed
import tensorflow as tf
tf.random.set_seed(0) 
seed(0)

SEED = 0
#########################################################################################################
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
#########################################################################################################
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

########## TRAIN_TEST_SPLIT ################################################################################
def train_test_data_split(dataset, h_step, f_step):
    x, y = [], []
    for i in range(h_step, len(dataset)-h_step):
        x.append(dataset[i - h_step:i])
        y.append(dataset[i : i+f_step])
    return np.array(x),np.array(y) 

###########NEW_SPLIT FUNCTION############################################################################## 
def split_stage(series, h_step, f_step):
    x, y = list(), list()
    for i in range(len(series)):
        win_end_indx = i + h_step
        future_indx = win_end_indx + f_step
        if future_indx > len(series):
            break
        series_x, series_y = series[i:win_end_indx], series[win_end_indx:future_indx ]
        x.append(series_x)
        y.append(series_y)
    return np.array(x), np.array(y)

########### MODEL_EVALUATION ###############################################################################
def evaluate_model(y_test , y_pred):
    scores = []
    #scores for each day
    for i in range (y_test.shape[1]):
        mse = mean_squared_error(y_test[:,i], y_pred[:,i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
    #scores for the whole prediction exercise
    overall_score = 0
    for row in range (y_test.shape[0]):
        for col in range (y_pred.shape[1]):
            overall_score = overall_score + (y_test[row,col]-y_pred[row,col])**2
    overall_score = np.sqrt(overall_score/(y_test.shape[0] * y_pred.shape[1]))
    return   scores[0]

########### Model_mlp#######################################################################################
def model_mlp(x_train, y_train , x_test, y_test):
    model_mlp = MLPRegressor(hidden_layer_sizes=(200,100,50,30,10),random_state = 0, max_iter = 2000)
    model_mlp.fit(x_train, y_train)
    y_pred = model_mlp.predict(x_test)
    t_score = evaluate_model(y_test,y_pred)
    return t_score

########### Model_xg_boost#################################################################################
def model_x_boost(x_train, y_train, x_test, y_test):
    model_x_boost = XGBRegressor(objective='reg:squarederror', n_estimators=1000,learning_rate=0.30000081)
    model_x_boost.fit(x_train, y_train)
    y_pred = model_x_boost.predict(x_test)
    t_score = evaluate_model(y_test,y_pred)
    return t_score  

############# Model_lstm###################################################################################
def model_lstm(x_train_lstm, y_train, x_test_lstm, y_test, n_features):
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(150, activation='relu'), input_shape=(x_train_lstm.shape[1], n_features)))
    model.add(Dense(y_test.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    history = model.fit(x_train_lstm, y_train, epochs=40, batch_size= 70, validation_data=(x_test_lstm, y_test), verbose=0, shuffle=False)
    y_pred = model(x_test_lstm)
    t_score = evaluate_model(y_test,y_pred)
    return t_score
################### MAIN_FUNCTION #########################################################################
def water_main(series,model,f_step):
    df = series.drop(['time'],axis=1)
    df2 = normalize(df)
    x = df2[df2.columns[0]].to_numpy()
    data_size = int(x.shape[0] * .80)
    data_train = x[:data_size].flatten()
    data_test = x[data_size:].flatten()
    histo_step = np.arange(1,16)
    n_features = 1
    total_score = []
    for i in histo_step:
        x_train, y_train = split_stage(data_train, i , f_step)
        x_test, y_test = split_stage(data_test, i , f_step )
        x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
        x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
        # model selection
        if model == 'mlp':
            t_out = model_mlp(x_train , y_train, x_test, y_test)
        elif model == 'x_boost':
            t_out = model_x_boost(x_train, y_train, x_test, y_test)
        elif model == 'lstm': 
            t_out = model_lstm(x_train_lstm, y_train, x_test_lstm, y_test, n_features)  
        total_score.append(t_out)
        
    print('optimal_window_size:', histo_step[np.argmin(total_score)]) 
    x_plot = list(histo_step)
    y_plot = list(total_score)
    plt.plot(x_plot, y_plot,linewidth=3)
    plt.title('RMSE vs window size (p)',fontsize=15,weight ='bold')
    plt.xlabel('window size (p)',fontsize =10,weight ='bold')
    plt.ylabel('RMSE',fontsize =10,weight ='bold')
    plt.grid(True) 
    #plt.savefig('hori4.png', dpi=450, orientation='portrait', bbox_inches='tight', facecolor='w',edgecolor='b',)
    plt.show()
############################################################################################################         


