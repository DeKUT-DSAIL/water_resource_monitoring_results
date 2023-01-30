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
    #scores for the whole prediction exercise
    overall_score = 0
    for row in range (y_test.shape[0]):
        for col in range (y_pred.shape[1]):
            overall_score = overall_score + (y_test[row,col]-y_pred[row,col])**2
    overall_score = np.sqrt(overall_score/(y_test.shape[0] * y_pred.shape[1]))
    return overall_score

###################AVERAGING BASELINE######################################################################
def av_model_baseline(x_test, y_test,  f_step):
    pred_list = []
    for i in x_test:
        i_mean = np.mean(i)
        pred_list.append([i_mean])
    pred_array = np.array(pred_list)
    y_pred = np.repeat(pred_array, f_step, axis=1)
    t_score = evaluate_model(y_test,y_pred)
    return t_score

###################NAIVE_BASELINE##########################################################################
def naive_baseline(x_test, y_test,  f_step):
    pred_array = np.column_stack([x_test[:,x_test.shape[1]-1]])
    y_pred = np.repeat(pred_array, f_step, axis=1)
    t_score = evaluate_model(y_test,y_pred)
    return t_score

################### MAIN_FUNCTION #########################################################################
def water_main(series,model,f_step):
    df = series.drop(['time'],axis=1)
    df2 = normalize(df)
    x = df[df.columns[0]].to_numpy()
    data_size = int(x.shape[0] * .80)
    data_train = x[:data_size].flatten()
    data_test = x[data_size:].flatten()
    histo_step = np.arange(1, 16)
    total_score = []
    for i in histo_step:
        x_train, y_train = split_stage(data_train, i , f_step)
        x_test, y_test = split_stage(data_test, i , f_step )
        
        # model selection
        if model =='av_baseline':
            t_out = av_model_baseline(x_test, y_test, f_step)
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


