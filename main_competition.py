
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

# import sys
# sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')

import pandas as pd
import numpy as np
from pre_process_competition import preprocess_data
from sklearn.model_selection import train_test_split, KFold
from lib_models_regression import learning_reg
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import streamlit as st

@st.cache
def load_file(name):
    df = pd.read_csv('data/'+name+'.csv')
    return df

@st.cache(allow_output_mutation=True)
def data_extraction_and_preparation(df_key, df_train):
    date = prep.submission_data_extraction(df_key)
    df_train = prep.train_data_generator(df_train)
    cols = df_train.columns
    return date, df_train, cols

def kfold_sampling(df, n_samples, shuffle = True):
    vetor_index = np.arange(df.shape[0])
    kfold = KFold(n_samples, shuffle)
    unfolder = kfold.split(vetor_index)

    train_dict = dict()
    test_dict = dict()
    for i, (train, test) in enumerate(unfolder):
        df_train = df.iloc[train].copy()
        df_test = df.iloc[test].copy()
        if shuffle:
            train_dict[i] = df_train.sample(frac = 1)
            test_dict[i] = df_test.sample(frac = 1)
        else:
            train_dict[i] = df_train
            test_dict[i] = df_test
    return train_dict, test_dict

def work_function(start, plot, X_train, Y_train, X_test, Y_test = None, net_type = 'normal'):

    #####################################################
    models = dict()
    models['deep learning '+net_type] = learn.get_deep_learning_model(input_dl = X_train.shape[1], output_dl = Y_train.shape[1], net_type = net_type)
    models, history = learn.train_models(models, X_train, Y_train, epochs = 10, batch_size = 2048)

    X_to_predict = X_test.copy()
    if net_type == 'vgg':
        xvgg_test = X_test.copy()
        xvgg_test = xvgg_test.to_numpy()
        xvgg_test = xvgg_test.reshape(xvgg_test.shape[0], xvgg_test.shape[1], 1)
        X_to_predict = xvgg_test.copy()
    y_pred = models['deep learning '+net_type].predict(X_to_predict)
    #####################################################

    y_pred = pd.DataFrame(y_pred)
    y_pred.index = X_test.index

    if start == 'train':
        y_pred.columns = Y_test.columns

    y_pred = y_pred.T

    mae = pd.DataFrame()
    for col in list(y_pred.columns):
        if start == 'train':
            mae[col] = [mean_absolute_error(Y_test.T[col], y_pred[col])]

        if plot:
            st.header(col)
            y_pred[col].plot()

            if start == 'train':
                Y_test.T[col].plot(color='red')

            st.write(mae.mean(axis=1))
            st.pyplot()

    return y_pred, mae.mean(axis=1)

df = load_file('train_2')
df_train = df.copy()
df = load_file('key_2')
df_key = df.copy()
df = load_file('sub_10epoc_2000batch')
df_results = df.copy()

# df = load_file('sample_submission_2')
# df_sub = df.copy()
# st.write(df_key.head(70))
# st.write(df_sub.head(50))

prep = preprocess_data()
learn = learning_reg()

date, df_train, cols = data_extraction_and_preparation(df_key, df_train)

plot = st.sidebar.radio('plot', [False, True])
sub_sample = st.sidebar.radio('subsample', [True, False])
sub_num = int(st.sidebar.text_input('num subsample', 4))
size_test = float(st.sidebar.text_input('size test', 0.3))
start = st.radio('go?', ['idle', 'train', 'submit', 'merge submission'])

if sub_sample:
    df_test = pd.DataFrame()
    for i in range(sub_num):
        df_test[cols[i]] = df_train[cols[i]]
    cols = df_test.columns
    df_train = df_test
df_train['ds'] = df_train.index

df_train = df_train.drop(['ds'], axis = 1)
df_train = df_train.T

if st.button('start'):

    if start == 'train':

        # train, test = train_test_split(df_train, test_size = size_test)
        # X_train = train.iloc[:,-122:-62]
        # Y_train = train.iloc[:,-62:]
        # X_test = test.iloc[:,-122:-62]
        # Y_test = test.iloc[:,-62:]
        # st.write('Xtrain len',X_train.shape)
        # st.write('Ytrain len',Y_train.shape)
        # st.write('Xtest len',X_test.shape)
        # st.write('Xtest len',Y_test.shape)
        # results, mae = work_function(start, plot, X_train, Y_train, X_test, Y_test, net_type = 'vgg')

        n_fold = 20
        train, test = kfold_sampling(df_train, n_fold)
        mae_fold = pd.DataFrame()
        mae_fold[0] = [float(0),float(0)]
        for i in range(n_fold):
            X_train = train[i].iloc[:,-122:-62]
            Y_train = train[i].iloc[:,-62:]
            X_test = test[i].iloc[:,-122:-62]
            Y_test = test[i].iloc[:,-62:]
            results, mae_norm = work_function(start, plot, X_train, Y_train, X_test, Y_test, net_type = 'normal')
            results, mae_vgg = work_function(start, plot, X_train, Y_train, X_test, Y_test, net_type = 'vgg')
            mae_fold[i] = [float(mae_norm), float(mae_vgg)]

        mae_fold = mae_fold.T
        mae_fold.columns = ['normal', 'vgg']
        st.write(mae_fold)
        mae_fold.plot.hist(alpha = 0.5)
        st.pyplot()

    if start == 'submit':

        submit_window = 60
        Ytrain_window = 62
        Xtrain_window = 60

        X_train = df_train.iloc[:,-(Ytrain_window+Xtrain_window):-Ytrain_window]
        Y_train = df_train.iloc[:,-Ytrain_window:]
        X_submit = df_train.iloc[:,-submit_window:]

        st.write('Xtrain len',X_train.shape)
        st.write('Ytrain len',Y_train.shape)
        st.write('Xsubmit len',X_submit.shape)

        results = work_function(start, plot, X_train, Y_train, X_submit)[0]
        results.index = date['ds']
        results.to_csv('sub_10epoc_2000batch.csv', float_format='%.0f')


    if start == 'merge submission':
        df_results.set_index('ds', inplace = True)

        st.write(df_results.head()[df_results.columns.to_list()[0:4]])
        st.write(df_results.T.head())
        st.write(df_key.head())

        df_results = df_results.T
        df_results = df_results.stack()
        df_results = df_results.reset_index()
        df_results['Page'] = df_results['level_0'] +'_'+ df_results['ds']
        df_results = df_results.drop(['level_0', 'ds'], axis = 1)
        df_results = df_results.rename(columns={0:'Visits'})
        final = pd.merge(df_key, df_results, on=['Page'], how='left')
        final = final.drop(['Page'], axis = 1)

        final.to_csv('sub_file_tot.csv', float_format='%.0f', index = False)

        df = pd.read_csv('sub_file_tot.csv')
        st.write(df.head())





# df_prova = pd.DataFrame()
# df_prova['pag1'] = [1,2,3,4,5,6,7,8]
# df_prova['pag2'] = [1,2,3,4,5,6,7,8]
# df_prova['pag3'] = [1,2,3,4,5,6,7,8]
# df_prova['pag4'] = [1,2,3,4,5,6,7,8]
# df_prova['pag5'] = [1,2,3,4,5,6,7,8]
# df_prova['pag6'] = [1,2,3,4,5,6,7,8]
# df_prova.index = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7', 'day8']
#
# df_prova = df_prova.T
# st.write(df_prova)
# df_prova = df_prova.stack()
# st.write(df_prova)
# df_prova = df_prova.reset_index()
# df_prova['Page'] = df_prova['level_0'] +'_'+ df_prova['level_1']
# df_prova = df_prova.drop(['level_0', 'level_1'], axis = 1)
# df_prova = df_prova.rename(columns={0:'Visit'})
# st.write(df_prova)

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
#
# import pandas as pd
# import streamlit as st
#
# st.write(df_sub.head())
#
# #apre il file con l'abbinamento nome pagina -> chiave
# #con la funzione lambda toglie le date dai nome nella colonna page (gli ultimi 11 caratteri di ogni nome)
# df = pd.read_csv("./data/key_2.csv",converters={'Page':lambda p:p[:-11]}, index_col='Page')
# st.write(df.head())
#
# #apre il file dei dati e vede quante colonne ci sono per stabilire il range su cui fare i calcoli successivi
# df2 = pd.read_csv("./data/train_2.csv")
# len_col = len(df2.columns.tolist()) - 1
# predict_range = 60
# st.write(df2.head())
#
# # #apre il file con i valori nella time table e seleziona solo le colonne dalla 755 alla 803 e mette la colonna page come indice
# df2 = pd.read_csv("./data/train_2.csv", usecols=[0]+list(range(len_col-predict_range,len_col)), index_col='Page')
# st.write(df2.head())
#
# #calcola la mediana per ogni riga, senza leggere i NaN, sostituisce il dataframe con una colonna che contiene le mediane
# df2 = df2.median(axis=1,skipna=True)
# st.write(df2.head())
#
# #la colonna viene rinominata Visits
# df2 = df2.to_frame(name='Visits')
# st.write(df2.head())
#
# # unisce il file delle chiavi con quello delle mediane usande l'indice di sinistra come guida
# df3 = df.join(df2, how='left')
# st.write(df3.head())
#
# #i NaN vengono sostituite con 0
# df3 = df3.fillna(0)
# st.write(df3.head(50))
#
# # il dataframe viene salvato su un file eliminando l'indice e usando il float come formato
# # df3.to_csv('sub.csv', float_format='%.0f', index=False)
