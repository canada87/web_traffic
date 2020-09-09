
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')

import pandas as pd
from pre_process_competition import preprocess_data
from sklearn.model_selection import train_test_split
from lib_models_regression import learning_reg

import streamlit as st

@st.cache
def load_file(name):
    print('here')
    df = pd.read_csv('data/'+name+'.csv')
    return df

def work_function(train, test, cols, start):
    results = pd.DataFrame()
    my_bar = st.progress(0)
    i = 0
    for col in cols:
        my_bar.progress(int((i/len(cols))*100))
        df_train2 = pd.DataFrame(train, columns = ['ds', col])
        df_train2 = df_train2.rename(columns = {'ds':'ds', col:'y'})
        models = learn.get_models(['Prophet'])
        models = learn.train_models(models, df_train2, 0)[0]
        y_pred = learn.predict_matrix_generator(models, test)
        y_pred.index = test.index
        results[col] = y_pred['Prophet']
        if plot:
            st.header(col)
            y_pred['Prophet'].plot()
            if start == 'train':
                test[col].plot(color='red')
            st.pyplot()
        i += 1
    return results

df = load_file('train_2')
df_train = df.copy()
df = load_file('key_2')
df_key = df.copy()
# df = load_file('sample_submission_1')
# df_sub = df.copy()
# st.write(df_key.head(70))
# st.write(df_sub.head(50))

prep = preprocess_data()
learn = learning_reg()

date = prep.submission_data_extraction(df_key)

df_train = prep.train_data_generator(df_train)
cols = df_train.columns

plot = st.sidebar.radio('plot', [False, True])
sub_sample = st.sidebar.radio('subsample', [True, False])
sub_num = int(st.sidebar.text_input('num subsample', 4))
# lag = int(st.sidebar.text_input('Lag', 5))
size_test = float(st.sidebar.text_input('size test', 0.08))

start = st.radio('go?', ['idle', 'train', 'submit'])

if sub_sample:
    df_test = pd.DataFrame()
    for i in range(sub_num):
        df_test[cols[i]] = df_train[cols[i]]
    cols = df_test.columns
    df_train = df_test
df_train['ds'] = df_train.index
if sub_sample:
    st.write(df_train.head())

# region_of_interest = 120
# segment = [i for i in range(df_train.shape[0]-region_of_interest, df_train.shape[0])]
# df_train = df_train.iloc[segment]

if st.button('start'):

    if start == 'train':

        train, test = train_test_split(df_train,shuffle = False, test_size = size_test)
        st.write('train len',train.shape[0])
        st.write('test len',test.shape[0])

        results = work_function(train,test,cols,start)

        st.write(results)

    if start == 'submit':

        train = df_train
        test = date
        st.write('train len',train.shape[0])
        st.write('test len',test.shape[0])

        results = work_function(train,test,cols,start)

        if plot:
            st.write(results)
        results.to_csv('sub_proph.csv', float_format='%.0f')

    # df = pd.read_csv('sub_proph.csv')
    # st.write(df)




##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

# import pandas as pd
# import streamlit as st
#
# #apre il file con l'abbinamento nome pagina -> chiave
# #con la funzione lambda toglie le date dai nome nella colonna page (gli ultimi 11 caratteri di ogni nome)
# df = pd.read_csv("./data/key_2.csv",converters={'Page':lambda p:p[:-11]}, index_col='Page')
#
# #apre il file dei dati e vede quante colonne ci sono per stabilire il range su cui fare i calcoli successivi
# df2 = pd.read_csv("./data/train_2.csv")
# len_col = len(df2.columns.tolist()) - 1
# predict_range = 60
#
# # #apre il file con i valori nella time table e seleziona solo le colonne dalla 755 alla 803 e mette la colonna page come indice
# df2 = pd.read_csv("./data/train_2.csv", usecols=[0]+list(range(len_col-predict_range,len_col)), index_col='Page')
#
# #calcola la mediana per ogni riga, senza leggere i NaN, sostituisce il dataframe con una colonna che contiene le mediane
# df2 = df2.median(axis=1,skipna=True)
#
# #la colonna viene rinominata Visits
# df2 = df2.to_frame(name='Visits')
#
# # unisce il file delle chiavi con quello delle mediane usande l'indice di sinistra come guida
# df3 = df.join(df2, how='left')
#
# #i NaN vengono sostituite con 0
# df3 = df3.fillna(0)
# st.write(df3.head(50))
#
# # il dataframe viene salvato su un file eliminando l'indice e usando il float come formato
# df3.to_csv('sub.csv', float_format='%.0f', index=False)
