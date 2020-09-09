import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.ensemble import RandomForestRegressor


@st.cache
def load_file(name):
    df = pd.read_csv('data/'+name+'.csv')
    return df

def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]

def create_lag_col(df_series, lag = -1):
    df = pd.DataFrame(df_series.values)
    df_shift = pd.concat([df.shift(lag), df], axis = 1)
    df_shift.columns = ['t - '+str(lag), 'original']
    return df_shift

def difference(series,lag):
    diff_ser = series.diff(lag)
    diff_ser.fillna(method='bfill', inplace = True)
    return diff_ser

def invert_diff(train, predict, lag):
    ind = predict.index
    back_to_life = [0 for i in range(predict.shape[0])]

    train = train.copy().tolist()
    predict = predict.copy().tolist()
    train_len = len(train)

    for i in range(len(back_to_life)):
        back_to_life[i] = train[train_len-lag+i] + predict[i]
        train.append(back_to_life[i])

    back_to_life = pd.Series(back_to_life)
    back_to_life.index = ind
    return back_to_life

def add_multy_lag(X, lag):
    X_pre = X.copy()
    st.write(X_pre)
    for j in range(1, lag):
        for i in range(len(X_pre.columns)):
            X[str(X_pre.columns[i])+'_'+str(j)+'_week_back'] = X[X_pre.columns[i]].shift(j)
            X[str(X_pre.columns[i])+'_'+str(j)+'_week_diff'] = X[str(X_pre.columns[i])+'_'+str(j)+'_week_back'].diff()
    X.fillna(method='bfill', inplace=True)
    return X

    # ███    ███  █████  ██ ███    ██
    # ████  ████ ██   ██ ██ ████   ██
    # ██ ████ ██ ███████ ██ ██ ██  ██
    # ██  ██  ██ ██   ██ ██ ██  ██ ██
    # ██      ██ ██   ██ ██ ██   ████


df = load_file('train_1')
df_train = df.copy()

split_page = list(df_train['Page'].apply(parse_page))

df_split = pd.DataFrame(split_page)
df_split.columns = ['name','project','access','agent']

df_train['project'] = df_split['project']
df_es = df_train[df_train['project'] == 'es.wikipedia.org']
df_es = df_es.drop('project', axis = 1)
df_es.set_index('Page', inplace = True)
df_es = df_es.T
df_es['date'] = df_es.index
df_es['date'] = pd.to_datetime(df_es['date'])
df_es.set_index('date', inplace = True)

df_es_sum = df_es.sum(axis=1)
df_es_sum /= df_es.shape[1]

df_df_es_sum = pd.DataFrame(df_es_sum.values)
df_df_es_sum.index = df_es_sum.index
df_df_es_sum = add_multy_lag(df_df_es_sum,5)

X=df_df_es_sum.drop(0, axis=1)
Y=df_df_es_sum[0]

train_x, test_x, train_y, test_y = train_test_split(X, Y, shuffle = False, test_size = 0.1)
def evaluation(test, test_pred, verbose = True):
    if verbose:
        test.plot()
        test_pred.plot(color = 'red')
        st.pyplot()

    rmse = sqrt(mean_squared_error(test, test_pred))
    st.write('persistent model RMSE ', rmse)

rf = RandomForestRegressor()
rf.fit(train_x, train_y)
pred_y = rf.predict(test_x)
pred_y = pd.Series(pred_y)
pred_y.index = test_y.index
test_y = pd.Series(test_y.values)
test_y.index = pred_y.index
evaluation(test_y, pred_y)
