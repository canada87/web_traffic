import pandas as pd
import numpy as np
import streamlit as st

import warnings

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from scipy.fftpack import fft

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot


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

def is_stationary(series):
    result = adfuller(series)
    st.write('ADF Statistic: %f' % result[0])# maggiore dei valori critici non e' stazionario
    isstat = 'is stationary' if result[1]<0.05 else 'is not stationary'
    st.write('p-value: ', result[1], isstat)# p-value > 0.05 non e' stazionario
    st.write('Critical Values:')
    for key, value in result[4].items():
        isstat = 'is stationary' if result[0]<value else 'is not stationary'
        st.write('\t%s: %.3f' % (key, value), isstat)

def eda(df_es_sum):
    st.write(df_es_sum.describe())

    pyplot.figure(3)
    pyplot.subplot(211)
    df_es_sum.plot()
    pyplot.subplot(212)
    groups = df_es_sum.groupby(pd.Grouper(freq='M'))
    frame = pd.DataFrame()
    for name, group in groups:
        df = pd.DataFrame(group.values)
        df.columns = [name]
        frame = pd.concat([frame, df], axis = 1)
    frame.boxplot()
    st.pyplot()

    days = [r for r in range(df_es_sum.shape[0])]
    fft_complex = fft(df_es_sum.tolist())
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]

    pyplot.ylabel('FFT Magnitude')
    pyplot.xlabel(r"Frequency [days]$^{-1}$")
    pyplot.title('Fourier Transform')
    pyplot.plot(fft_xvals[1:],fft_mag[1:])
    # pyplot.xlim(0,0.02)
    pyplot.axvline(x=1./7,color='red',alpha=0.3)#settimana
    pyplot.axvline(x=0.0054,color='red',alpha=0.3)#6 mesi
    st.pyplot()
    st.write('linea rossa sui 6 mesi e su 1 settimana')

    pyplot.figure(1)
    pyplot.subplot(211)
    df_es_sum.hist()
    pyplot.subplot(212)
    df_es_sum.plot(kind='kde')
    st.pyplot()
    st.write('la presenza di trend non lineari come esponenziali vengono osservati qui come una distribuzione non gaussiana')
    st.write('Se la distribuzione e gaussiana o flat vuol dire che non ci sono andamenti non lineari nei dati')
    st.write('Se ci sono andamenti non lineari puo valere la pena eliminarli per avvicinare i dati alla stazionarieta')

    st.subheader('correlation plot with lag')
    pd.plotting.lag_plot(df_es_sum)
    st.pyplot()

    st.subheader('autocorrelation on t')
    pd.plotting.autocorrelation_plot(df_es_sum)
    st.pyplot()

    pyplot.figure(2)
    pyplot.subplot(211)
    plot_acf(df_es_sum, lags = 20, ax=pyplot.gca())
    # A partial autocorrelation is a summary of the relationship between an observation in a time
    # series with observations at prior time steps with the relationships of intervening observations
    # removed.
    pyplot.subplot(212)
    plot_pacf(df_es_sum, lags = 20, ax=pyplot.gca())
    st.pyplot()
    st.write('act is related to the lag in the AR parameter (p), you should use the first value that drops within the significance region')
    st.write('pact is related to the lag in the MA parameter (q), you should use the first value that drops within the significance region')

    is_stationary(df_es_sum)


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

df_es_sum_diff = difference(df_es_sum,7)
df_es_sum_diff6m = difference(df_es_sum_diff,180)

temp_6m = [x for x in df_es_sum_diff6m[180:]]
temp_6m = pd.Series(temp_6m)
ind = df_es_sum_diff6m.index
ind = ind[180:]
temp_6m.index = ind

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_es_sum, model='additive')#model can be 'additive' or 'multiplicative'
print(result.trend)#trend present in the original data
print(result.seasonal)#seasonality present in the original data
print(result.resid)#original data without trend and seasonality
print(result.observed)#original data untouched
result.plot()
st.pyplot()

if st.checkbox('EDA'):
    st.title('original')
    eda(df_es_sum)

    st.title('differenciating')
    eda(df_es_sum_diff)

    st.title('differenciating 6M')
    eda(temp_6m)


df_es_sum = df_es_sum.astype('float32')
df_es_sum_diff = df_es_sum_diff.astype('float32')
df_es_sum_diff6m = df_es_sum_diff6m.astype('float32')

XY, validation = train_test_split(df_es_sum, shuffle = False, test_size = 0.1)
train, test = train_test_split(XY, shuffle = False, test_size = 0.3)

XY_diff, validation_diff = train_test_split(df_es_sum_diff, shuffle = False, test_size = 0.1)
train_diff, test_diff = train_test_split(XY_diff, shuffle = False, test_size = 0.3)

XY_diff6m, validation_diff6m = train_test_split(temp_6m, shuffle = False, test_size = 0.146)

#persisten model
####################################################################################################
st.title('persistent model')
df = pd.DataFrame(test.values)
persisten_pred = df.shift(-1)
persisten_pred.index = test.index
persisten_pred[0].iloc[-1] = persisten_pred[0].iloc[-2]

pyplot.plot(test)
pyplot.plot(persisten_pred, color = 'red')
st.pyplot()

rmse = sqrt(mean_squared_error(test, persisten_pred[0]))
st.write('persistent model RMSE ', rmse)
####################################################################################################

def evaluate_ARIMA(test_diff, train_diff, order):
    history = [train_diff[x] for x in range(train_diff.shape[0])]
    test_pred_diff = list()
    for t in range(test_diff.shape[0]):
        model = ARIMA(history, order = order)
        model_fit = model.fit(trend = 'nc', disp = 0)#disp sarebbe il verbose
        # model_fit = model.fit(disp = 0)
        pred = model_fit.forecast()[0]
        test_pred_diff.append(pred)
        history.append(pred)

    test_pred_diff = pd.Series(test_pred_diff)
    test_pred_diff.index = test_diff.index

    return test_pred_diff

def evaluation(test, test_pred, order, verbose = True):
    if verbose:
        test.plot()
        test_pred.plot(color = 'red')
        st.pyplot()

    rmse = sqrt(mean_squared_error(test, test_pred))
    st.write('persistent model RMSE ', rmse, ' order ', order)


def hyper_search(validation_diff6m, XY_diff6m, XY_diff, XY, validation):
    p_values = range(0, 7)
    d_values = range(0, 3)
    q_values = range(0, 4)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:

                    test_pred_diff = evaluate_ARIMA(validation_diff6m, XY_diff6m, order)
                    test_pred = invert_diff(XY_diff, test_pred_diff, 180)
                    test_pred = invert_diff(XY, test_pred, 7)
                    evaluation(validation, test_pred, order, verbose = False)

                except:
                    continue
    st.write('fine')


# hyper_search(validation_diff6m, XY_diff6m, XY_diff, XY, validation)

test_pred_diff = evaluate_ARIMA(validation_diff6m, XY_diff6m, order = (0,0,2))
test_pred = invert_diff(XY_diff, test_pred_diff, 180)
test_pred = invert_diff(XY, test_pred, 7)
evaluation(validation, test_pred, order = (0,0,2))

warnings.filterwarnings("ignore")
