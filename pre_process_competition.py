import pandas as pd
from sklearn.model_selection import train_test_split

# import streamlit as st


class preprocess_data:

    def train_data_generator(self, data):

        def parse_page(page):
            x = page.split('_')
            return ' '.join(x[:-3]), x[-3], x[-2], x[-1]

        def matrix_invert(data):
            # data = data.drop('project', axis = 1)
            data.set_index('Page', inplace = True)
            data = data.T
            data['date'] = data.index
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace = True)
            return data

        # page_info = list(data['Page'].apply(parse_page))
        # df_page_info = pd.DataFrame(page_info)
        # df_page_info.columns = ['name','project','access','agent']
        # data['project'] = df_page_info['project']
        # data_sub_selection = data[data['project'] == 'es.wikipedia.org']
        # data_sub_selection = matrix_invert(data_sub_selection)

        data = matrix_invert(data)
        data.fillna(0, inplace = True)
        return data

    def add_diff_lag(self, data, lag, with_diff = False):
        def add_multy_lag(X, lag, with_diff = False):
            X_pre = X.copy()
            for j in range(1, lag):
                for i in range(len(X_pre.columns)):
                    X[str(X_pre.columns[i])+'_'+str(j)+'_week_back'] = X[X_pre.columns[i]].shift(j)
                    if with_diff:
                        X[str(X_pre.columns[i])+'_'+str(j)+'_week_diff'] = X[str(X_pre.columns[i])+'_'+str(j)+'_week_back'].diff()
            X.fillna(method='bfill', inplace=True)
            return X

        data = add_multy_lag(data, lag, with_diff)
        return data

    def split_data(self, data, cols, lag, with_diff, size_test = 0.1):
        data_dict = dict()

        for col in cols:
            df_temp = pd.DataFrame()
            df_temp[col] = data[col]
            for i in range(1,lag):
                df_temp[col+'_'+str(i)+'_week_back'] = data[col+'_'+str(i)+'_week_back']
                if with_diff:
                    df_temp[col+'_'+str(i)+'_week_diff'] = data[col+'_'+str(i)+'_week_diff']
            X_temp = df_temp.drop(col, axis = 1)
            Y_temp = df_temp[col]
            train_x, test_x, train_y, test_y = train_test_split(X_temp, Y_temp, shuffle = False, test_size = size_test)
            data_dict[col] = train_x, test_x, train_y, test_y

        return data_dict


    def submission_data_extraction(self, data):

        def parse_page(page):
            x = page.split('_')
            return ' '.join(x[:-1]), x[-1]

        # st.write(data.head())
        page_info = list(data['Page'].apply(parse_page))

        df_page_info = pd.DataFrame(page_info)
        df_page_info.columns = ['page','date']
        random_page = df_page_info['page'].iloc[0]
        date = df_page_info[df_page_info['page'] == random_page]['date']

        date = pd.DataFrame(date)
        date['date'] = pd.to_datetime(date['date'])
        date['date2'] = date['date']
        date.set_index('date2', inplace = True)
        date = date.rename(columns = {'date':'ds'})

        return date
