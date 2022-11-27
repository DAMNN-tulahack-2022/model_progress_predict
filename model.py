import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def codeMean(data, cat_feature, real_feature):
    """
    Возвращает словарь, где ключами являются уникальные категории признака cat_feature, 
    а значениями - средние по real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())

def prepareData(data, lag_start=3, lag_end=7):
    """ Подготовка данных для обучения и тестировния"""
    # размер тестовой выборки 7 дней - предсказание на неделю
    test_size = 7
    data = pd.DataFrame(data.copy())

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.experience.shift(i)

    data['weekday'] = data.index.weekday
    data['is_weekend'] = data.weekday.isin([5, 6])*1

    # считаем средние только по тренировочной части, чтобы избежать лика
    data['weekday_average'] = data.weekday.map(codeMean(data[:-test_size], 'weekday', 'experience').get)

    # выкидываем закодированные средними признаки 
    data.drop(['weekday'], axis=1, inplace=True)

    data = data.dropna()

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.iloc[:-test_size].drop(['experience'], axis=1)
    y_train = data.iloc[:-test_size]['experience']
    X_test = data.iloc[-test_size:].drop(['experience'], axis=1)
    y_test = data.iloc[-test_size:]['experience']

    return X_train, X_test, y_train, y_test

def dataToDataFrame(experience, authDate):
    """ Формирование исходного датафрейма """
    days = pd.date_range(authDate, periods=len(experience))
    data = {'experience': experience, 'day': days}
    work_data = pd.DataFrame(data)
    work_data = work_data.set_index('day')
    return work_data

def trainAndPredict(data):
    X_train, X_test, y_train, y_test = prepareData(data, lag_start=1, lag_end=4)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'linear_regression_model.pkl')

    prediction = model.predict(X_test)
    return prediction

    
    