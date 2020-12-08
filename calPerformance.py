from __future__ import division
# encoding:utf-8
import pandas as pd
import numpy as np

import Lasso
import model
import person
import SVR
# 计算回归分析模型中常用的四大评价指标

from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error, r2_score


def calPerformance(y_true, y_pred):

    model_metrics_name = [mean_absolute_error,
                          median_absolute_error, explained_variance_score, r2_score]
    tmp_list = []
    for one in model_metrics_name:
        tmp_score = one(y_true, y_pred)
        tmp_list.append(tmp_score)
    print(['mean_absolute_error', 'median_absolute_error',
           'explained_variance_score', 'r2_score'])
    print(tmp_list)
    return tmp_list


if __name__ == '__main__':
    Lasso
    person
    model
    SVR
    inputfile = './datasave/new_reg_data_GM11_revenue.csv'  # 输入的数据文件
    data = pd.read_csv(inputfile)
    data.drop(data[np.isnan(data['y'])].index, inplace=True)
    y_pred = data['y_pred']
    y_true = data['y']
    calPerformance(y_true, y_pred)
