from __future__ import division
#encoding:utf-8
import pandas as pd
import numpy as np
'''
功能：计算回归分析模型中常用的四大评价指标
'''
 
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
def calPerformance(y_true,y_pred):
    '''
    模型效果指标评估
    y_true：真实的数据值
    y_pred：回归模型预测的数据值
    
    mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    ，其其值越小说明拟合效果越好。

    explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    的方差变化，值越小则说明效果越差。
    
    r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    变量的方差变化，值越小则说明效果越差。
    '''
    model_metrics_name=[mean_absolute_error, median_absolute_error,explained_variance_score, r2_score]  
    tmp_list=[]  
    for one in model_metrics_name:  
        tmp_score=one(y_true,y_pred)  
        tmp_list.append(tmp_score)  
    print(['mean_absolute_error','median_absolute_error','explained_variance_score','r2_score'])
    print(tmp_list)
    return tmp_list
 
if __name__=='__main__':
    inputfile = './datasave/new_reg_data_GM11_revenue.csv' #输入的数据文件
    data = pd.read_csv(inputfile)
    data.drop(data[np.isnan(data['y'])].index, inplace=True)
    y_pred = data['y_pred']
    y_true = data['y']
    calPerformance(y_true,y_pred)