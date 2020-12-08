#构建灰色预测模型，并预测2014年和2015年的政财收入
import numpy as np
import pandas as pd
from GM11 import GM11 #引入灰色预测函数
inputfile = './datasave/new_reg_data.csv' #输入的数据文件
inputfile1 = './data.csv' #输入的数据文件
new_reg_data = pd.read_csv(inputfile) #读取经过特征选择后的数据
data = pd.read_csv(inputfile1) #读取经过特征选择后的数据
new_reg_data.index = range(1994,2014)
new_reg_data.loc[2014] = None
new_reg_data.loc[2015] = None
l = ['x1', 'x4', 'x5', 'x6', 'x7', 'x8']
for i in l:
    #new_reg_data.loc[range(1994,2014),i]获取l(i)的列数据
    f = GM11(new_reg_data.loc[range(1994,2014),i].values)[0]
    ##将2014和2015的预测结果放入new_reg_data的l对应列表里
    new_reg_data.loc[2014,i] = f(len(new_reg_data)-1)#2014年预测结果
    new_reg_data.loc[2015,i] = f(len(new_reg_data)) ##2015年预测结果
    new_reg_data[i] = new_reg_data[i].round(2) ## 保留两位小数
outputfile = './datasave/new_reg_data_GM11.csv' ## 灰色预测后保存的路径
y = list(data['y'].values) ## 提取财政收入列，合并至新数据框中
y.extend([np.nan,np.nan])
new_reg_data['y'] = y
new_reg_data.to_csv(outputfile) ## 结果输出
print('预测结果为：',new_reg_data.loc[2014:2015,:]) ##预测结果展示