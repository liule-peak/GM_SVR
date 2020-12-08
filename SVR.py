import pandas as pd
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt

inputfile = './datasave/new_reg_data_GM11.csv'  # 灰色预测后保存的路径
data = pd.read_csv(inputfile)  # 读取数据
data.index = range(1994, 2016)
feature = ['x1', 'x4', 'x5', 'x6', 'x7', 'x8']
data_train = data.loc[range(1994, 2014)].copy()  # 取2014年前的数据建模
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std  # 数据标准化

x_train = data_train[feature].values  # 特征数据
y_train = data_train['y'].values  # 标签数据
linearsvr = LinearSVR()  # 调用LinearSVR()函数
linearsvr.fit(x_train, y_train)
x = ((data[feature] - data_mean[feature]) /
     data_std[feature]).values  # 预测，并还原结果。
data[u'y_pred'] = linearsvr.predict(x) * \
    data_std['y'] + data_mean['y']
# SVR预测后保存的结果
outputfile = './datasave/new_reg_data_GM11_revenue.csv'
data.to_csv(outputfile)
print('真实值与预测值分别为：', data[['y', 'y_pred']])

p = data[['y', 'y_pred']].plot(style=['b-o', 'r-*'])
p.set_ylim(0, 2500)
p.set_xlim(1993, 2016)
plt.show()
