# GM_SVR代码说明

### Lasso.py

在考虑一般的线性回归问题，给定n个数据样本点 ，其中每个X是一个d维的向量，即每个观测到的数据点是由d个变量的值组成的，每个y是一个实值。现在要做的是根据观察到的数据点，寻找到一个映射f使得误差平方和最小，Lasso优化目标为：
$$
\beta^* = argmin_\beta\frac{1}{n}||y-X\beta||^2_2+\lambda||\beta||_1
$$
模型的特征较多，需要压缩，选择Lasso回归是比较好好的选择。

使用

 `mask=lasso.coef_ != 0`

 `new_reg_data = data.iloc[:, mask]`

将系数非0的项保留，系数为0的删除。

### person.py

使用person方法计算相关系数。 `data = np.round(data.corr(method = 'pearson'),2)`

将数据原index扔掉，然后重置 `data.reset_index(drop = True)`

### model.py

构建灰色预测模型，并预测2014年和2015年的政财收入

#### GM11.py

典型的灰色预测函数

根据lasso和person确定要保留下的项 `l = ['x1', 'x3','x4', 'x5', 'x6', 'x7', 'x8','x13']`

将2014和2015的预测结果放入 `new_reg_data` 的l对应列表里

### SVR.py

使用支持向量回归做y的预测

根据lasso和person确定进行的项 `feature = ['x1', 'x3','x4', 'x5', 'x6', 'x7', 'x8','x13']`

之后将数据标准化，并调用调用 `LinearSVR()` 函数

### calPerformance.py

​		用来评估预测模型是否良好。

  

``` 
模型效果指标评估指标：
y_true：真实的数据值

y_pred：回归模型预测的数据值
mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度，其其值越小说明拟合效果越好。
explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
  ```

### data save文件夹

​		用来存储产生的中间和结果文件

### data.csv

​		待处理的源文件

### 
