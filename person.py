import numpy as np
import pandas as pd
#correlation of pearson

inputfile = './data.csv'
data = pd.read_csv(inputfile)
#print('the correlation of matrix: ',np.round(data.corr(method= 'pearson'),2))
data = np.round(data.corr(method = 'pearson'),2)
outputfile = './datasave/correlation of matrix.csv'
data.to_csv(outputfile)

data.sort_values(by=['y'],ascending=False,inplace=True)
data.reset_index(drop = True)
print(data['y'])

