import sklearn 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

model_lr2 =  LinearRegression()

tips_df = sns.load_dataset('tips')

sns.scatterplot(data = tips_df, x = 'total_bill', y = 'tip')
# plt.show()

X = tips_df[['total_bill']]
y = tips_df[['tip']]

model_lr2.fit(X, y)

# 예측값 생성
y_true_tip = tips_df['tip']
y_pred_tip = model_lr2.predict(tips_df[['total_bill']])

print(y_true_tip[:5])

print(y_pred_tip[:5])

# pred 컬럼 추가하여 예측값 넣기
tips_df['pred'] = y_pred_tip
tips_df.head(3)

sns.lineplot(data = tips_df, x = 'total_bill', y = 'pred', color='red')
plt.show()