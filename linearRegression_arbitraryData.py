import sklearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()

weights = [87, 81, 82, 92, 90, 61, 86, 66, 69, 69]
heights = [187, 174, 179, 192, 188, 160, 179, 168, 168, 174]

body_df = pd.DataFrame({'height' : heights, 'weight' : weights}) # 표 형태의 데이터를 생성
# print(body_df.head(3)) # 터미널에 데이터 3줄 출력

sns.scatterplot(data = body_df, x = 'weight', y = 'height') # data = body_df : 사용할 데이터, x축이 weight, y축이 height
# plt.show()

X = body_df[['weight']]
y = body_df[['height']]

model_lr.fit(X = X, y = y) # 학습하는 코드

# 학습 결과
print(model_lr.coef_) # 가중치(w1)
print(model_lr.intercept_) # 편향(bias, w0)

# 학습된 직선의 기울기와 절편을 꺼냄
w1 = model_lr.coef_[0][0]
w0 = model_lr.intercept_[0]

# print('y = {}x + {}'.format(w1, w0)) # 수식

# print('y = {}x + {}'.format(w1.round(2), w0.round(2))) # 소수점 둘째자리까지만 표시


# 수동으로 MSE 계산 해보기
body_df['pred'] = body_df['weight']*w1 + w0 # 예측 값 계산
body_df['error'] = body_df['height'] - body_df['pred'] # 오차 범위 계산

# print(body_df.head(3))


body_df['error^2'] = body_df['error'] * body_df['error']

# print(body_df.head(3))

body_df['error^2'].sum()/len(body_df) # MSE 최종 계산 단계
# 각 데이터 오차 제곱을 모두 더함 => 공정한 비교를 위해 데이터 수로 나눔

sns.scatterplot(data = body_df, x = 'weight', y = 'height')
sns.lineplot(data = body_df, x = 'weight', y = 'pred', color='red') # 예측한 값이 선형으로 그려져야 해서 y값은 pred로 가져온다
plt.show()




