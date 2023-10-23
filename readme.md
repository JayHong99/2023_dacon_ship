# Server Information


# Objective
- 1) CI_HOUR를 예측하고, MAE로 평가
- 2) 데이터를 확인해보니, 25%의 데이터는 0으로 이루어져있고, 50%가 넘어가면서부터 데이터의 Outlier가 매우 커지고 있음
- 3) 0~50%까지의 데이터를 예측하는 모델을 만들고, 50% 이상의 데이터는 0으로 예측하는 모델을 만들어보자

```python
print(train_df['CI_HOUR'].describe())
print(np.quantile(train_df['CI_HOUR'], 0.90))

"""
count    367441.000000
mean         61.877118
std         170.575224
min           0.000000
25%           0.000000
50%           7.949444
75%          49.153333
max        2159.130556
Name: CI_HOUR, dtype: float64
151.9941667
"""
```


# Modeling
Self-supervised Contrastive Learning for Feature Extraction by SCARF (https://github.com/yfeng95/SCARF)

--> simple 2-layer MLP + Machine Learning