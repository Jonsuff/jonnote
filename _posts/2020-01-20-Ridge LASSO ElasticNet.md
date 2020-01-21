---
layout: post
title:  "Ridge, LASSO, ElasticNet"
date:   2020-01-20 15:28:13
categories: Machine_Learning
---



# 10.6 회귀에 규제 적용

선형 회귀 규제는 주로 세 가지 방법으로 사용한다.

1.  **릿지 회귀(Ridge Regression)**
2. **라쏘(Least Absolute Shrinkage and Selection Operator, LASSO)**
3. **엘라스틱 넷(Elastic Net)**



### 1. 릿지 회귀

릿지 회귀는 L2 규제 모델로, 최소 제곱 비용 함수에 가중치의 제곱합을 추가한 모델이다.





비용함수 : 
$$
J(w)_{Ridge} = \sum_{i=1}^n \left(y^{(i)} - \hat{y}^{(i)}\right)^2 + \lambda \lVert w \rVert_{2}^{2}
$$




L2 규제 :
$$
\lambda \lVert w \rVert_{2}^{2} = \lambda \sum_{j=1}^m {w_j}^2
$$


하이퍼 파라미터인 $$\lambda$$를 증가시키면 규제 강도가 증가하고, 모델의 가중치 값이 감소한다. 절편 $$w_0$$는 규제하지 않는다.



### 2. 라쏘

라쏘는 L1 규제 모델로, 규제 강도에 따라 특정 가중치는 0이 되어 완전히 제외될 수 있다. 따라서 라쏘를 지도 학습의 특성 선택 기법으로 사용할 수 있다.



비용함수 :
$$
J(w)_{LASSO} = \sum_{i=1}^n \left(y^{(i)} - \hat{y}^{(i)}\right)^2 + \lambda \lVert w \rVert_1
$$




L1 규제 :
$$
\lambda \lVert w \rVert_1 = \lambda \sum_{j=1}^m \left\vert w_j \right\vert
$$





### 3. 엘라스틱 넷

엘라스틱 넷은 릿지 회귀와 라쏘의 절충안이다. 즉 L1, L2 두 가지의 규제 모두 사용할 수 있는데 이들을 혼합하는 혼합 정도는 각각 $$\lambda_1$$과 $$\lambda_2$$로 나타낸다.


$$
J(w)_{ElasticNet} = \sum_{i=1}^n \left(y^{(i)} - \hat{y}^{(i)}\right)^2 + \lambda_1 \sum_{j=1}^m {w_j}^2 + \lambda_2 \sum_{j=1}^m \left\vert w_j \right\vert
$$




### 사용법

규제 선형 모델은 모두 사이킷런에 준비되어 있다. 하이퍼파라미터 $$\lambda$$를 매개변수 $$\alpha$$로 조절하여 사용한다.

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1.0)
```

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 1.0)
```

```python
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
```

