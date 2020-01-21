---
layout: post
title:  "Linear Regression - OLS"
date:   2020-01-20 13:43:13
categories: Machine_Learning
---



# 10.3 최소 제곱 선형 회귀 모델 구현

### 최소 제곱법 (Ordinary Least Squares, OLS)

> 정확한 답을 구하지 못하는 경우, 해를 근사하여 문제를 해결할 수 있는 방법 중 하나이다. 

선형 회귀(Linear regression)의 가장 기본적인 개념은 "예측선 긋기" 이다.

예를들어 한 학생이 시험공부를 몇 시간 했느냐에 따라 성적이 달라지는 데이터셋이 있다고 가정해보자.

| 공부 시간 | 10   | 20   | 30   | 40   |
| --------- | ---- | ---- | ---- | ---- |
| 시험 성적 | 55   | 80   | 70   | 95   |

위와 같은 데이터를 그래프로 나타내보면 다음과 같다. 

~~공부를 많이 할수록 성적이 오르면 좋겠지만 어림도 없지.~~

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/10장/예시표01.png)

선형 회귀에서의 예측선은 1차함수의 선이여야 한다.
$$
y = ax+b
$$
 따라서 하나의 직선으로 저 데이터를 표현해야 하지만 정확한 정답은 존재하지 않는다. 그렇다면 최대한 근사한 직선을 그어야 하는데 이 때 최소 제곱법을사용한다.

우선 최소 제곱법의 공식은 직선의 기울기 a와 y절편 b에 대하여 다음과 같다.
$$
a = {\sum_{i=1}^n (x-mean(x))(y-mean(y)) \over \sum_{i=1}^n (x-mean(x))^2}
$$

$$
b = mean(y)-(mean(x)*a)
$$



이를 통해 최적의 값을 노란 점으로 그려보면 아래와 같다.

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/10장/예시표02.png)



### 경사 하강법으로 회귀 모델의 파라미터 구하기

OLS에서 사용할 비용함수는 아달린의 비용함수와 같다. 이는 제곱 오차합(SSE)로 공식은 다음과 같다.
$$
J(w) = {1 \over 2} \sum_{i=1}^n \left(y^{(i)}- \hat{y}^{(i)}\right)^2
$$


여기서 $$\hat{y}$$는 예측값으로 $$\hat{y}=w^Tx$$ 이다. 근본적으로 OLS회귀는 단위 계단 함수가 빠진 아달린으로 해석할 수 있다. 다시말해 -1과 1로 결과가 나오는 대신 연속적인 값을 얻어낸다. 예제 학습을 위해 LinearRegressionGD 클래스를 아달린을 참고하여 만든다(이때 아달린에서 단위 계단함수 부분을 제거한다).



```python
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)
```



주택 데이터셋에서 RM(방 개수) 변수를 특성으로 사용하여 MED(주택 가격)을 예측하는 모델을 훈련해보자.

우선 경사하강법 알고리즘이 잘 수렴하도록 StandardScaler를 사용하여 특성을 표준화 전처리한다.

```python
X = df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)
```

y_std를 계산할 때 np.newaxis와 flatten을 사용한다. 사이킷런에서 제공하는 대부분의 변환기는 데이터가 2차원 배열로 저장되어 있다고 가정한다. 즉 원래 y는 1차원 행렬 데이터이기 때문에 np.newaxis를 통해 2차원 데이터로 바꿔준 후 학습을 진행하고, 다시 flatten()을 사용하여 1차원 데이터로 변환시킨 것이다.

훈련을 반복하는 에포크에 따라 제곱 오차합이 비용함수의 최솟값으로 수렴하는지를 확인해본다.

```python
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()
```

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/10장/SSE by epoch.png)

결과에서 볼 수 있듯이 5번 째 에포크에서 최솟값으로 수렴하였다.

이제 훈련 데이터와 선형 회귀 모델이 어느정도 일치하는지를 직관적으로 확인하기 위해 그래프로 확인해 본다.

```python
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 
    
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

plt.show()
```

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/10장/선형회귀 결과비교.png)



### 사이킷런으로 회귀 모델의 가중치 추정

실전에서는 위에서 구현한 선형 회귀 모델보다 더 효율적이고 경량화된 구현이 필요하다.

사이킷런의 많은 회귀 추정기는 고수준의 최적화 알고리즘을 사용한다(ex- LinearRegression에서 제공하는 scipy.linalg.lstsq). 또한 특정 애플리케이션에서 필요한 표준화되지 않은 변수와도 잘 동작하는 최적화 기법을 사용한다.

LinearRegression 클래스를 사용해보자.

```python
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

plt.show()
```

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/10장/선형회귀 결과비교(비표준).png)