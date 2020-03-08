---
layout: post
title:  "About CNN and Pooling"
date:   2020-03-08 19:13:13
categories: Machine_Learning
---



# CH.15 심층 합성곱 신경망으로 이미지 분류

- **합성곱 신경망**(Convolutional Neural Network, CNN)



## 15.1 합성곱 신경망의 구성 요소

CNN은 **뇌의 시각 피질이 물체를 인식할 때 동작하는 방식**에서 영감을 얻은 모델이다(CNN이 뇌와 같은 방법으로 작동한다는 의미는 아니다).

CNN은 이미지 분류 작업에서 탁월한 성능을 내어 주목받은 기술이다.



### 15.1.1 CNN과 특성 계층 학습

이미지를 분류하는데 핵심이 되는 특징을 올바르게 추출하는 것은 모든 머신러닝 알고리즘의 성능에서 중요한 요소이다. 머신러닝에서 특징을 추출해내는 방법은 크게 두가지가 있다.

1. 도메인 전문가가 만든 특성에 의존
2. 컴퓨터를 사용한 특성 추출 기법을 이용

신경망은 원본 데이터에서 작업에 가장 유용한 특성을 자동으로 학습할 수 있기 때문에 신경망을 특성 추출 엔진으로 생각하기도 한다. (ex) 입력에 가까운 층은 저수준 특성을 추출할 수 있다.

고수준 특성을 추출하기 위해서는 다층 신경망을 사용한다.

- 심층 합성곱 신경망 : 

  각 층별로 저수준 특성을 연결하여 고수준 특성을 만듦으로써 특성 계층을 구성.

- 저수준 특성 : 

  에지(edge), 동그라미, 네모 등 도형이나 일정한 패턴의 선

- 고수준 특성 : 

  저수준 특성들이 모여서 건물, 자동차, 강아지 등 의미있는 이미지를 형성한 것.

- 특성 맵(feature map) : 

  입력된 이미지에서 만드는 특성 맵은 전체 이미지에서 작은 일부분을 선택해 그 부분 내에서의 특성을 파악하여 한 픽셀로 표현한 것.

  다음과 같은 이미지로 예를 들 수 있다.

  ![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/ch15_feature_map_fixed.png)

- CNN은 다음과 같은 이유로 이미지 관련 작업을 잘 수행한다.

  - 희소 연결 : 

    특성 맵에 있는 하나의 원소는 작은 픽셀 패치 하나에만 연결된다(퍼셉트론처럼 모든 입력 이미지에 연결되는 것과 매우 다르다).

  - 파라미터 공유 : 

    동일한 가중치가 입력 이미지의 모든 패치에 사용된다.

  이 두 아이디어의 결과로 네트워크의 파라미터 개수가 감소하고 중요한 특징은 더 잘 잡아낸다.

- 일반적으로 CNN은 여러개의 **합성곱 층(convolution)**과 풀링(Pooling) 이라고도 불리는 **서브샘플링 층(subsampling)**으로 이루어져 있다.

  마지막에는 하나 이상의 완전히 연결된(Fully connected) 층이 붙는다. 이 층은 모든 입력이 모든 출력에 가중치가 곱하여 연결되어있는 다층 퍼셉트론이다.

- **풀링 층(=서브샘플링 층)**은 학습되는 파라미터가 없다. 

  $$\Rightarrow$$ 가중치나 절편이 존재하지 않는다.



### 15.1.2 이산 합성곱 수행

CNN의 기본 연산인 합성곱은 **이산 합성곱**(discrete convolution) 방법이다.

- 이산 합성곱의 수학적 정의
  $$
  \mathbf{y} = \mathbf{x} * \mathbf{w} \rarr \mathbf{y[\mathrm{i}]} = \sum^{+\infin}_{k = -\infin} \mathbf{x[\mathrm{i-k}]} \mathbf{w[\mathrm{k}]}
  $$
  [] : 벡터 원소의 인덱스

  i = 출력 벡터 **y**의 각 원소에 대응

  - 합연산에서 k의 범위가 $$+\infin$$ 에서 $$-\infin$$ 까지인것은 바르지 않다. 입력이 유한한 상황에서 k의 범위가 무한대로 뻗어나가면 출력또한 유효한 입력이 아닌 부분이 0으로 채워진 무한한 크기의 벡터가 되기 때문이다. 이를 해결하기 위해 **패딩**(padding) 작업을 해준다.

  

- **제로 패딩**(zero padding) : 

  유한한 개수의 0를 추가하여 **x**의 크기를 결정한다.

  만약 원본 입력 **x**와 필터 **w**가 각각 n, m개의 원소를 가지고 m $$\le$$ n 이고, 패딩된 벡터 $$\mathbf{x}^p$$의 크기는 n+2p 이다. 따라서 이산 합성곱 공식은 다음과 같다.
  $$
  \mathbf{y} = \mathbf{x} * \mathbf{w} \rarr \mathbf{y[\mathrm{i}]} = \sum^{k = m-1}_{k = 0} \mathbf{x^{\mathrm{p}}[\mathrm{i+m-k}]} \mathbf{w[\mathrm{k}]}
  $$
  

- 패딩의 종류

  실전에서 크게 사용되는 패딩은 다음과 같다.

  - 풀 패딩 : 

    패딩 파라미터 p를 p = m-1로 설정한다. 이는 출력 크기를 증가시키기 때문에 CNN에서는 거의 사용하지 않는다.

  - 세임 패딩 : 

    출력 크기가 입력 벡터 **x**와 같아야 할 때 사용한다. 패딩 파라미터 p는 입력과 출력의 크기를 맞출수 있도록 결정된다.

    실전에서 가장 많이 쓰이는 패딩이며 세임 패딩으로 너비와 높이를 유지하고 풀링에서 크기를 감소시킨다.

  - 밸리드 패딩 : 

    p = 0인 경우이다. 즉 패딩이 적용되지 않은 경우이다.

- 합성곱 출력 크기 계산

  합성곱 출력의 크기는 다음과 같은 수식에 의해 계산된다.
  $$
  O = \left[{n+2p-m} \over s \right]+1
  $$
  n = 입력 벡터 크기

  p = 패딩

  m = 필터(커널) 크기

  s = 스트라이드(밀어내는 칸 수)

  (단 []속의 연산은 소숫점을 버림하는 연산이다.)

- 1차원 이산 합성곱

  만약 **x** = (3,2,1,7,1,2,5,4)이고, **w** = (1/2, 3/4, 1, 1/4)라면,

  ![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/ch15_conv.png)

  

  1차원 합성곱 예제 코딩

  ```python
  import numpy as np
  def conv1d(x, w, p=0, s=1):
      w_rot = np.array(w[::-1])
      x_padded = np.array(x)
      if p > 0:
          zero_pad = np.zeros(shape=p)
          x_padded = np.concatenate([zero_pad, x_padded, 
                                     zero_pad])
      res = []
      for i in range(0, int(len(x)/s),s):
          res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*
                            w_rot))
      return np.array(res)    
  ```

- 2D 이산 합성곱 : 

  앞선 1차원 이산 합성곱을 2차원으로 확장하여 이미지 데이터를 합성곱으로 계산한다.

  2차원 합성곱 예제 코딩

  ```python
  import numpy as np
  import scipy.signal
  
  
  def conv2d(X, W, p=(0,0), s=(1,1)):
      W_rot = np.array(W)[::-1,::-1]
      X_orig = np.array(X)
      ## 이미지의 상하좌우 끝에 제로 패딩
      n1 = X_orig.shape[0] + 2*p[0]
      n2 = X_orig.shape[1] + 2*p[1]
      X_padded = np.zeros(shape=(n1,n2))
      X_padded[p[0]:p[0] + X_orig.shape[0], 
               p[1]:p[1] + X_orig.shape[1]] = X_orig
  
      res = []
      for i in range(0, int((X_padded.shape[0] - 
                             W_rot.shape[0])/s[0])+1, s[0]):
          res.append([])
          for j in range(0, int((X_padded.shape[1] - 
                                 W_rot.shape[1])/s[1])+1, s[1]):
              X_sub = X_padded[i:i+W_rot.shape[0], 
                               j:j+W_rot.shape[1]]
              res[-1].append(np.sum(X_sub * W_rot))
      return(np.array(res))
  ```

  위의 예제는 개념 이해용으로 작성되어서 메모리 효율이 매우 비효율적이다.

  연산을 효율적으로 수행하는 **위노그라드 최솟값 필터링**(Winograd's Minimal Filtering)같은 알고리즘이 개발되었다.

  > Fast Algorithms for Convolutional Neural Networks, Andrew Lavin and Scott Gray, 2015
  >
  > https://arxiv.org/abs/1509.09308



### 15.1.3 서브샘플링

서브샘플링은 전형적인 두 종류의 풀링 연산으로 CNN에 적용된다.

- 최대 풀링(Max-pooling) : 영역에서 최댓값을 뽑아냄
- 평균 풀링(Mean-pooling / average-pooling) : 영역의 평균을 뽑아냄

![](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/ch15_pooling.png)

풀링은 특성의 크기를 감소시켜 계산 효율성을 높이고, 과대적합을 감소시킨다.

