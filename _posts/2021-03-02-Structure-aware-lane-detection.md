---
layout: post
title:  "Ultra Fast Strucrue-aware Deep Lane"
date:   2021-03-02 15:11:00
categories: Deep_Learning
---

# Ultra Fast Structure-aware Deep Lane Detection

| 제목 | Ultra Fast Structure-aware Deep Lane Detection |
| ---- | :--------------------------------------------: |
| 저자 |       Zequn Qin, Huanyu Wang, and Xi Li        |
| 출판 |                   arXiv 2020                   |

> github : https://github.com/cfzd/Ultra-Fast-Lane-Detection
>
> 파이토치 기반 코드, auxiliary segmentation step이 존재하면 일반버전, 없으면 light weight버전



## Abstract

최근 딥러닝을 이용한 차선인식 문제는 픽셀 segmentation 문제로 접근하였다. 하지만 이는 막대한 연산량때문에 속도가 느리다는 단점이 있다. 본 논문의 저자는 차선인식 문제를 횡방향 기반 선택문제로 접근하여 segmentation task보다 더 빠르고,어려운 시나리오에서 보다 효과적인 알고리즘을 제안했다.

본 논문 저자의 github에는 성능을 생각한 일반 버전과 속도를 중시한 light weight버전이 있는데, lilght weight버전은 일반버전과 같은 해상도를 사용하지만 초당 300프레임 이상의 속도를 보이며 기존 차선인식 문제의 SOTA를 달성한 모델들보다 4배가량 빠르다고 한다.



## Introduction

일반적인 차선인식 문제의 해결 방법으로는, 첫 째로 전통 이미지 프로세싱을 통한 방법이 있고, 둘 째로 딥러닝을 사용하여 segmentation을 하는 방법이 있다. 최근들어 GPU성능이 좋아지면서 딥러닝을 사용하는 방법이 좋은 성과를 보이는 추세이다. 하지만 segmentation 방법에는 아직 짚고 넘어가야할 문제가 있다.

가장 근본적인 문제는 자율주행 상황에서 차선인식은 아주 무겁게 돌아간다는 것이다. 현대 자율주행 기술에는 차량의 앞, 뒤, 양옆 등 다수의 카메라를 장착하고, 이를통해 최대한 다양한 시점의 이미지를 얻어내는 방법들이 빈번하게 사용된다. 사용하는 카메라가 많을수록 한 frame에 대한 이미지가 많아지고, 이들을 빠르게 처리하기 위해서는 더 빠른 pipeline을 지닌 학습모델을 구현해야 한다. 

두 번째 큰 문제는 생각보다 주행시 차선이 보이지 않는 경우가 많다는 것이다. 

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/1.png)

위의 예시처럼 실제로 도로에서는 다른 객체에 가려지거나, 너무 어두워서 차선이 카메라에 잘 보이지 않는 경우가 많다. 이런 상황에서는 보다 고차원적인 분석이 필요하고, deep segmentation이 해답을 제시했다. 하지만 이는 모두 pixel wise 연산이므로, 연산 cost가 크다는것이 문제다.

위에서 언급한 모든 단점들을 해결하고자, 본 논문에서는 아주 빠른 속도와 no-visual-clue(시각적인 정보 없이)에서도 사용 가능한 것을 목표로하는 차선인식 공식을 제안한다. 이 방법을 바탕으로 이전 상황의 차선 정보를 현재 상황에 적용하여 학습하는 structural loss를 정의한다. 이 모든 방법의 가장 keypoint는 이미 횡방향으로 정의되어있는 차선의 위치들 중 현재 차선의 위치를 예측하는 것이다. 따라서 모든 픽셀로 연산하는것이 아니기때문에 cost가 적다.

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/2.png)







## Network

본 논문의 네트워크 구조는 다음과 같다.

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/4.png)





## Training Method

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/3.png)

위와 같은 parameter를 사용하는 본 논문의 핵심 공식과 lane structural loss에 대해 알아본다.



### 3.1 Formulation for lane detection

빠르고 정확한 차선인식을 위해 글로벌 이미지를 이용한 횡방향 기반 위치선택 방법을 소개한다. 다시 말을 정리해보면 이 작업은 입력 이미지에 대해 미리 지정된 차선의 위치후보들 중에서 어떤 후보가 가장 적합한지를 선택하는 작업이다. 이때 각 row에는 지정된 anchor가 존재한다. 방법은 다음과 같은 순서로 진행된다.

1. 입력 이미지에서 최대로 검출될 수 있는 차선의 수가 C, 차선이 위치할 수 있는 후보가 되는 row anchor의 수가 h, 그리고 row anchor를 일정한 간격으로 나눈 cell의 수를 w라고 한다.

2. 만약 $X$가 이미지 feature이고, $f^{ij}$가 차선의 cell위치를 예측하는 classifier라고 한다면(i번째 차선의 j번째 row) 차선에 대한 예측은 다음 식과 같다.
   
   $$\\P_{i, j, :} = f^{ij}(X), s.t. i \in[1, C], j \in [1, h]\\$$
   
> $P_{i,j,:}$ : (w+1)-차원의 벡터로, i번째 차선의 j번째 row의 cell에 대한 각각의 확률들을 모아놓은 것이다. P의 차원이 w+1인 이유는 그 row에 더이상 차선이 존재하지 않는 경우를 대비하였기 때문이다. 즉 마지막 차원은 해당 row에 차선이 없는 경우에 속한다.
   >
   > s.t. : such that(이를 만족시키는)

3. 만약 $T_{i, j,:}$가 gt차선의 실제 위치cell이 원핫인코딩 된 상태의 데이터라면, classifier는 다음과 같이 정의할 수 있다.
   
   $$\\L_{cls} = \sum^{C}_{i = 1} \sum^h_{j = 1}L_{CE}(P_{i, j, :}, T_{i, j, :})\\$$

segmentation과 연산량을 단순히 비교해보면, 해상도가 (H, W)인 이미지를 segmentation하는 경우 파라미터가 $H \times W \times (C+1)$ 이지만, 본 논문의 방법은 이미지 해상도를 일정 cell로 나누어 grid화 했기 때문에 $grid(h) << H$, $grid(w) << W$인 그리드값을 사용하여 $C \times h \times (w+1)$로 훨씬 적은 수임을 짐작할 수 있다.



### 3.2 Handling with no-visual-clue problem

차선이 보이지 않는 상황에서는 어떤 방법으로 차선인식이 진행되는지를 알아본다. 다른 객체에 의해 차선이 가려지더라도, 우리는 주변의 다른 차들의 위치와 다른 차선, 그리고 도로의 모양등으로 차선의 위치를 예측할 수 있다. 본 논문의 방법은 grid로 뭉쳐진 feature를 row의 영역에서 바라보기때문에 주변 상황에 맞추어 차선을 인식할 수 있다. 보다 자세한 내용은 3.3절의 structural loss에서 설명한다.



### 3.3 Lane structural loss

차선이 위치하는 cell을 예측하는 classification loss 이외에도, 본 논문의 저자는 두가지의 loss함수를 제안한다. 

1. 첫 번째 방법은 만약 차선이 이어져있다면, 서로 이웃하는 row의 차선이 위치하는 cell은 서로 가까이 위치한다는 개념에서 출발한다. 차선의 위치는 classification을 통해 예측되고, 이는 각각의 row에서 진행되므로 모든 row에 대한 차선예측을 합치면 이는 classification vector가 되고, 이를 연속적 확률분포로 여겨 다음과 같은 공식으로 정의한다.
   
   $$\\L_{sim} = \sum^{C}_{i=1} \sum^{h-1}_{j=1}||P_{i, j, :} - P_{i, j+1, :}||_1\\$$
   
> $||a||_1$ : L1 norm

2. 두 번째 방법은 차선의 모양에 중점을 맞추어 생각한다(**second-order difference equation**). 일반적으로 대부분의 차선은 직선이고, 실제로 곡선 차선이더라도 운전자(카메라)가 바라보는 시점에 의해 직선처럼 보인다. 이 작업의 결과물은 차선이 직선이냐 곡선이냐를 판단하는데, 0에 가까울수록 직선임을 의미한다.

   차선의 모양을 정하기 위해서는 각각의 row들의 모든 차선cell들이 계산되어야 한다. 이를 위해 각 cell의 classification 결과들의 maximum peak를 이용한다.
   
   $$\\Loc_{i, j} = \underset{k}{argmax} P_{i,j,k}, s.t. k\in[1,w]\\$$
   
   이때 k는 location index이고, 차선이 없는 배경에 대한 정보는 사용하지 않을것이므로 (w+1)대신 w의 범위내에서 결정한다.

   하지만 argmax함수는 row들 전체에 대해서 연속되는 값이 아니기때문에 서로의 상관관계를 정의하기 어렵다(미분불가). 따라서 다음과 같이 softmax함수를 사용하여 확률들끼리의 또다른 확률분포를 만들어낸다.
   
   $$\\Prob_{i, j, :}=softmax(P_{i, j, 1:w})\\$$
   
   이를 argmax함수 대신 사용한다면 다음과 같은 수식으로 정리가 된다.
   
   $$\\Loc_{i, j} = \sum^{w}_{k=1} k \cdot Prob_{i, j, k}\\$$
   
   위의 식을 이용하여 서로 이웃한 3개의 row들끼리의 상관관계를 구하면 다음과 같이 loss가 정의된다.
   
   $$\\L_{shp} = \sum^C_{i=1} \sum^{h-2}_{j=1}||(Loc_{i, j}- Loc_{i, j+1}) - (Loc_{i, j+1} - Loc_{i, j+2})||_1\\$$
   
   이때 3개의 row를 사용하는 이유는 두개를 사용할 경우 대부분은 0이 나오지 않는다는 점 때문이라고 한다.
   
   lane structural loss는 계수 $\lambda$를 추가적으로 사용하여 다음과 같이 정의한다.
   
   $$\\L_{str} = L_{sim} + \lambda L_{shp}\\$$


### 3.4 학습

학습에 사용되는 최종 loss는 다음과 같다.

$$\\L_{total} = L_{cls} + \alpha L_{str} + \beta L_{seg}\\$$

> $L_{seg}$ : segmentation loss - 본 논문에서는 cross entropy를 사용했다. 이는 픽셀별 cross-entropy연산을 진행한 후 그 모든 값들을 평균내는 것이다.



## Result

사용 데이터셋 : TuSimple

### evaluation

성능검증에 사용되는 수식은 다음과 같이 정의한다.

$$\\accuracy = {\Sigma_{clip} C_{clip} \over \Sigma_{clip}S_{clip}}\\$$

> $C_{clip}$: lane point 예측에 성공한것(TP)
>
> $S_{clip}$: GT total

C의 값을 얻어내는 기준은 다음과 같다.

- 각각의 lane point를 중심으로하고, 세로길이는 row의 높이, 가로길이는 30픽셀인 박스를 예측과 gt에대해 지정한 후 둘 사이의 IOU를 계산한다. 
- Threshold를 0.5로 지정하고 0.5보다 큰것은 TP로 사용한다.



### eval / classification accuracy

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/5.png)



### 다른 모델과 비교

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/6.png)

> Multiple: 가장 느린 SCNN모델과의 속도 비교수치



### 시나리오별 정확도

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/7.png)



### 검출 결과 영상

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/lane/8.png)