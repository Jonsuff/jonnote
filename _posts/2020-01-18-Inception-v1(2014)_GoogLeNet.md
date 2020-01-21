---
layout: post
title:  "Inception-v1(GoogLeNet)"
date:   2020-01-18 17:03:13
categories: Deep_Learning
---



# Inception-v1(2014)

- *Going deeper with convolutions* - Chiristian Szegedy, Wei Liu etc...

#### 특징

- 구글이 알고리즘 개발에 참여하여 *GoogLeNet* 이라는 이름이 생겼다.

- VGG-19를 이기고 2014 ILSVRC에서 우승을 차지했다.

- 22개의 layer로 구성되어 있으며 AlexNet보다 파라미터 수가 12배 적다. 적은 수의 파라미터를 사용하였기 때문에 컴퓨터의 메모리 사용이나 전력 사용이 크게 감소하였다. 

- 모델의 성능은 심층구조 그 자체나 더 큰 모델에서 크게 증가되는 것이 아니라 모델의 심층구조와 classical computer vision의 시너지에서 증가한다. (ex) R-CNN 알고리즘

- > the models were designed to keep a computational budget of 1.5 billion multiply-adds at inference time,
  >
  > 이 문장이 무슨 뜻일까? 이러한 이유 때문에 이 논문의 모델이 순수한 학문적 결과들을 도출해 내는 것 뿐만 아니라 현실에서도 사용이 가능할 수 있다고 한다.
  >
  > 혹시  파라미터 수가 줄어서 필요한 전력량이나 컴퓨터의 메모리가 줄어 하드웨어적 한계가 줄었기 때문에 연구실에서나 쓸법한 슈퍼컴퓨터 없이도 이 모델을 사용할 수 있다는 것인가?



#### GoogLeNet 구조

- ![GoogLeNet Architecture](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/GoogLeNet_Architecture.jpg)



#### 1x1 convolution

GoogLeNet 구조에서 보이듯이 곳곳에 1x1 convolution 연산이 등장한다. 이 모델에서 1x1 컨볼루션은 **특성 맵의 개수를 줄이는 목적으로 사용된다**(Fewer feature maps $$\rightarrow$$ compute reductions)

- Ex) 480 채널의 14x14 사이즈의 feature map(14x14x480)이 있다고 가정한다.

  1. 이것을 바로 48개의 5x5x480 filter로 convolution(zero padding = 2, stride =1)  해주면 48장의 14x14 feature map(14x14x48)이 생성된다.

     필요한 연산횟수 : 
     $$
     (14*14*48)*(5*5*480)\cong 112.9M\\
     $$

  2. 먼저 16개의 1x1x480의 filter로 convolution(zero padding = 2, stride =1)  한다. 결과적으로 16장의 14x14 feature map(14x14x16)이 생성된다. 5x5 filter로 convolution 하기 전에 feature map의 수가 확연하게 줄어든 것을 볼 수 있다.

     이를 5x5x16의 filter로 convolution하면 48개의 14x14 feature map(14x14x48)이 생성된다.

     필요한 연산횟수 : 
     $$
     (14*14*16)*(1*1*480)+(14*14*48)*(5*5*16) \cong 5.3M\\
     $$

  결과를 보면 1번 상황에서는 112.9M번의 연산이, 2번상황에서는 5.3M번의 연산이 진행되었다. 즉 2번 방법으로 연산을 진행하면 연산횟수가 확연히 줄어들게 되고, 연산량이 줄어들면 Network의 depth를 더 늘릴 수 있게 된다.



#### Inception module

GoogLeNet은 총 9개의 inception module을 포함하고 있다. 한 모듈을 확대해서 살펴보면 아래의 그림과 같다.

![inception module](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/GoogLeNet_inception module.png)

(a)와 (b)의 차이는 1x1 convolution이 연산에 포함이 되었는지 되지 않았는지이다. GoogLeNet에서 실제로 사용된 모듈은 (b) 이다. 1x1 convolution은 앞에서 설명했듯이 연산 횟수를 줄여주는 역할을 할 뿐이므로 모듈 구조의 해석은 (a)로 해보자.

(a) 모델을 살펴보면, 이전 층에서 생성된 feature map을 1x1, 3x3, 5x5 convolution을 각각 진행하고, 3x3 Max pooling하여 결과를 얻은 특성맵들을 모두 함께 쌓는다. AlexNet, VGGNet 등 이전 CNN 모델들은 한 층에서 동일한 사이즈의 filter를 이용하여 convolution 해주었는데 GoogLeNet에서는 확연히 다른점을 보여준다.

이전 CNN 모델들과 확연히 다른 (a) 모델은 더 다양한 종류의 특성이 도출되도록 한다. 즉 1x1 convolution을 통해 줄어든 연산량의 맹점을 더 다양한 특성을 이용한다는 것으로 상쇄하여 성능은 높아지고 연산량은 줄어든 결과를 도출해 내었다.



#### Global average pooling

이전 모델들은 분류의 후반부에 FC(fully connected layer)를 사용하여 이미지를 분류해 냈다. 하지만 GoogLeNet은 FC방식 대신 **Global average pooling**이라는 방식을 사용한다.

![Global average pooling](https://raw.githubusercontent.com/Jonsuff/MLstudy/master/images/GoogLeNet_Global average pooling.png)

Global average pooling은 이전 층에서 산출된 feature map들을 각각 평균낸 것을 이어 1차원 벡터를 만들어주는 것이다(1차원 벡터를 만들어야 최종적으로 이미지를 분류하기 위한 soft-max layer를 연결할 수 있기 때문이다).

예를 들면 이전 층에서 1024장의 7x7 feature map(7x7x1024)이 생성되었다면, 각각의 7x7 feature map을 평균내주어 얻은 1024개의 데이터값을 하나의 벡터로 연결해 주는 방식이다.

이 방식을 통해 얻을 수 있는 장점은 weight의 수를 줄일 수 있다는 점이다. FC방식과 Global average pooling방식의 weight 개수를 계산해보면 

1. FC : $$7*7*1024*1024 \cong 51.3M\\$$

   7x7x1024의 feature map을 1024개의 7x7x1024 filter convolution을 진행했기 때문에 convolution마다 1개의 weight가 발생한다.

2. Global average pooling : 0

   각각의 feature map의 평균치를 1x1x1024 벡터에 연결해 주었기 때문에 weight가 발생하지 않는다.

즉 파라미터 수를 줄일 수 있다.



#### Auxiliary classifier

Network의 depth가 커질수록 vanishing gradient 문제를 피하기 어렵다. 이 문제를 극복하기 위해서 Network 중간에 두 개의 보조 분류기(Auxiliary classifier)를 달아주었다.

> Vanishing gradient
>
> weight를 훈련하는 과정에서 back propagation을 사용하는데, 이 과정에서 weight를 update하는데 사용되는 gradient가 점점 작아져서 0이 되어버리는 문제이다. 이는 weight 훈련이 정상적으로 이루어지지 않는 결과를 초래한다.

Auxiliary classifier의 구성을 살펴보면 아래와 같은 순서로 진행된다.

1. 5x5 Average pooling(stride = 3)
2. 1x1x128 filter convolution
3. FC layer(1024)
4. FC layer(1000)
5. soft-max

이 보조 분류기들은 훈련에서만 사용되고 실제로 모델을 사용할 때는 제거해 준다.