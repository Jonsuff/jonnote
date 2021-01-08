---
layout: post
title:  "Convolution Neural Network(CNN)"
date:   2020-01-13 15:43:13
categories: Deep_Learning
---



# Convolution Neural Network(CNN)

### CNN

- 입력된 이미지 데이터를 Convolution layer를 통과하면서 Filter를 통해 이미지의 특징을 추출한다.

- Convolution(합성곱) filter : 

  1. 특정한 행렬로 정해진 convolution filter의 크기에 맞춰 입력 데이터에의 좌측 상단부터 부분 행렬을 추출한다.
  2. 추출된 부분 행렬과 필터를 각각의 원소끼리 곱한 후 원소들을 모두 더한다.
  3. 입력 데이터에서 우측으로 한칸 이동한 후 새로운 부분 행렬을 추출한다. 이 때 우측으로 더이상 이동할 수 없으면 1열로 돌아와 아래로 한칸 이동한다.
  4. 2번을 반복한다.
  5. 2번 행위에서 나온 값들로 새로운 행렬을 만든다.

  (ex)
  $$
  \mathrm{Input(A)}:\begin{bmatrix}2&4&6&1&7 \\ 2&6&0&7&0 \\ 5&0&0&8&9\\7&0&1&1&4\\6&7&2&0&1\end{bmatrix}
  $$
  
- 입력된 데이터 A가 위와 같은 5x5 행렬이고,
  
$$
  \mathrm{Filter(b)}:\begin{bmatrix}1&0&1\\0&1&0\\1&1&0\end{bmatrix}
  \\
  $$
  
- 합성곱 필터 B가 위와 같은 3x3 행렬이면 A의 좌측 상단 행렬인 sub(A)와 B를 원소끼리 곱한다.
  
$$
  \mathrm{Sub(A)}:\begin{bmatrix}2&4&6\\2&6&0\\5&0&0\end{bmatrix}
  \\
  $$
  
$$
  \mathrm{Sub(A)} \times \mathrm{Filter(B)} = \begin{bmatrix}2&4&6\\2&6&0\\5&0&0\end{bmatrix} * \begin{bmatrix}1&0&1\\0&1&0\\1&1&0\end{bmatrix} = \begin{bmatrix}2&0&6\\0&6&0\\5&0&0 \end{bmatrix}
  \\
  $$
  
- 곱하진 수를 모두 더한다.
  
$$
  2+6+0+6+0+5+0+0 = 19
  \\
  $$
  
- 입력 데이터에서 한칸 우측으로 이동하여 다른 부분행렬을 추출한다.
  
$$
  \mathrm{Sub(A_2)} =\begin{bmatrix}4&6&4\\6&0&7\\0&0&8\end{bmatrix}
  \\
  $$
  
- 위의 연산을 반복한다.
    $$
    \begin{bmatrix}4&6&4\\6&0&7\\0&0&8\end{bmatrix}*\begin{bmatrix}1&0&1\\0&1&0\\1&1&0\end{bmatrix} =\begin{bmatrix}4&0&4\\0&0&0\\0&0&0\end{bmatrix} \Rightarrow 4+4 = 8
    \\
  $$
    
  - 연산을 반복하여 나온 숫자들로 새로운 행렬을 만든다.
  
- Filter를 통과한 결과는 입력데이터에 비해 축소된 것을 알 수 있다. 이 때 출력되는 행렬의 크기는 다음 식과 같다.
  $$
  \mathrm{out} = {[{\mathrm{size(Input)-size(Filter)}}] \over \mathrm{stride}}+1
  \\
  $$
  (ex) 5x5의 입력데이터와 3x3의 필터를 거치면 결과는 5-3+1 = 3이 되어 결과는 3x3행렬이 된다.



### Pooling layer

- Pooling layer는 이미지를 압축하는데 의미가 있다.

- Pooling layer는 convolution layer를 지난 후 사용된다.

- 보통 Max pooling과 Average pooling을 사용한다.(Max pooling이 더 많이 쓰인다.)

- Max pooling : 정해진 크기의 부분 행렬중에서 최댓값을 추출한다.

  (ex)

  - 다음 4x4 크기의 행렬을 예시로 사용한다.
    $$
    \mathrm{example} : \begin{bmatrix}8&10&7&5\\16&12&20&19\\7&8&9&13\\10&15&15&14\end{bmatrix}\\
    $$
    이 예시에서 2x2 Max pooling (Stride = 2)를 적용하면 다음과 같다.
    $$
    \mathrm{result} : \begin{bmatrix}16&20\\15&15\end{bmatrix}\\
    $$
    
  - Max pooling의 결과로 4x4 행렬이 2x2 행렬로 축소된 것을 확인할 수 있다.

- Average pooling은 평균값을 반환해 준다.



### Fully Connected Layers(FC)

- Pooling layer를 지난 후 이미지를 분류하는 작업.
- 윗단계까지 추출된 모든 특성들을 통해 어떤 이미지에 가장 가까운지를 판단하여 해당 클래스 레이블로 분류한다.







