---
layout: post
title:  "Python Module and Package"
date:   2019-09-09 09:00:13
categories: 2019-2-robotics
---



## 모듈과 패키지는 무엇?

지금까지 우리는 하나의 파일에서만 프로그램을 실행해왔지만 실제로 개발을 할 때는 다수의 파이썬 파일을 만들어 유기적으로 연결해야 할 뿐만 아니라 남이 만든 코드도 적극적으로 활용해야 한다. 어떤 파이썬 파일에서 `import` 키워드를 통해 외부 파이썬 파일의 함수, 변수, 클래스를 가져다 쓸 수 있다. 외부 코드는 기능별로 함수나 클래스 등을 묶어서 파일에 저장하는데 코드 규모가 작으면 하나의 파일에 담을 수도 있고 규모가 크면 여러 파일에 구현한 다음 하나의 폴더에 담을 수도 있다. 이때 각각의 `.py` 파일을 **모듈(Module)**이라 하고 여러 모듈을 묶은 폴더를 **패키지(Package)**라고 한다. 파이썬 오픈소스는 패키지 단위로 설치가 되기 때문에 패키지라는 용어를 자주 사용한다. 



## 모듈 만들기

모듈이란 쉽게 말해 파이썬 파일(`.py`)이다. 어떤 파이썬 파일에 함수와 변수를 만들고 그것들을 다른 파이썬 파일에서 가져다 쓸 수 있다. 실습을 위해 프로젝트 폴더에 `package`라는 폴더를 만들고 그 아래 `list_ops.py`라는 리스트 연산 모듈을 만들어보자.

```python
# list_ops.py
def add(foo, bar):
    out = []
    for f, b in zip(foo, bar):
        out.append(f + b)
    return out

def subtract(foo, bar):
    out = []
    for f, b in zip(foo, bar):
        out.append(f - b)
    return out

def multiply(foo, bar):
    out = []
    for f, b in zip(foo, bar):
        out.append(f * b)
    return out

def divide(foo, bar):
    out = []
    for f, b in zip(foo, bar):
        out.append(f / b)
    return out

spam = [51, 23]
ham = [34, 67]

if __name__ == '__main__':
    eggs = add(spam, ham)
```

리스트 원소에 대한 사칙연산 함수 외에 `spam, ham, eggs`라는 변수를 만들었다. 함수에서 쓰인 `foo, bar`나 아래서 선언된 변수명들을 프로그래밍에서 [Metasyntactic variable](https://futurecreator.github.io/2018/06/05/metasyntactic-variables-foo-bar/) 이라고 한다. syntactic이라고 하지만 프로그래밍 언어의 문법과는 아무상관이 없고 그냥 예시에서 관습적으로 많이 쓰이는 변수명들을 말한다.

`list_ops.py` 옆에 `use_list_ops.py`라는 새로운 파일을 만들어 저 함수와 변수들을 사용해보자. `import`를 활용해 외부 모듈을 가져오는 방법은 다양하다. 



### 1. 모듈 이름 그대로 가져오기

```python
foo = [1, 2, 3, 4, 5]
bar = [24, 52, 13, 27]

import list_ops

goo = list_ops.add(foo, bar)
print("{} + {} = {}".format(foo, bar, goo))
goo = list_ops.multiply(list_ops.spam, list_ops.ham)
print("{} * {} = {}".format(list_ops.spam, list_ops.ham, goo))
print("list_ops.spam: {}".format(list_ops.spam))
# => list_ops.spam: [51, 23]
```

외부 모듈에서 어떤 객체를 가져올 때는 `모듈명.객체명`으로 가져오면 된다. 위 예제에서 `add`와 `multiply` 함수가 성공적으로 수행된 것을 확인할 수 있다. 마지막 줄을 보면 import 한 모듈에서 변수를 가져올 수도 있다. 얼핏 보면 당연한 것 같지만 C언어를 생각해보면 당연하지는 않다. `list_ops.spam`을 프린트 했을 때 `[51, 23]`이 나온다는 것은 `list_ops.py`라는 스크립트가 **실행**되었다는 것을 뜻하기 때문이다. `list_ops`를 import 만 했지만 내부적으로는 `list_ops.py`가 실행된 것이다. 

`list_ops.py`에 실행 가능한 코드를 넣되 외부에서 import 할 때는 실행되지 않게 하고 싶다면 함수에서 배웠듯이 `__name__` 변수를 써야한다. `list_ops.py`에서는 `eggs`라는 변수가 `if __name__ == '__main__':` 조건문 아래 있기 때문에 외부에서 import 할 때는 선언되지 않는다.

```python
try:
    print("list_ops.eggs: {}".format(list_ops.eggs))
except Exception as e:
    print(e)
    # => module 'list_ops' has no attribute 'eggs'
```



### 2. 모듈 이름을 바꿔서 가져오기

일반적으로 모듈 이름은 모듈의 기능을 이해하기 쉽게 몇 개의 단어를 이어서 길게 짓는것이 보통이다. 게다가 패키지에서 복잡한 폴더 구조 아래 있는 모듈을 가져올 때는 더욱 import 문이 길어진다. 하지만 모듈의 객체를 불러올 때마다 긴 모듈 이름을 모두 쓰는 것은 번거로우므로 짧은 별명을 지어 가져올 수 있다. `import A as B` 라고 하면 `A` 모듈을 `B`라는 이름으로 가져오겠다는 뜻이다. 아래 코드에서도 `list_ops` 대신 `lo`를 써서 모듈 객체를 불러왔다.

```python
import list_ops as lo

goo = lo.subtract(foo, bar)
print("{} + {} = {}".format(foo, bar, goo))
goo = lo.divide(bar, foo)
print("{} * {} = {}".format(bar, foo, goo))
```



### 3. 모듈에서 지정한 객체만 가져오기

아예 외부 모듈명을 쓰지 않고 바로 함수나 변수를 사용하고 싶을 때는 `from module import object`를 쓰면 된다. `from`을 써서 가져온 객체는 모듈명을 생략하고 사용가능하다.

```python
from list_ops import add, subtract, spam

goo = add(foo, bar)
print("{} + {} = {}".format(foo, bar, goo))
goo = subtract(bar, foo)
print("{} - {} = {}".format(bar, foo, goo))
print("spam = {}".format(spam))
```



## 3. 패키지 만들기

패키지는 여러 모듈이 폴더 단위로 모인 것으로 폴더 안에 폴더를 넣으면 계층적인 패키지를 만들 수도 있다. 여러 모듈을 모은 패키지를 만들기 위해 `package`란 폴더 아래 `dict_ops.py`를 다음과 같이 만들어보자. 함수를 만들 때 원래는 함수 이름을 `list_add`나 `dict_add`처럼 함수 이름에 기능이 모두 표현되는 것이 좋으나 앞에 붙은 접미사는 모듈명으로 대체 가능하기 때문에 함수 이름을 간단하게 지어도 된다.

```python
# dict_ops.py
def add(foo, bar):
    out = {}
    for key in foo.keys():
        if key in bar:
            out[key] = foo[key] + bar[key]
    return out

def subtract(foo, bar):
    out = {}
    for key in foo.keys():
        if key in bar:
            out[key] = foo[key] - bar[key]
    return out

def multiply(foo, bar):
    out = {}
    for key in foo.keys():
        if key in bar:
            out[key] = foo[key] * bar[key]
    return out

def divide(foo, bar):
    out = {}
    for key in foo.keys():
        if key in bar:
            out[key] = foo[key] / bar[key]
    return out
```



이제 프로젝트 폴더 바로 아래 새로운 파이썬 파일을 만들고 (`package` 폴더 옆에) 예제를 실행해보자. 여러 사람의 키와 몸무게를 입력하여 BMI를 계산하는 코드다. 첫 줄을 보면 `package` 라는 패키지 안의 `list_ops`라는 모듈을 `lo`라는 약칭으로 불러왔다. C언어에서 `#include`를 항상 코드 맨 위에서 쓰듯이 파이썬에서도 `import`는 그 모듈이 언제 쓰이던지 항상 맨 윗줄에 사용하는 것을 권장한다.

```python
import package.list_ops as lo
import package.dict_ops as do

weights = [65, 90, 42, 76]
heights = [1.65, 1.78, 1.59, 1.80]
heights_sq = lo.multiply(heights, heights)
bmi = lo.divide(weights, heights_sq)
print("BMI:", bmi)
# => BMI: [23.875114784205696, 28.40550435551067, 16.61326688026581, 23.456790123456788]
```



그런데 알고보니 키와 몸무게를 사람 순서대로 넣은 것이 아니라 임의로 섞여서 들어간 것이었다. 이를 정확히 처리하기 위해 리스트를 딕셔너리로 바꿔서 연산을 해보자. `w_names`는 `weights`에 해당하는 이름이고 `h_names`는 `heights`에 해당하는 이름이다. 이름의 순서와 구성이 다르다는 것을 알 수 있다. 같은 사람의 키와 몸무게를 이용해 BMI를 계산하기 위해 `dict_ops`를 사용하였다.

```python
w_names = ["RM", "Suga", "Jin", "V"]
h_names = ["Jimin", "RM", "Suga", "Jin"]
weights = dict(zip(w_names, weights))
heights = dict(zip(h_names, heights))
print("dict weights:", weights)
print("dict heightss:", heights)
heights_sq = do.multiply(heights, heights)
bmi = do.divide(weights, heights_sq)
print("BMI:", bmi)
# => BMI: {'RM': 20.515086478979924, 'Suga': 35.59985760056959, 'Jin': 12.962962962962962}
```

`dict_ops`는 입력인자 `foo, bar`에 공통적으로 있는 key에 대해서만 연산을 하여 결과를 출력한다. 예시의 결과를 보면 키와 몸무게에서 이름이 겹치는 RM, Suga, Jin 세 명의 BMI만 나온 것을 확인할 수 있다.  



## HW1. 행렬 연산 패키지 만들기

`matrix`라는 패키지를 만들고 그 아래 `dense_matrix.py`와 `sparse_matrix.py`라는 두 개의 모듈을 만드시오.

- 제출기한: 10.01 (화) 수업시간
- 제출방식: LMS 수업페이지에 "HW1"이라는 게시판에 제출
- 제출내용: 아래 파일들을 묶어 압축한  `HW1_이름.zip`  제출
    1. `HW1.py`: 패키지 실행 코드 
    2. `matrix`: 패키지 폴더
    3. `result.txt`: `HW1.py`의 실행 결과



### 1. 리스트 2차원 행렬 연산

다음과 같은 코드가 동작하도록 `dense_matrix.py`에 리스트로 이루어진 두개의 행렬을 더하는 `add`와 곱하는 `multiply` 함수를 구현하시오. 

```python
import matrix.dense_matrix as dm

print("="*30 + "\nProblem 1")
dm1 = [[1, 2], [3, 4], [5, 6]]
dm2 = [[1, 2], [3, 4]]

res = dm.add(dm1[:2], dm2)
print("add:", res)
res = dm.multiply(dm1, dm2)
print("multiply:", res)
```

결과

```
==============================
Problem 1
add: [[2, 4], 
	  [6, 8]]
multiply: [[7, 10], 
           [15, 22], 
           [23, 34]]
```



### 2. 딕셔너리 2차원 행렬 연산

행렬 중에 대부분 값이 0이고 일부에서만 값을 갖는 행렬을 sparse matrix 라고 한다. Sparse matrix는 대부분 크기가 굉장히 크고 일부만 값을 가지고 있기 때문에 모든 값을 메모리에 올리는 건 비효율적이다. 그래서 `eigen`같은 수학 라이브러리에서는 sparse matrix를 위한 클래스를 따로 만들어 값을 가지고 있는 원소들만 저장한다. 아래 예시는 같은 행렬을 리스트로 만든 dense matrix와 딕셔너리로 만든 sparse matrix를 비교한 것이다.

```python
dense_mat = [[1, 0, 0], 
             [0, 2, 0], 
             [0, 0, 3]]
sparse_mat = {'rows': 3, 'cols': 3, '00':1, '11':2, '22':3}
```

딕셔너리의 키 값으로 좌표('yx')를 문자로 입력하고 그곳에 값을 넣는 방식이다.  

1. `sparse_matrix.py`에 딕셔너리 기반 두 개의 sparse matrix를 더하는 `add` 함수를 구현하시오. 출력 또한 딕셔너리 기반 sparse matrix로 출력하시오.
2. `sparse_matrix.py`에 딕셔너리 기반 sparse matrix를 리스트 기반 dense matrix로 변환하는 `dense`라는 함수를 구현하시오.

```python
import matrix.sparse_matrix as sm

print("="*30 + "\nProblem 2")
sm1 = {'rows': 3, 'cols': 3, '00': 1, '11': 2, '22': 3}
sm2 = {'rows': 3, 'cols': 3, '01': 1, '11': 2, '21': 3}
res = sm.add(sm1, sm2)
print("add:", res)
res = sm.dense(res)
print("dense:", res)
```

결과

```
==============================
Problem 2
add: {'rows': 3, 'cols': 3, '00': 1, '11': 4, '22': 3, '01': 1, '21': 3}
dense: [[1, 1, 0], 
        [0, 4, 0], 
        [0, 3, 3]]
```



### 3. 예외 처리 (심화)

`matrix` 패키지의 모든 함수에서 행렬의 크기가 맞지 않을 때의 예외처리 코드를 추가한 함수들을 새로 만드시오. 새로운 함수 이름은 기존 함수 이름에 `_handle_exception`이란 접미사를 붙여 만든다. (예를 들어 `add`는 `add_handle_exception`이 된다.) 예외처리는 다음 코드를 참고한다.

```python
def example(num):
    if num > 10:
        raise Exception("number > 10")

try:
    example(11)
except Exception as ex:
    print("[Exception]", ex)
```

다음은 각 함수에 대한 예외 상황들이다.

1. `dense_matrix.add_handle_exception(m1, m2)`: m1의 크기와 m2의 크기가 다를 때
2. `dense_matrix.add_handle_exception(m1, m2)`: m1이나 m2의 열의 개수가 일정하지 않을 때 e.g. `[[1, 2], [3]]`
3. `dense_matrix.multiply_handle_exception(m1, m2)`: m1의 너비와 m2의 높이가 다를때
4. `dense_matrix.multiply_handle_exception(m1, m2)`: m1이나 m2의 열의 개수가 일정하지 않을 때 e.g. `[[1, 2], [3]]`
5. `sparse_matrix.add_handle_exception(m1)`: m1이나 m2에 `rows, cols` 키가 없을 때
6. `sparse_matrix.add_handle_exception(m1, m2)`: m1이나 m2에 `rows, cols` 범위를 벗어나는 인덱스(키)가 있을 때
7. `sparse_matrix.add_handle_exception(m1, m2)`: m1의 크기와 m2의 크기가 다를 때
8. `sparse_matrix.dense_handle_exception(m1)`: m1에 `rows, cols` 키가 없을 때
9. `sparse_matrix.dense_handle_exception(m1)`: m1에 `rows, cols` 범위를 벗어나는 인덱스(키)가 있을 때



`dense_matrix`에서 행렬의 열의 개수가 일정한지를 확인하는 기능은 반복적으로 쓰이므로 아래와 같은 함수를 구현하여 사용하시오.

```python
def check_consistent_cols(mat):
    ...
    if not consistent:
        raise Exception("matrix columns are not consistent")
```

`sparse_matrix`에서 인덱스(키)가 `rows, cols`의 범위를 벗어나는지를 확인하는 기능은 반복적으로 쓰이므로 아래와 같은 함수를 구현하여 사용하시오.

```python
def check_indices(sparse):
    ...
    if out of bound:
        raise Exception("index out of bound")
```

다음은 각 예외 상황을 확인하는 코드다.

```python
print("="*30 + "\nProblem 3")
dm1 = [[1, 2], [3, 4], [6]]
dm2 = [[1, 2], [3, 4]]
dm3 = [[1, 2]]

try:
    res = dm.add_handle_exception(dm2, dm3)
except Exception as ex:
    print("[Exception] 1:", ex)

try:
    res = dm.add_handle_exception(dm1, dm2)
except Exception as ex:
    print("[Exception] 2:", ex)

try:
    res = dm.multiply_handle_exception(dm2, dm3)
except Exception as ex:
    print("[Exception] 3:", ex)

try:
    res = dm.multiply_handle_exception(dm1, dm2)
except Exception as ex:
    print("[Exception] 4:", ex)


sm1 = {'rows': 3, '00': 1, '11': 2, '22': 3, '33': 4}
sm2 = {'rows': 3, 'cols': 2, '00': 1, '11': 2, '22': 3}
sm3 = {'rows': 3, 'cols': 2, '00': 1, '11': 2}
sm4 = {'rows': 3, 'cols': 3, '01': 1, '11': 2, '21': 3}

try:
    res = sm.add_handle_exception(sm1, sm2)
except Exception as ex:
    print("[Exception] 5:", ex)

try:
    res = sm.add_handle_exception(sm2, sm4)
except Exception as ex:
    print("[Exception] 6:", ex)

try:
    res = sm.add_handle_exception(sm3, sm4)
except Exception as ex:
    print("[Exception] 7:", ex)

try:
    res = sm.dense_handle_exception(sm1)
except Exception as ex:
    print("[Exception] 8:", ex)

try:
    res = sm.dense_handle_exception(sm2)
except Exception as ex:
    print("[Exception] 9:", ex)
```

이를 구현하여 실행한 결과는 다음과 같다. 예외 메시지는 상황에 맞춰 임의로 작성하였다.

```
==============================
Problem 3
[Exception] 1: different matrix size
[Exception] 2: matrix columns are not consistent
[Exception] 3: cannot multiply (2,2) x (1,2)
[Exception] 4: matrix columns are not consistent
[Exception] 5: either 'rows' or 'cols' is not in keys
[Exception] 6: index out of bound
[Exception] 7: different matrix size
[Exception] 8: either 'rows' or 'cols' is not in keys
[Exception] 9: index out of bound
```

