---
layout: post
title:  "Python Containers"
date:   2019-09-08 09:00:13
categories: 2019-2-robotics
---



# 컨테이너 (Container)

파이썬에는 네 가지 기본 컨테이너(container) 타입이 있다. 컨테이너란 말 그대로 무언가를 담을 수 있는 것을 말하고 파이썬에서는 여러 데이터를 담을 수 있는 데이터 타입을 뜻한다. 기본 컨테이너로는 `List, Dictionary, Tuple, Set` 이 있다. 이 네 가지 컨테이너는 파이썬이 가진 편리함의 원천이다. 파이썬이 동적 타입이므로 서로 다른 종류의 데이터도 상관없이 하나의 컨테이너에 담을 수 있다. 네 가지 타입은 List > Dictionary >> Tuple > Set 순서로 많이 쓰인다. 특히 List와 Dictionary는 많이 쓰이기도 하고 정말 쉽고 유용하기 때문에 잘 알아야한다. Set은 자주 쓰이지 않기 때문에 여기서는 생략한다.



## 1. List

리스트(`List`)는 말 그대로 여러개의 데이터를 목록(list)처럼 담아둘 수 있는 자료형이다. 그런데 파이썬 자체가 동적 타입이기 때문에 타입에 상관없이 아무 자료나 담을 수 있다. 리스트 안에 리스트를 담을 수도 있고 이 후에 배울 Dictionary 나 클래스 객체 등 어떤 데이터든 담을 수 있다. 

```python
empty_list1 = []
empty_list2 = list()
basic_list = ["Hello", 1234, 1.234, True]
depth2_list = ["Hello", 1234, [1.234, True]]
depth3_list = [["Hello"], [1234, [1.234, True]]]
```

`empty_list1`, `empty_list2`는 비어있는 리스트를 만드는 두 가지 방법이다. `basic_list`은 네 가지 기본 자료형으로 만든 리스트이고 `depth2_list`, `depth3_list`은 리스트 안에 리스트를 담은 것이다.

### 1.1 리스트 인덱싱과 슬라이싱

리스트를 다루는 방법은 앞서 문자열을 다루는 방법과 비슷하다. 리스트에서 특정 원소를 출력하는 인덱싱(Indexing)은 `[]`(braket operator)를 쓴다. 문자열처럼 뒤에서부터 음수로 인덱싱도 가능하다. 리스트가 여러겹으로 겹쳐있다면 원하는 원소가 나올때까지 단계별로 인덱스를 붙이면 된다. 마지막 줄처럼 인덱스가 리스트 범위를 넘어가면 에러가 발생하므로 주의해야 한다.  

```python
print("\nIndexing: 1.234에 접근하기")
print("basic indexing:", basic_list[2])
print("negative indexing:", basic_list[-2])
print("first indexing:", depth2_list[2])
print("second indexing:", depth2_list[2][0])
print("first indexing:", depth3_list[1])
print("second indexing:", depth3_list[1][1])
print("third indexing:", depth3_list[1][1][0])
try:
    print(depth2_list[5])
except IndexError as ie:
    print("IndexError:", ie)
```

결과

```
Indexing: 1.234에 접근하기
basic indexing: 1.234
negative indexing: 1.234
first indexing: [1.234, True]
second indexing: 1.234
first indexing: [1234, [1.234, True]]
second indexing: [1.234, True]
third indexing: 1.234
IndexError: list index out of range
```

인덱싱이 리스트의 원소를 출력한다면 슬라이싱(Slicing)은 리스트에서 일부 원소들을 묶어 **새로운 리스트로 출력**한다. 마찬가지로 음수를 이용해 뒤에서부터 슬라이싱 범위를 지정할 수도 있다. 슬라이싱의 경우 슬라이싱 범위가 인덱스 범위를 넘어가더라도 슬라이싱 범위 안의 데이터가 있으면 그만큼만 출력하고 없으면 빈 리스트(`[]`)를 출력한다.

```python
print("\nSlicing")
print("[start:end]", basic_list[1:3])
print("[start:]", basic_list[2:])
print("[:end]", basic_list[:2])
print("[start:negative_end]", basic_list[1:-1])
print("[:negative_end]", basic_list[:-2])
print("[negative_start:negative_end]", basic_list[-4:-2])
print("[nested list1]", depth2_list[1:3])
print("[nested list2]", depth3_list[0:2])
print("[partially overlap]", basic_list[2:10])
print("[out of range]", basic_list[5:10])
```

결과

```
Slicing
[start:end] [1234, 1.234]
[start:] [1.234, True]
[:end] ['Hello', 1234]
[start:negative_end] [1234, 1.234]
[:negative_end] ['Hello', 1234]
[negative_start:negative_end] ['Hello', 1234]
[nested list1] [1234, [1.234, True]]
[nested list2] [['Hello'], [1234, [1.234, True]]]
[partially overlap] [1.234, True]
[out of range] []
```

마지막 실행 결과를 보면 `basic_list[2:10]`은 2번부터 9번까지의 데이터를 출력해야 하는데 `basic_list`가 3번까지 밖에 없으므로 2, 3번 원소만 출력한다. `basic_list[5:10]`는 슬라이싱 범위 전체가 `basic_list`의 범위를 벗어나 있으므로 빈 리스트(`[]`)를 출력한다.  

#### 연습문제

1)  `depth2_list`에서 `[1.234]`를 출력해 보세요. (주의: 1.234가 아닌 [1.234] 입니다.)



### 1.2 리스트 연산

리스트도 문자열처럼 `+`를 통해 두 리스트를 합칠 수 있고 `*`를 통해 리스트 원소들을 반복할 수 있다.

```python
mammal = ["dog", "cat", "human"]
reptile = ["snake", "lizard", "frog"]
bird = ["eagle", "sparrow", "chicken"]
animal = mammal*2 + reptile + bird
print("\nlist concatenation")
print("animal:", animal)
# => animal: ['dog', 'cat', 'human', 'dog', 'cat', 'human', 
# 'snake', 'lizard', 'frog', 'eagle', 'sparrow', 'chicken']
```



### 1.3 리스트 내장 함수

리스트를 쓰다보면 (크기순, 알파벳순) 정렬을 한다던지 (`sort`), 원소를 추가한다던지 (`append`), 중간에 삽입한다던지 (`insert`), 특정 원소의 인덱스(위치)를 반환한다던지(`index`) 등의 기능이 필요하다. **특히 `append`와 `sort`**가 많이 쓰인다. 함수명을 굳이 외울 필요없이 해당기능을 영어로 써보면 그 함수가 이미 있다. 다음 예시를 보며 함수명들을 익혀보자. 리스트 함수들을 쓸 때 주의할 점은 대부분의 함수들이 `in-place` 함수라는 것이다. 함수로 수정한 리스트 결과가 리턴으로 나오지 않고 함수를 실행한 변수 자체를 수정한다는 것이다. 하지만 대부분의 외부 패키지들은 데이터를 수정하는 함수들이 원본은 그대로 둔채 수정된 결과를 리턴하므로 잘 구분해서 써야한다.

```python
print("\nlist functions")
tottenham = ['Kane', 'Moura', 'Lloris', 'Sissoko', 'Alli', 'Rose']
print("Tottenham vs Southampton 2019-03-10 starting line up: \n", tottenham)
# sort(): 원소 알파벳순, 크기순 정렬, in-place 함수기 때문에 아무것도 리턴하지 않는다.
print("sort() is a in-place function:", tottenham.sort())
print("sort by name:", tottenham)
# remove(): 입력한 원소를 삭제
tottenham.remove('Moura')
# insert(): 원하는 위치에 원소 삽입
tottenham.insert(1, 'Son')
print("At 72, Moura out Son in:", tottenham)
# pop(): 입력이 없으면 마지막 원소를 삭제하고 pop(index)는 index의 원소를 삭제한다.
print("At 82, pop Alli:", tottenham.pop(0))
print("At 82, del Rose")
del tottenham[-2]
# append(): 원소를 마지막에 추가한다. 두 리스트의 원소들을 합칠 때는 +나 extend()를 쓴다.
tottenham.append('Davies')
tottenham.append('Llorente')
print("At 82, Davies and Llorente in:", tottenham)
# reverse(): 순서를 거꾸로 뒤집는다.
tottenham.reverse()
print("reverse order", tottenham)
```

결과

```
list functions
Tottenham vs Southampton 2019-03-10 starting line up: 
 ['Kane', 'Moura', 'Lloris', 'Sissoko', 'Alli', 'Rose']
sort() is a in-place function: None
sort by name: ['Alli', 'Kane', 'Lloris', 'Moura', 'Rose', 'Sissoko']
At 72, Moura out Son in: ['Alli', 'Son', 'Kane', 'Lloris', 'Rose', 'Sissoko']
At 82, pop Alli: Alli
At 82, del Rose
At 82, Davies and Llorente in: ['Son', 'Kane', 'Lloris', 'Sissoko', 'Davies', 'Llorente']
reverse order ['Llorente', 'Davies', 'Sissoko', 'Lloris', 'Kane', 'Son']
```

#### 연습문제

1) `top_lang_2009`에서 원소를 삭제, 추가하여 `top_lang_2019`을 만들어 보세요.

```python
top_lang_2009 = ["Java", "C", "Python", "C++", "C#"]
# => top_lang_2019 = ["Java", "C", "Python", "C++", "PHP"]
```



### 1.4 리스트 관련 함수

- `len()`:  파이썬 내장 함수로 여러 원소를 담고 있는 대부분의 객체들은 이 함수로 길이(갯수)를 잴 수 있다. 문자열(str)의 길이, `list, dict, tuple, set` 등의 자료구조의 원소 수를 읽을 수 있다.

```python
string = "Hello"
print("\nlen of {}:".format(string), len(string))
# => len of Hello: 5
mylist = [1, 2, 3, 4]
print("len of {}:".format(mylist), len(mylist))
# => len of [1, 2, 3, 4]: 4
```

- `del`: 객체를 삭제하는 함수로 자료 구조에서 특정 원소를 삭제할 때 쓰인다. `del()`이 아님에 유의하자.

```python
mylist = [1, 2, 3, 4, 5]
del mylist[2]
print("\nafter deleting [2]:", mylist)
# => after deleting [2]: [1, 2, 4, 5]
del mylist[2:]
print("after deleting [2:]:", mylist)
# => after deleting [2:]: [1, 2]
```

- 원소 변경: 인덱싱이나 슬라이싱으로 잡은 리스트 범위에 `=` operator로 원소들을 수정할 수 있다. 참고로 문자열은 `=`를 이용한 일부 문자 수정이 안된다.

```python
mylist = [1, 2, 3, 4, 5]
mylist[0] = "Life"
print("\nchange element by indexing:", mylist)
# => change element by indexing: ['Life', 2, 3, 4, 5]
mylist[1:4] = ["is", "too", "short"]
print("\nchange elements by slicing:", mylist)
# => change elements by slicing: ['Life', 'is', 'too', 'short', 5]
```

- `join()`: 리스트의 내부 문자열 원소들을 하나의 문자열로 연결해준다. 연결할 때 각 문자열 사이에 `" "` 사이에 들어있는 문자열을 끼워 넣어준다.

```python
print("\njoin strings")
path = ["/home", "ian", "work", "ian-lecture"]
print("joined path:", "/".join(path))
# => joined path: /home/ian/work/ian-lecture
time = ["13", "20", "30"]
print("joined time:", ":".join(time))
# => joined time: 13:20:30
```

- `in`: `in`은 함수가 아니라 operator다. 리스트에 특정 원소가 들어있는지 확인할 때 쓴다.

```python
print("\n'in' operator")
twice = ["나연", "정연", "모모", "사나", "지효", "미나", "다현", "채영", "쯔위"]
if "채영" in twice:
    print("채영은 트와이스 입니다.")
if "채령" not in twice:
    print("채령은 트와이스가 아닙니다. 있지 입니다.")
```



## 2. Dictionary

데이터를 다루다 보면 단순히 자료를 나열하는 리스트로는 충분하지 않을 때가 있다. 리스트는 같은 속성의 자료를 여러개 가지고 있을 때는 유용하지만 다른 속성의 자료를 리스트에 넣으면 코드를 읽기가 힘들어진다. 

```python
good_for_list = ["pooh", "tigger", "piglet", "rabbit"]
pooh = ["bear", 5, 50]
tigger = ["tiger", 4, 40]
print("list based data management")
print("pooh's species is", pooh[0])
print("pooh's weight is", pooh[2])
print("tigers's age is", tigger[1])
```

첫 줄의 `good_for_list`는 리스트로 저장하기 좋은 자료다. 모두가 이름을 나타내고 있다. 하지만 그 아래를 보라. 자료를 이렇게 저장하면 순서를 똑같이 맞추지 못 할 경우 잘못된 결과를 나타내기 쉽고 코드만 봐서는 왜 [0]이 종이고 [1]이 나이인지 이해하기 어렵다.  

이러한 대응관계를 명시적으로 보여줄 수 있는 자료 구조가 딕셔너리(`Dictionary`)다. 사전에서 어떤 단어를 찾으면 그 단어에 대산 설명이 나오듯이 `Dictionary`는 키(Key)를 입력하면 그에 해당하는 값(Value)를 출력해주는 자료 구조다. 리스트는 `[]`(bracket)으로 만들었지만 딕셔너리는 `{}`(brace)로 만든다. `:`(colon)을 사이에 두고 앞에 Key 값을 쓰고 뒤에 Value 값을 쓴다. 여러개의 Key:Value 쌍은 쉼표(,)로 구분한다. 값을 꺼낼때는 리스트처럼  `[]`(bracket)를 쓰지만 리스트는 인덱스 숫자를 입력하고 딕셔너리는 키 값을 입력한다.

```python
mydict = {Key1:Value1, Key2:Value2, ...}
print("call value from key1", mydict[Key1])
```

곰돌이 푸우 예제를 이해하기 쉽게 다시 써보자. 타이핑하는데 손은 더 들지만 코드를 명시적으로 이해할 수 있다. 특히 여럿이 같이 작업할 때는 남들이 편하게 이해할 수 있게 쓰는 것이 중요하다.

```python
pooh = {"species": "bear", "age": 5, "weight": 50}
tigger = {"species": "tiger", "age": 4, "weight": 40}
print("\ndict based data management")
print("pooh's species is", pooh["species"])
print("pooh's weight is", pooh["weight"])
print("tigers's age is", tigger["age"])
```

딕셔러너리를 만들 때 `Value`에는 어떤 객체가 들어가도 상관없다. 리스트가 들어갈 수도, 또 다른 딕셔너리가 들어가도 된다. 하지만 `Key`는 숫자나 문자열 (혹은 그 변수)을 써야한다. 딕셔너리는 `Key`로 들어온 값을 `hash()` 함수를 통해 hash를 생성하고 이를 기록해 두었다가 특정 키를 찾을 때 활용한다. `Key` 값을 직접 비교하지 않고 hash로 변환해서 검색하기 때문에 검색 속도가 빠르다. 다음 예시를 실행해보자.

```python
print("\nhash example")
print("hash of 1:", hash(1))
print("hash of python:", hash("python"))
try:
    print("hash of []:", hash([]))
except TypeError as te:
    print("[TypeError]", te)
```

이제 딕셔너리의 활용법에 대해 더 알아보자.

### 2.1  기본 사용법

딕셔너리의 기본 특징은 **순서가 없다**는 것이다.(Unordered) 보통은 입력한 순서가 그대로 유지되는 것 처럼 보이지만 값을 넣고 빼고 하다보면 `print()` 했을 때 순서가 뒤바뀔 때도 있다. 사실 딕셔너리는 오직 `Key`로만 `Value`에 접근할 수 있으니 순서는 상관없다. 다음 예제를 통해 기본 사용법을 익혀보도록 하자.

```python
print("\nbasic usage")
pooh = {"species": "bear", "age": 5, "weight": 50}
# 데이터 읽기: 특정 `key`에 연결된 `Value`를 읽기 위해서는 `[Key]`하면 된다.
print("pooh's age is", pooh["age"])
# 데이터 수정: 특정 `key`에 연결된 `Value`를 수정할 때는 그냥 값을 키에 넣으면 된다.
pooh["age"] = 10
print("pooh's age is", pooh["age"])
# 데이터 추가: 이미 만들어진 딕셔너리에 `Key:Value` 쌍을 추가하는 방법 역시 그냥 값을 키에 넣으면 된다.
pooh["height"] = 1.2
print("pooh:", pooh)
# 데이터 삭제: 리스트처럼 `del`을 이용한다.
del pooh["weight"]
print("pooh:", pooh)
```

결과

```
basic usage
pooh's age is 5
pooh's age is 10
pooh: {'species': 'bear', 'age': 10, 'weight': 50, 'height': 1.2}
pooh: {'species': 'bear', 'age': 10, 'height': 1.2}
```

만약 없는 키를 입력하면 `KeyError`가 발생한다. 해당 키가 있는지 확신이 없을 때는 `get()` 함수를 쓰거나 `in` 연산자를 통해 키의 존재유무를 확인 후 쓸 수도 있다. `get()`을 쓰면 키가 없을 경우 `None`을 출력하는데 `None`은 말 그대로 아무것도 아니라는 뜻으로 True, False 처럼 파이썬에 지정된 키워드다.

```python
pooh = {"species": "bear", "age": 5, "weight": 50}
try:
    print("\ntry non-existing key")
    print("pooh's color?", pooh["color"])
except KeyError as ke:
    print("[KeyError]", ke)

print("pooh's color?", pooh.get("color"))
if "color" in pooh:
    print("pooh's color is", pooh["color"])
else:
    print("pooh has no color")
```

결과

```
try non-existing key
[KeyError] 'color'
pooh's color? None
pooh has no color
```



### 2.2 관련 함수

딕셔너리를 쓰다보면 Key만 혹은 Value만 따로 쓰고 싶을 때가 있다. 예를 들어 딕셔너리를 `"이름":"점수"` 형태로 만들었는데 명단만 필요하거나 점수만 필요한 경우가 있다. 혹은 이들을 `Key:Value` 관계가 아닌 동등만 데이터 쌍(tuple)으로 받고 싶을 수도 있다. 다음 예제에서 그 답을 찾아보자.

```python
print("\ndict functions")
scores = {"pooh": 80, "tigger": 70, "piglet": 90, "rabbit": 85}
print("names:", scores.keys())
print("scores:", scores.values())
print("items:", scores.items())
```

결과

```
dict functions
names: dict_keys(['pooh', 'tigger', 'piglet', 'rabbit'])
scores: dict_values([80, 70, 90, 85])
items: dict_items([('pooh', 80), ('tigger', 70), ('piglet', 90), ('rabbit', 85)])
```

각 결과는 리스트처럼 보이지만 `dict_keys`, `dict_values`, `dict_items` 라는 다른 데이터 형식이다. 이들은 언제든 `list()`로 감싸면 쉽게 리스트로 변환할 수 있다.

```python
print("names:", list(scores.keys()))
print("scores:", list(scores.values()))
print("items:", list(scores.items()))
```

결과

```
names: ['pooh', 'tigger', 'piglet', 'rabbit']
scores: [80, 70, 90, 85]
items: [('pooh', 80), ('tigger', 70), ('piglet', 90), ('rabbit', 85)]
```



## 3. Tuple

튜플(Tuple)을 한 마디로 말하면 수정불가능한 리스트다. 리스트는 `[]`, 딕셔너리는 `{}`로 만들었다면 튜플은 `()`(parenthesis)로 만든다. 혹은 `()`를 생략해도 된다. 튜플은 생성만 할 뿐 원소를 추가, 수정, 삭제할 수 없다. 생성된 튜플은 값을 읽을 수만 있는데 리스트와 동일하게 인덱싱과 슬라이싱을 통해 읽을 수 있다.

```python
print("\nHow to use tuple")
empty_tuple1 = ()
empty_tuple2 = tuple()
basic_tuple1 = ("Hello", 1234, 1.234, True)
basic_tuple2 = "Hello", 1234, 1.234, True
depth2_tuple = ("Hello", 1234, (1.234, True))
print("read tuple", basic_tuple1[0])
print("read tuple", basic_tuple2[1])
print("read tuple", basic_tuple[:3])
```

튜플은 여러 값을 하나에 담았다가 다시 여러 변수에 나눠줄 수 있어서 경우에 따라 유용하게 쓸 수 있다.

```python
print("\ndistribute values")
pooh = "pooh", "bear", 5, 50
name, species, age, weight = pooh
print("tupled pooh info:", name, species, age, weight)
```

