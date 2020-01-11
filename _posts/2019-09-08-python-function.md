---
layout: post
title:  "Python Control Statements and Function"
date:   2019-09-08 09:10:13
categories: 2019-2-robotics
---



# 제어문 (Control Statements)



## 1. 기본 사용법

이전 장에서도 배웠듯이 파이썬에도 `if, for, while`이 있다. 특히 파이썬은 자료구조형과 밀접하게 연계되어 자료구조에 들어있는 자료들에 대해 쉽게 반복문을 실행할 수 있다. 제어문의 기본 용법과 편리하게 쓸 수 있는 다양한 문법들을 배워보자. `if, for, while`의 기본 용법은 다음과 같다.

```
if condition1:
	statements_when_condition1_is_true
elif condition2:
	statements_when_condition2_is_true
else:
	statements_when_no_condition_is_true

for elem in something_iterable:
	statements_to_process_elem

while condition:
	statements_while_condition_is_true
```

마블과 DC의 영웅들~~아니고 돈벌이 캐릭터~~로 `if, for, while`의 기본 용법을 배워보자. 

```python
marvel_heroes = ["iron man", "thor", "hulk", "spider man", "black widow", "capt. america", "capt. marvel"]
dc_heroes = ["batman", "superman", "aquaman", "wonder woman", "harley quinn"]
all_heros = marvel_heroes + dc_heroes

myhero = "batman"
if myhero in marvel_heroes:
    print("{}: Avengers are super cool!".format(myhero))
elif myhero in dc_heroes:
    print("{}: We save martha...".format(myhero))
else:
    print("{}: We are villains!!".format(myhero))

print("list of marvel heroes")
for hero in marvel_heroes:
    print(f"    {hero}")

print("list of dc heroes")
for i in range(len(dc_heroes)):
    print(f"{i})  {dc_heroes[i]}")

i = 0
print("DC's male heroes ends with 'man'")
while i < len(dc_heroes) and dc_heroes[i].endswith('man'):
    print(f"    {dc_heroes[i]}")
    i += 1
```



## 2. 반복문 제어: `continue, break`

반복문을 돌리다 보면 조건에 따라 중간에 처리를 건너뛰고 싶거나 그만두고 반복문을 종료하고 싶을 때가 있다. `continue`는 반복문에서 `continue` 이후의 과정을 건너뛰고 다음 loop으로 넘어가는 것이고, `break`는 반복문 자체를 끝낸다.

```python
print("list of marvel heroes")
for hero in marvel_heroes:
    if hero.startswith("spider"):
        print("    Peter Parker by Tobey Maguire was not a kid: \"With great power comes great responsibility.\"")
        continue
    if hero.startswith("capt"):
        print("    one captain is enough. let me stop here")
        break
    print(f"    {hero} is cool")

name = None
print("Press 'q' to quit")
while name != 'q':
    print("type dc hero's name")
    name = input()
    if name is 'q':
        break
    if name not in dc_heroes:
        print(f"{name} is not dc hero")
        continue
    index = dc_heroes.index(name)
    print(f"{name}'s index =", index)
```



## 3. for 응용: `enumerate, zip`

파이썬에서는 `while`보다 `for`문의 활용법이 다양하다. 대표적으로 `enumerate, zip`이 있다. `enumerate`는 `for`문에서 리스트를 반복할 시 원소 뿐만 아니라 인덱스도 받을 수 있는 반복 객체를 만들어 준다. `zip`은 두 개의 리스트를 묶어서 각 리스트의 원소가 하나씩 합쳐진 튜플의 반복 객체를 만들어준다. 이들을 리스트로 변환해보면 무엇을 주는지 명확히 볼 수 있다.

```python
print("\nwhat does enumerate return?", enumerate(marvel_heroes))
print(list(enumerate(marvel_heroes)))
print("what does zip return?", zip(marvel_heroes, dc_heroes))
print(list(zip(marvel_heroes, dc_heroes)))

print("print only first 3 heroes with index")
for index, name in enumerate(marvel_heroes):
    if index >= 3:
        break
    print("marvel hero:", index, name)

print("print pairs of marvel and dc heroes")
for mv, dc in zip(marvel_heroes, dc_heroes):
    print("{} vs {}".format(mv, dc))
```



## 4. for for dict

지금까지 주로 리스트를 이용한 `for`문을 보았는데 딕셔너리도 반복문에 자주 사용된다. 딕셔너리 자체로는 `for`문을 통해 반복할 수 없고 `Key`가 필요한지, `Value`가 필요한지, 둘 다 필요한지에 따라 맞는 함수를 통해 반복가능한(iterable) 자료형으로 바꾸어 쓰면 된다.

```python
hero_names = {"iron man": "로다주", "thor": "햄식이"}
print("iterate over Dictionary KEYS")
for character in hero_names.keys():
    print("character:", character)
print("iterate over Dictionary VALUES")
for name in hero_names.values():
    print("name:", name)
print("iterate over Dictionary ITEMS")
for character, name in hero_names.items():
    print(f"character:name = {character}:{name}")
```



## 5. List Comprehension

리스트를 다루다 보면 다른 리스트에 기반해서 새로운 리스트를 만들거나 기존 리스트에서 필요한 것만 걸러서 새로운 리스트를 만들고 싶을 때가 있다. 일단 위에서 배운 내용을 토대로 이를 구현해보자.

```python
print("\ncreate a new list from the existing list")
super_heroes = []
for hero in marvel_heroes:
    super_heroes.append(hero + "_super")
print("super heroes", super_heroes)

print("\nextract a subset of the existing list")
man_heroes = []
for hero in marvel_heroes:
    if hero.endswith("man"):
        man_heroes.append(hero)
print("man heroes:", man_heroes)
```

파이썬에는 리스트를 선언하는 `[]` 안에 `for`문을 넣어 저 두 개의 `for`문을 한 줄로 처리하는 방법이 있는데 이를 `list comprehesion`이라고 한다.

```python
print("\nsquare of integers")
int_square = [i ** 2 for i in range(10)]
print(int_square)

print("\ncreate a new list by list comprehension")
super_heroes = [hero + "_super" for hero in marvel_heroes]
print("super heroes", super_heroes)

print("\nextract a subset by conditioned list comprehension")
man_heroes = [hero for hero in marvel_heroes if hero.endswith("man")]
print("man heroes:", man_heroes)
```

`list comprehension`은 앞에 간단한 처리과정을 넣어서 다양하게 쓰일 수 있고 다중 `for`문이 들어갈 수도 있다. 이러한 기능을 `list`만 쓸 수 있는 건 아니고 `dictionary`와 `set`에도 적용될 수 있다.

```python
print("\ndictionary comprehension")
abilities = ["suit", "Mjölnir", "physical power", "spider web"]
heroes = {name: power for name, power in zip(marvel_heroes, abilities)}
print("hero's ability", heroes)
```



# 함수 (Function)

함수란 주어진 입력에 대해서 정해진 동작을 하고 필요시 결과값을 반환하는 코드 묶음이다. 함수를 사용하는 이유는 크게 두 가지다. 

1. 반복적으로 사용되는 기능(코드)을 재사용하기 위해서다. 함수가 없다면 어떤 기능이 필요할 때마다 같은 코드를 다시 써야하고 그러면 전체 코드가 불필요하게 길어진다. 또한 함수로 만들어 놓으면 다른 프로젝트에서도 해당 기능만 가져다 쓸 수 있다.
2. 그러나 보통은 한번만 쓰이는 코드라도 함수로 묶는 경우가 많다. 그 이유는 코드를 기능별로 묶기 위해서다. 함수를 기능단위로 묶으면 코드를 계층적으로 구조화할 수 있어 코드를 읽기도 편하고 특정 기능을 하는 코드를 찾고 수정하는 일도 편해진다. 코드를 기능별로 잘 나누고 함수 이름을 기능에 맞게 지으면 코드에 주석을 쓸필요가 거의 없어진다.



## 1. 함수 정의

C언어에서 함수의 프로토 타입을 먼저 적는 것을 함수 선언, 함수 내용을 구현하는 코드를 함수 정의라고 부르는데 파이썬에서는 이를 나누지 않고 바로 정의만하면 된다. 기본적인 함수의 구조는 다음과 같다. 파이썬은 다중 입력과 다중 출력을 지원하고 입력이나 출력이 없을 수도 있다. 

```python
def function_name(arg1, arg2):
    statement1
    statement2
    return out1, out2

def function_no_output(arg1, arg2):
    statement1
    statement2

def function_no_input_output():
    statement1
    statement2
```

가장 간단한 예로 아래와 같은 더하기 함수를 만들 수 있다.

```python
def add(n1, n2):
    return n1+n2

print(add(1, 2))
```

좀 더 복잡한 리스트 평균을 내는 함수를 다음과 같이 정의할 수 있다. `average_list`는 이름 그대로 리스트의 평균을 구하는 함수인데 시작 인덱스(`start`)와 끝 인덱스(`end`)를 지정할 수 있고 중간에 빠져야할 인덱스(`skip`)를 지정할 수 있다. 

```python
def average_list(data, start, end, skip):
    if end is None:
        avg_data = data[start:]
    else:
        avg_data = data[start:end]

    sum = 0
    skip_count = 0
    for ind, num in enumerate(avg_data):
        if ind in skip:
            skip_count += 1
        else:
            sum += num
    dlen = len(avg_data) - skip_count
    average = sum / dlen
    print(f"average {start}~{end}, skip={skip} over {len(data)} numbers = {average}")
    return average
```



## 2. Keyword Arguments

파이썬은 동적 타입이기 때문에 입력인자에 어떤 값을 넣어도 일단 실행은 된다. 그렇기 때문에 함수에 입력인자가 많을 때는 순서에 맞지 않는 입력인자를 넣는 실수를 범할 가능성이 크다. 위에서 정의한 `average_list`는 다음과 같이 사용할 수 있다.

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
avg = average_list(data, 2, 7, [4])
```

이렇게 입력인자가 다수인 경우에 어떤 인자를 몇 번째에 넣어야 할지 헷갈릴 수 있다. 혹은 다른 사람이 코드를 읽을 때 `average_list(data, 2, 7, [4])` 만 보면 의미를 이해하기 어렵다.  
파이썬에서는 명시적인 코드를 선호하기 때문에 다음과 같이 입력 값이 어느 입력인자로 들어가야 하는지 명시적으로 보여줄 수 있다. 이렇게 입력하는 인자를 **keyword argument (키워드 인자)**라 한다. 반면 기존 방식대로 순서에 의해 할당되는 입력 인자를 **positional argument (위치 인자)**라 한다.

```python
print("use keyword arguments")
avg = average_list(data, 2, 7, skip=[4])
avg = average_list(data, 2, end=7, skip=[4])
avg = average_list(data, start=2, end=7, skip=[4])
avg = average_list(data=data, start=2, end=7, skip=[4])
```

키워드 인자를 쓸 때 유의할 점은 위치 인자와 섞어 쓸 경우 **키워드 인자는 반드시 위치 인자 뒤**에 나와야 한다는 것이다. 따라서 아래 코드는 에러가 난다.

```python
avg = average_list(data, start=2, 7, [4])
```

조금 더 생각해보면 어느 인자로 어떤 값이 들어가는지 지정해주니 굳이 순서를 지킬 필요도 없다. **키워드 인자 끼리는 순서를 섞어도 된다.** 이때도 위치 인자를 먼저 쓰고 키워드 인자를 써야한다.

```python
print("mix keyword arguments")
avg = average_list(data, start=2, end=7, skip=[4])
avg = average_list(data, end=7, start=2, skip=[4])
avg = average_list(data, skip=[4], start=2, end=7)
```



## 3. 인자 기본값 지정

`average_list`는 네 개의 입력 인자로 리스트의 평균 계산 과정을 자세히 조절할 수 있지만 보통 많이 쓰는 기능은 단순히 리스트 전체에 대해서 평균을 구하는 것일 것이다. 이때도 `average_list(data, 0, 0, [])`과 같이 나머지 입력인자를 다 써주는 것이 번거로울 수 있다.  `data`로 들어오는 필수 인자를 제외하고 나머지 인자에 기본값을 지정해주면 입력 인자를 적게 넣어도 작동한다. 기본값을 주는 방법은 C언어와 동일하게 함수 선언에서 입력인자에 기본 값을 할당하면 된다. (`argument=default_value` )  `average_list`를 기본 값을 이용해 사용하게 해주는 `average_list_with_default`를 다음과 같이 정의하고 사용할 수 있다.

```python
def average_list_with_default(data, start=0, end=None, skip=None):
    if skip is None:
        skip = []
    return average_list(data, start, end, skip)

print("function default arguments")
avg = average_list_with_default(data)
avg = average_list_with_default(data, 3)
avg = average_list_with_default(data, end=5)
avg = average_list_with_default(data, skip=[3, 4])
```

- 첫 번째 호출에서는 `data`를 제외한 나머지 인자를 기본값으로 사용한 것이다.
- 두 번째 호출은 `start`까지만 값을 입력한 것이다. 
- 세 번째 호출은 키워드 인자를 사용하여 `start`는 기본값을 쓰고 `end`만 입력한 것이다.
- 네 번째 호출은 키워드 인자를 사용하여 `start, end`는 기본값을 쓰고 `skip`만 입력한 것이다.

다음은 그래프를 그리는 파이썬 패키지인 `matplotlib`에서 boxplot을 그리는 함수의 선언부다. 입력인자가 굉장히 많지만 대부분 `None`이라는 기본 값이 들어가 있어 필수 입력인자는 아니라는 것을 알 수 있다. 실제로 `x`라는 입력인자에 데이터만 넣으면 기본 boxplot이 그려지고 나머지는 boxplot의 세부적인 모양에 관한 옵션이다.

```python
matplotlib.pyplot.boxplot(x, notch=None, sym=None, vert=None, whis=None, positions=None,
                          widths=None, patch_artist=None, bootstrap=None, 
                          usermedians=None, conf_intervals=None, meanline=None, 
                          showmeans=None, showcaps=None, showbox=None, showfliers=None, 
                          boxprops=None, labels=None, flierprops=None, medianprops=None, 
                          meanprops=None, capprops=None, whiskerprops=None, 
                          manage_xticks=True, autorange=False, zorder=None, *, data=None)
```



## 4. 변수 범위(Scope)

변수 범위란 코드에서 변수에 접근 가능한 범위를 말한다. C언어에서는 `{}`로 묶인 범위 안에서 선언된 변수는 `{}` 안에서만 접근 가능하다. 파이썬에서는 `{}` 대신 들여쓰기로 코드 블럭을 구분하지만 블록 안에서 선언한 변수를 블록 밖에서도 쓸 수 있다. 하지만 이 경우 블록이 실행되지 않으면 변수도 생성되지 않기 때문에 에러가 날 가능성도 있다.

```python
print("\nvariable scope")
if True:
    var_created = "created"
print("variable created inside block:", var_created)

if False:
    var_not_created = "not created"
try:
    print("variable NOT created inside block:", var_not_created)
except NameError as ne:
    print(ne)
```

함수 내부에서 선언한 변수는 지역 변수(local variable)로 인식되어 함수 외부에서 쓸 수 없다. 반대로 함수 외부에서 선언된 변수는 전역 변수(global variable)로 인식되어 함수 내부에서도 쓸 수 있다. 하지만 경우에 따라 에러가 날 수도 있으므로 주의해야 한다. 함수 내부에서 어떤 위치건 `myvariable=value`처럼 **변수에 값을 넣는 코드가 있으면 그 변수는 모두 지역 변수로 인식**하게 된다. 어떤 변수(`myvariable`)을 지역 변수로 인식하는 함수가 있고 그 외부에도 `myvariable`이라는 전역 변수가 있을 때 지역 변수를 생성하기 전에 전역 변수로서 값을 읽어들이면 에러가 나게 된다. 함수 내부에서 전역 변수의 값을 바꾸고 싶다면 함수 내부에서 `global var_name`을 선언해줘야 한다. 아래 예시를 통해 확인해보자.

```python
global_var = 10
def add_ten_local():
    local_var = global_var + 10
    print("add_ten_local:", local_var)

def add_ten_global():
    try:
        global_var = global_var + 10
        print("add_ten_global:", global_var)
    except NameError as ne:
        print(ne)

def add_ten_global_two_steps():
    try:
        local_var = global_var + 10
        global_var = local_var
        print("add_ten_global_two_steps:", global_var)
    except NameError as ne:
        print(ne)

def add_ten_global_use_global():
    global global_var
    global_var = global_var + 10
    print("add_ten_global_use_global:", global_var)

add_ten_local()
add_ten_global()
add_ten_global_two_steps()
add_ten_global_use_global()
print("global_var=", global_var)
```



## 5. 바람직한 파이썬 코딩 스타일

파이썬 코딩을 할 때 보통은 지금까지 해온 것 처럼 함수밖에 바로 코드를 쓰진 않는다. 함수 밖에 생성된 변수들은 모두 전역 변수(global variable)로 인식되어 함수 내부 변수들과 섞여서 버그를 유발할 가능성이 높다. **따라서 모든 코드는 함수 안에 있어야한다.** 스크립트에서 실행될 메인 함수를 먼저 만들고 함수 밖에는 메인 함수를 호출하는 코드만 넣는다. 메인 함수 안에서는 코드를 기능별로 묶어서 세부 함수로 나누어 세부 함수들을 호출한다. 다음 예시를 따라해 보자.

```python
def main():
    print("function default arguments")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    avg = average_list_with_default(data)
    avg = average_list_with_default(data, 3)

def average_list_with_default(data, start=0, end=None, skip=None):
    if skip is None:
        skip = []
    return average_list(data, start, end, skip)

def average_list(data, start, end, skip):
    if end is None:
        avg_data = data[start:]
    else:
        avg_data = data[start:end]

    sum = 0
    skip_count = 0
    for ind, num in enumerate(avg_data):
        if ind in skip:
            skip_count += 1
        else:
            sum += num
    dlen = len(avg_data) - skip_count
    average = sum / dlen
    print(f"average {start}~{end}, skip={skip} over {len(data)} numbers = {average}")
    return average

if __name__ == '__main__':
    main()
```

맨 아래 `if __name__ == '__main__':` 이라는 조건문이 보이는데 이것은 이 스크립트 파일이 직접 실행이 된 것인지를 확인하는 것이다. 나중에 나오겠지만 여러 파일이 연계되서 실행될 때 이 스크립트가 다른 스크립트에서 사용되는 스크립트인지 주로 실행되는 스크립트인지를 확인하는 것이다. 주로 실행되는 스크립트인지를 확인하고 `main()` 함수를 실행한다. `main()` 함수는 첫 줄에 데이터를 선언하고 그 아래서 다른 함수들을 호출한다. **호출하는 함수는 호출 받는 함수 위에 있어야** 위에서부터 아래로 코드를 쉽게 읽을 수 있다. **신문처럼 자연스럽게 읽을 수 있는 코드가 좋은 코드다.**   

강의에서는 문법을 익히는데 필요한 코드만 보여주기 위해 함수 밖에 바로 코드를 쓰는 예시를 보여줄 수도 있지만 앞으로 숙제, 프로젝트 등을 할 때는 반드시 위와 같은 스타일로 스크립트를 작성해야 한다.

