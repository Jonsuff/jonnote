---
layout: post
title:  "파이썬- 변수와 list - by 김지은"
date:   2020-06-09 10:20:13
categories: Nalgaezit-study
---



## 1. 파이썬 변수와 타입

> 파이썬 변수는 **동적 타입**이다.
>





## 2. 숫자 타입

> 파이썬에서 숫자를 나타내는 타입

- int (정수)

  $$\rightarrow$$ 메모리 크기를 지정하지 않고 필요에 따라 늘릴수 있기 때문에 범위가 없다.

  

- float (실수)





## 3. 조건문과 비교 연산자

### 조건문

- 들여쓰기(공백 4칸) 또는 탭으로 블럭 구분

  

- 새로운 블럭은 : (colon) 기호 아래 들여쓰기 후 시작

  

- 두번째 조건이 있을 시 else if가 아닌 elif를 사용



### 비교 연산자

- is, and, or, in 연산자들이 있다.

  

- 연산자는 두 객체가 동일한 객체인지 비교, 상수 비교에서는 ==과 비슷하게 기능

  

- and , or는 각각 C언어의 &&, ||에 해당





## 4. 문자열

> "  "이나 '  '사이에 글자를 쓰면 된다.
>

### **문자열 연산**

- +를 쓰면 문자열을 이어 붙일 수 있고 *를 쓰면 문자열을 반복할 수 있다.



### **문자열 슬라이싱**

- 시작 인덱스 <= i < 끝 인덱스` 범위의 문자열을 가져온다.

  ex)  str[2:5]이면 str변수에서 2~4번째에서 문자를 가져오는 것이다.



### **문자열 포매팅**

- C언어식 포매팅 : %`를 쓴 후 값이나 변수명을 쓰면 그 값이 포맷 코드로 들어간다.

  문자열은  %s, 정수형은 %d, 실수형은 %f를 사용한다. (비추)

  

- 파이썬 포매팅 : print("값 : {}", 변수) 처럼 사용한다. 만일 출력하고 싶은 변수가 많으면 변수가 출력될 {}를 쓰고 , 뒤에 순서대로 변수를 적는다. (출력 변수가 많아지면 순서가 혼란스러울 수 있음)

  ex) print("a값 : {}, b값 : {}", a, b)

  

- f열 포매팅 : print(f"{변수}") 처럼 사용한다. {}내부에 변수를 바로 넣어주면 된다. 

  (가장 직관적. 추천!)



### **문자열 함수**

- count() 함수 : 특정 문자열의 포함 횟수를 셀 수 있다.

  ```python
  text = "How many e's in this sentence ?"
  count_e = text.count("e")
  print(f"There are {count_e} e's in the sentence !")
  ```

  ```
  결과 : There are 4 e's in the sentence !
  ```

  

- replace() :  문자열 안의 특정 문자열을 다른 문자열로 교체하는 함수이다.

  ```python
  text = "hello"
  replaced = text.replace("h", "b")
  print(f"replaced text : {replaced}")
  ```

  ```
  결과 : replaced text : bello
  ```






## 5. List

> 여러개의 데이터를 목록(list)처럼 담아둘 수 있는 자료형이다
>

### **리스트 관련 함수**

- len()  함수는 객체들의 길이를 잴 수 있다.

  ```python
  text = "How many words in this sentence?"
  print(f"There are {len(text)} words in the sentence")
  ```

  ```
  결과 : There are 32 words in the sentence
  ```

  

- del함수는 특정 원소를 삭제할 때 쓰인다. del()이 아니다

  ```python
  list_alphabet = ['a', 'b', 'c', 'd', 'e']
  del list_alphabet[4]
  print(f"e deleted list : {list_alphabet}")
  ```

  ```
  결과 : e deleted list : ['a', 'b', 'c', 'd']
  ```

  

- join()함수는 리스트의 내부 문자열 원소들을 첨가될 문자와 함께 하나의 문자열로 연결해준다. 

  ```python
  hello_list = ['h', 'e', 'l', 'l', 'o']
  hello_joined = "".join(hello_list)
  print(hello_joined)
  ```

  ```
  결과 : hello
  ```

  

- in은 함수가 아니라 operator다. 리스트에 특정 원소가 들어있는지 확인할 때 사용한다.

  ```python
  list_alphabet = ['a', 'b', 'c', 'd', 'e']
  if 'h' in list_alphabet:
      print("There is 'h' in the list")
  else:
      print("There is no 'h' in the list")
  ```

  ```
결과 : There is 'h' in the list
  ```
  
  

### **리스트 내장 함수**

- sort() : 크기순, 알파벳순 정렬 

  

- append() : 원소 추가

  

- insert() : 중간에 삽입

  

- index() : 특정 인덱스(위치) 반환



### **반복문과 리스트**

- for로 시작하는 줄은 : 로 끝나고, 반복되는 블럭은 들여쓰기로 구분한다.

  

- in을 기준으로 앞에는 원소 변수를 쓰고 뒤에는 반복가능한 객체를 쓴다. 

