---
layout: post
title:  "CH09 Overloading"
date:   2020-11-04 12:09:13
categories: C++
---



# CH09 타입 변환 연산자와 연산자 오버로딩

## 9_02 친구 지정

### 친구 지정이란?

객체 지향 언어의 클래스는 자신이 가지고 있는 멤버에 대해 **캡슐화**와 **정보 은닉성**을 보장해야 한다

-> 외부 함수나 클래스가 다른 클래스의 public 지정자로 선언되지 않은 멤버에 대해 접근할 수 없어야 한다

친구 지정(friend) 클래스는 서로 상속관계가 없는 클래스끼리 서로의 private 이나 protected 지정자로 선언된 멤버에 접근할 수 있게 해준다

> 만약 A클래스를 B클래스의 firend 클래스로 선언하면 A클래스 내부에서 B클래스의 모든 멤버에 접근할 수 있다
>
> 클래스 뿐만 아니라 함수도 friend로 지정할 수 있다.

- 선언 방법 : 

  ```cpp
  class B {
  private:
      ....
      friend class A;
  };
  
  class A {
      함수 선언(B& n) {
          ...
      }
  };
  ```

  

- 예제 01) : 

  ```cpp
  #include <iostream>
  #include <string>
  using namespace std;
   
  class GroupB {
  private :
      string name;
   
      friend class GroupA;
  };
   
  class GroupA{
  public : 
      void set_name(GroupB& f, string s) {
          f.name = s;
      }
      void show_name(GroupB& f) {
          cout << f.name << "\n";
      }
  };
   
  int main(void) {
      GroupB f1;
      GroupA f2;
      
      f2.set_name(f1, "유종섭");
      f2.show_name(f1);
   
      return 0;
  }
  ```

  실행 결과 : 

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex1.png)

  이 예제는 GroupB를 GroupA의 friend로 지정하여 GroupB의 모든 멤버를 GroupA에서 접근할 수 있다.

  위와 같이 friend 지정자는 한 클래스가 다른 클래스에게 자신의 멤버를 공개하는 일방적인 개방이며 서로가 서로를 friend로 지정하면 양쪽 모두 자유롭게 멤버에 접근할 수 있다.

  > 반대로 말하면 현재 GroupA에서는 GroupB를 friend로 지정해주지 않았기 때문에 GroupB에서는 GroupA에 접근할 수 없다



친구 지정은 객체 지향의 원칙에 어긋나는 규칙이지만, **연산자 오버로딩의 기능을 지원하기 위해 불가분한 예외 규정이다**. (이러한 이유로 자바는 연산자 오버로딩 기능을 제공하지 않는다)

클래스 내부에서 친구로 선언된 함수는 컴파일 하는 과정에서 extern 키워드처럼 외부에서 선언된 함수처럼 취급된다.

> extern : 전역 변수를 선언할 때 사용할 수 있는 키워드로, 다른 파일에서도 해당 변수로 접근할 수 있게 해준다. (<-> static : 해당 파일에서만 접근할 수 있는 전역 변수로 선언한다)



### C++과 자바의 설계 관점 차이

자바의 경우 객체를 중심으로 접근 제한을 두고 있다(이를 **객체 접근 제한(Instance Access Control)**이라고 부른다). 따라서 두 개의 객체가 같은 클래스로부터 생성된 인스턴스라고 하더라도 private으로 선언된 변수와 함수의 접근이 제한된다.

C++언어는 설계 관점에서 접근 제한이 클래스 중심으로 되어 있다(이를 **클래스 접근 제한(Class Access Control)**이라고 부른다). 따라서 같은 클래스로부터 생성된 객체는 private으로 선언되어 있다 하더라도 선언과 무관하게 접근이 허용된다. 

```cpp
#include <iostream>
#include <cstring>
 
class Person {
private :
    std::string name;
    int age;

public:
    Person(const char* n, int a) {
        name = n;
        age = a;
    }
    
    bool compare_age(Person& p) {
        if(this->age < p.age) return false;
        return true;
    }
    
    const char *get_name() {
        return name.c_str();
    }
};
 
int main() {
    Person hong("홍길동", 23);
    Person lee("이순신", 53);
    
    if(hong.compare_age(lee)) {
        std::cout<< hong.get_name()<<"의 나이가 "<<lee.get_name()<<"보다 많다."<<std::endl;
    }
    else {
        std::cout<< lee.get_name()<<"의 나이가 "<<hong.get_name()<<"보다 많다."<<std::endl;
    }
    
    return 0;
}
```

결과 : 

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex2.png)

여기서 만약 아래와 같이 직접적으로 private으로 선언된 변수에 접근하게 되면 에러가 발생한다.

```cpp
if(lee.age < 18){
    std::cout << "미성년자이다." << std::endl;
}
```

C++언어의 이러한 특징은 연산자 오버로딩이나 함수가 어떤 규칙으로 동작하여야 할지 설계하는데 많은 도움이 된다.



### 친구 관계와 클래스 상속

친구와 클래스간의 관계에 있어 주의할 사항은 다음과 같다.

- 클래스간의 상속이 이루어진다고 하더라도 친구 사이의 관계까지 상속이 이루어지지 않는다.

  -> 친구 관계는 수평관계이다.

- 클래스와 클래스가 서로 친구라고 하더라도 친구의 친구인 클래스에 대해 접근은 허용되지 않는다.

  ![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex3.png)



## 9_03 연산자 오버로딩

연산자 오버로딩은 클래스 내부에서 연산자의 기능을 커스터마이징 할 수 있는 기능이다. 모든 연산자에 대해 연산자 오버로딩을 허용하면 클래스 구조 자체가 복잡해지며, 원래 연산자의 기능과 혼동이 될 수도 있다.

- 연산자 오버로딩 포맷 : 

  operator 키워드와 함께 연산자를 사용하여 정의한다.

  ```
  데이터타입 operator연산자{
      ...
  }
  ```

연산자 오버로딩은 일종의 함수이다. 아래는 sum이라는 함수를 만든것과 연산자 오버로딩을 사용한 것이 결과가 같음을 보이는 예제이다.

```cpp
#include <iostream>
#include <cstring>
 
class Class {
private:
    int x;
    
public:
    Class(int x_) {
        x = x_;
        std::cout<<"생성자 호출" << std::endl;
    }
    
    Class sum(Class& b) {
        std::cout<<"일반 함수 호출"<<std::endl;
        return Class(x + b.x);
    }
    
    Class operator+(Class& b) {
        std::cout<<"연산자 오버로딩 호출"<<std::endl;
        return Class(x + b.x);
    }
    
    void print() {
        std::cout << "출력 : "<<x<<"\n"<<std::endl;
    }
};

int main() {
    Class a(10), b(20);
    Class c = a.sum(b);
    Class d = a + b;
    
    std::cout << "a : ";
    a.print();
    std::cout << "b : ";
    b.print();
    std::cout << "c : ";
    c.print();
    std::cout << "d : ";
    d.print();
    return 0;
}
```

결과 :

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex4.png)



위의 예시는 자신의 a객체를 기준으로 같은 클래스 타입의 b라는 인수와의 연산 작업을 수행하는 함수이며, 이를 **멤버 연산자 오버로딩**(Member Operator Overloading)이라고 부른다. 이는 결과적으로 자기 자신과 같은 데이터 타입을 반환한다.

만약 다른 타입의 데이터를 반환하는 연산을 하고싶다면 어떻게 해야 할까? 이런 경우에는 **전역 연산자 오버로딩**(Global Operator Overloading 또는 non-member Operator Overloading)으로 제작해야 한다.

```cpp
Template<class T1, class T2>
bool operator==(const T &a, const T2 &b){....}
```



예시 : 

```cpp
#include <iostream>
#include <cstring>

class Class {
public:
    int x;
    Class(int x_) {
        x = x_;
        std::cout<<"생성자 호출" << std::endl;
    }
    
    Class sum(Class& b) {
        std::cout<<"일반 함수 호출"<<std::endl;
        return Class(x + b.x);
    }
    
    Class operator+(Class& b) {
        std::cout<<"연산자 오버로딩 호출"<<std::endl;
        return Class(x + b.x);
    }
    
    void print() {
        std::cout << "출력 : "<<x<<"\n"<<std::endl;
    }
};


bool operator>(Class& a, Class& b){
    if (a.x > b.x) return true;
    else return false;
}


int main() {
    Class a(10), b(20);
    Class c = a.sum(b);
    Class d = a + b;
    
    std::cout << "a : ";
    a.print();
    std::cout << "b : ";
    b.print();
    std::cout << "c : ";
    c.print();
    std::cout << "d : ";
    d.print();
    
    if(a > b){
        std::cout << a.x <<"가 " <<b.x<<"보다 크다"<<std::endl;
    }
    else{
        std::cout << a.x <<"가 " <<b.x<<"보다 작다"<<std::endl;
    }
    return 0;
}
```

결과 : 

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex5.png)



### 오버로딩이 가능한 연산자

다음 표에 속한 42개 연산자는 오버로딩이 가능하다.

| +    | -    | *    | /     | %      | ^        |
| ---- | ---- | ---- | ----- | ------ | -------- |
| &    | \|   | ~    | !     | ,      | =        |
| <    | >    | <=   | \>=   | ++     | --       |
| <<   | \>>  | ==   | !=    | &&     | \|\|     |
| +=   | -=   | /=   | %=    | ^=     | &=       |
| !=   | *=   | <<=  | \>>=  | []     | ()       |
| ->   | ->*  | new  | new[] | delete | delete[] |

단항 연산자와 이항 연산자 등의 연산자는 멤버 연산자 오버로딩과 전역 연산자 오버로딩으로 정의할 수 있는 반면, **대입 연산자('='), 함수 호출 연산자('()'), 배열 연산자('[]'), 포인터 멤버 선택 연산자('->')**는 오로지 **클래스의 멤버 연산자 오버로딩**으로만 정의가 가능하다.

연산자 오버로딩에 관한 권고사항은 다음과 같다.

- 단항 연산자는 클래스 멤버 연산자 오버로딩으로 구현한다.

- 이항 연산자 가운데 연산 결과가 객체에 반영되지 않는다면, 멤버 연산자 오버로딩으로도 가능하지만 되도록 전역 연산자 오버로딩으로 구현하는 것이 좋다.

- 클래스의 멤버 연산자 오버로딩은 new와 delete 연산자를 제외하고는 static 지정자를 사용하지 않는다.

  > new 연산자 오버로딩은 객체를 생성하기 이전에 힙 메모리를 할당받는 작업을 수행하므로 다른 연산자와 달리 정적으로 선언 해야 한다.



### 연산자 오버로딩의 3가지 기본 법칙

좋은 프로그램을 작성하기 위한 연산자 오버로딩에 관한 3가지 기본 법칙(Three Basic Rules)이 존재한다.

1. 연산자의 기능이 분명하지 않거나 **논쟁의 소지가 있다면, 해당 연산자는 오버로딩하지 않는다.** 

   이 경우 연산자 오버로딩보다 함수를 만들어서 사용하는 것이 좋다.

2. **연산자 오버로딩은 보편적인 상식을 기준으로 한다**. 

   +연산자 오버로딩을 하면서 - 연산자처럼 만들면 프로그램 구현에 혼란을 야기한다.

3. **연관된 연산자 모두를 오버로딩 해야 한다**. 

   +연산자를 오버로딩 했다면 -연산자를 포함하여 += 연산자 역시 오버로딩 해야 한다. 다른 예로, >연산자를 오버로딩 했다면 <연산자도 오버로딩 한다.



## 9_04 산술 연산자 오버로딩

다음과 같은 산술 연산자 오버로딩을 하는 방법을 알아보자.

- 단항 연산자
- 이항 연산자
- 전위 연산자
- 후위 연산자
- 대입 연산자
- 타입 변환 연산자



### 단항 연산자

단항 연산자중 +는 자기 자신을 참조로 return해주면 된다.

```cpp
Complex& operator+() {
    std::cout << " + 단항 연산자 오버로딩 호출" << std::endl;
    return *this;
}
```



단항 연산자중 -는 자기 자신의 멤버변수에 -1을 곱한 후 생성자를 새로 호출하여 return해주면 된다.

```cpp
Complex& operator-() {
    std::cout << " - 단항 연산자 오버로딩 호출" << std::endl;
    Complex complex(-real, -imaginary);
    return Complex;
}
```



### 이항 연산자

이항 연산자중 +/-는 다른 클래스 참조를 입력받고, 그 멤버변수와 자기 자신의 멤버변수를 연산하여 생성자를 호출한 후 return 한다.

```cpp
Complex operator+(const Complex& other) {
    std::cout << " + 이항 연산자 오버로딩 호출" << std::endl;
    return Complex(real+other.real, imaginary+other.imaginary);
}
Complex operator-(const Complex& other) {
    std::cout << " - 이항 연산자 오버로딩 호출" << std::endl;
    return Complex(real-other.real, imaginary-other.imaginary);
}
```



### 전위 연산자 오버로딩

전위 연산자는 멤버변수 왼쪽에 '++'나 '--'가 붙는것으로, 현재 멤버변수에 ++/--연산을 취한 후 새로운 메모리에 할당한다. 이 경우에는 이미 할당되어있는 자기 자신에게 ++/--연산을 진행하므로 생성자를 호출할 필요가 없고, 스스로를 참조로 return해주면 된다.

```cpp
Complex& operator++() {
    std::cout << " ++ 전위 연산자 오버로딩 호출" << std::endl;
    ++real;
    ++imaginary;
    return *this;
}

Complex& operator--() {
    std::cout << " -- 전위 연산자 오버로딩 호출" << std::endl;
    --real;
    --imaginary;
    return *this;
}
```



### 후위 연산자 오버로딩

후위 연산자는 멤버변수 오른쪽에 '++'나 '--'가 붙는 것으로, 현재 멤버변수를 새로운 메모리에 할당한 후 ++/--연산을 취한다. 이 경우에는 현재 멤버변수를 새로운 메모리에 할당을 먼저 해야하기 때문에 새로운 클래스 생성자를 호출한 후 ++/--연산을 진행하고 클래스 객체를 return해주면 된다.

`참고` : 연산자 오버로딩시 연산자만으로 전위와 후위를 구분할 수 있는 방법이 없기 때문에 후위 연산자에 인위적으로 int 타입의 인수를 넣어주는 방식으로 후위 연산자를 구분한다. 이는 하나의 약속이다.

```cpp
Complex operator++(int) {
    std::cout << " ++ 후위 연산자 오버로딩 호출" << std::endl;
    Complex complex(real, imaginary);
    ++real;
    ++imaginary;
    return complex;
}
    
Complex operator--(int) {
    std::cout << " -- 후위 연산자 오버로딩 호출" << std::endl;
    Complex complex(real, imaginary);
    --real;
    --imaginary;
    return complex;
}
```



### 대입 연산자 오버로딩

대입 연산자는 연산자를 기준으로 왼쪽이 lvalue, 오른쪽이 rvalue이다. 따라서 대입 연산자는 반드시 this 객체를 참조로 반환해 주어야 한다.

```cpp
Complex& operator=(const Complec& other) {
    if(this == &other) return *this;
    real = other.real;
    imaginary = other.imaginary;
    return *this;
}

Complex& operator+=(const Complex& other) {
    real += other.real;
    imaginary += other.imaginary;
    return *this;
}

Complex& operator-=(const Complex& other) {
    real -= other.real;
    imaginary -= other.imaginary;
    return *this;
}
```



### 타입 변환 연산자 오버로딩

타입 변환 연산자 오버로딩은 반환할 타입을 명시하지 않는다.

```cpp
operator double(){
    return real;
}
```



### 복소수 연산을 위한 산술 연산자 오버로딩 예시

산술 연산자 오버로딩의 예로, 복소수 연산자 오버로딩의 경우를 살펴보자.

예시 : 

```cpp
#include <iostream>
#include <string>
#include <cstring>
#include <memory>
#include <cmath>

class Complex {
public:
    double real;
    double imaginary;
    Complex(double real_) {
        real = real_;
        imaginary = 0.0;
    }
    
    Complex(double real_, double imaginary_) {
        real = real_;
        imaginary = imaginary_;
    }
    
    double get_real() {
        return real;
    }
    
    double get_imaginary() {
        return imaginary;
    }
    
    Complex get_sum(const double real_, const double imaginary_) {
        return Complex(real+real_, imaginary+imaginary_);
    }
    
    Complex operator+(const Complex& other) {
        std::cout << " + 이항 연산자 오버로딩 호출" << std::endl;
        return Complex(real+other.real, imaginary+other.imaginary);
    }
    
    Complex operator-(const Complex& other) {
        std::cout << " - 이항 연산자 오버로딩 호출" << std::endl;
        return Complex(real-other.real, imaginary-other.imaginary);
    }
    
    Complex operator*(const Complex& other) {
        std::cout << " * 이항 연산자 오버로딩 호출" << std::endl;
        return Complex((real * other.real) - (imaginary * other.imaginary),
                       (real * other.imaginary) + (imaginary * other.real));
    }
    
    Complex operator/(const Complex& other) {
        double r = ((real * other.real) + (imaginary * other.imaginary)) / 
                   ((other.real * other.real) + (other.imaginary * other.imaginary));
        double i = ((imaginary * other.real) - (real * other.imaginary)) / 
                   ((other.real * other.real) + (other.imaginary * other.imaginary));
        return Complex(r, i);
    }
    
    Complex& operator+() {
        std::cout << " + 단항 연산자 오버로딩 호출" << std::endl;
        return *this;
    }
    
    Complex operator-() {
        std::cout << " - 단항 연산자 오버로딩 호출" << std::endl;
        Complex new_complex(-real, -imaginary);
        return new_complex;
    }
    
    Complex& operator++() {
        std::cout << " ++ 전위 연산자 오버로딩 호출" << std::endl;
        ++real;
        ++imaginary;
        return *this;
    }
    
    Complex& operator--() {
        std::cout << " -- 전위 연산자 오버로딩 호출" << std::endl;
        --real;
        --imaginary;
        return *this;
    }
    
    Complex operator++(int) {
        std::cout << " ++ 후위 연산자 오버로딩 호출" << std::endl;
        Complex complex(real, imaginary);
        ++real;
        ++imaginary;
        return complex;
    }
    
    Complex operator--(int) {
        std::cout << " -- 후위 연산자 오버로딩 호출" << std::endl;
        Complex complex(real, imaginary);
        --real;
        --imaginary;
        return complex;
    }
    
    Complex& operator=(const Complex& other) {
        if(this == &other) return *this;
        real = other.real;
        imaginary = other.imaginary;
        return *this;
    }
    
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imaginary += other.imaginary;
        return *this;
    }
    
    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imaginary -= other.imaginary;
        return *this;
    }
    
    operator double(){
        return real;
    }
};



std::ostream& operator<<(std::ostream& os, const Complex& other) {
    os << '(' << other.real << "+" << other.imaginary << "i" << ')' ;
    return os;
}

int main() {
    Complex comp1(1.0, 2.0);
    Complex comp2(2.0, 3.9);
    Complex comp3 = comp1 + comp2;
    Complex mult = comp1 * comp2;
    Complex divi = comp2 / comp1;
    std::cout << "comp1 : " << comp1 << std::endl;
    std::cout << "comp2 : " << comp2 << std::endl;
    std::cout << "comp1 + comp2 : " << comp3 <<"\n" << std::endl;
    std::cout << "comp1 * comp2 : " << mult << "\n" << std::endl;
    std::cout << "comp2 / comp1 : " << divi << "\n" << std::endl;
    for(int i = 0; i < 3; i++){
        Complex comp4 = comp1++;
        std::cout << "comp1++ : " << comp1 << std::endl;
        std::cout << "comp4++ : " << comp4 << std::endl;
        Complex comp5 = ++comp2;
        std::cout << "++comp2 : " << comp2 <<std::endl;
        std::cout << "++comp5 : " << comp5 <<std::endl;
    }
    std::cout << "current comp1 : " << comp1 << std::endl;
    comp1 = comp2;
    std::cout << "after comp1 = comp2 : " << comp1 << std::endl;
    comp1 += comp2;
    std::cout << "comp1 += comp2 : " << comp1 << std::endl;
    double real = double(comp1);
    std::cout << "real part of comp1 : " << real << std::endl;
    return 0;
}
```

결과 :

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex6.png)



## 9_05 기타 연산자 오버로딩

### 배열 인텍스 연산자 오버로딩

배열 인텍스 연산자 오버로딩은 배열이나 컨테이너 클래스를 만들때 사용한다.

예시 : 

```cpp
#include<iostream>
#include<cassert>

template<typename Type>
class Array {
public:
    typedef std::size_t size_type;
    Array(size_type size) : _size(size) {
        _data = new Type[_size]();
    }
    
    ~Array() {
        if(_data) {
            delete[] _data;
            _size = 0;
        }
    }
    
    inline size_type size() const {
        return _size;
    }
    
    Type& operator[](size_type index) {
        assert(index >= 0 && index < _size);
        return _data[index];
    }
    
    const Type& operator[](size_type index) const {
        assert(index >= 0 && index < _size);
        return _data[index];
    }
    
private:
    Type* _data;
    size_type _size;
};

int main() {
    Array<double> array(4);
    array[0] = 2.3;
    array[1] = 1.1;
    array[2] = 4.2;
    array[3] = 17.32;
    
    for(size_t i = 0; i < array.size(); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

결과 : 

![](https://raw.githubusercontent.com/Jonsuff/jonnote/master/_posts/image/ex7.png)



배열 인텍스 연산자는 다음과 같은 두 가지 특징을 가지고 있다.

1. 위와 같이 _data 변수는 private으로 선언하였지만, 참조로 반환 시 데이터의 수정이 가능하다.

   접근 제어는 변수나 함수에 대한 직접적인 접근(또는 호출)의 기능 여부를 확인하는 특징을 제공한다. 하지만 간접적으로 public 접근 지정자를 통해 반환받은 멤버 변수의 참조는 언제든 수정이 가능하다. 즉 private으로 선언된 함수도 public으로 선언된 함수를 통해 참조나 포인터로 반환하면 호출이 가능하다는 것이다.

2. 배열 인텍스 연산자를 오버로딩할때 위처럼 const로 선언된 연산자(출력용 연산자)와 그렇지 않은 연산자(입력용 연산자) 두 개를 만들어 주어야 한다.

   이는 하나는 데이터의 수정이 이루어지는 입력 연산자 오버로딩이고, 하나는 데이터를 조회하거나 콘솔에 출력할때 호출하는 출력 연산자 오버로딩으로 구분을 한 것인데, 이렇게 하면 원치않는 상황에서 데이터가 변하는 일을 방지할 수 있다.



