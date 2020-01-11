---
layout: post
title:  "Linux Commands"
date:   2019-09-03 09:00:13
categories: 2019-2-robotics
---



# 1. 리눅스의 기본 명령어들



## 사전약속

- box 안에서 각 줄의 앞에 붙는 기호의 의미

```
# 주석
(현재 경로)$ 명령어 입력
명령어 결과 출력
```

- 터미널 열기 단축키: alt + ctrl + t
- 디렉토리와 폴더는 동의어이나 커맨드 라인에서는 디렉토리가 조금 더 정확한 용어이다.
- 리눅스에서는 디렉토리도 일종의 특수한 파일이기 때문에 일반적으로 '파일'이라 함은 보통의 파일과 디렉토리를 통칭한다.  



## cd (change directory)
: 터미널에서 현재 경로를 변경하는 명령어, 윈도우 탐색기에서 클릭을 통해 경로를 이동하듯 CLI에서 경로를 이동할 수 있다.
> **cd [상대/절대경로]**
>
> 아래와 같은 폴더 구조가 있을 때   
> /home/ian  
> /usr/local/bin/  
>     /include/X11  

```bash
# 현재의 절대경로 확인
$ pwd
/home/ian

# 절대경로 활용해 이동 
~$ cd /usr/local
/usr/local$

# 사용자의 홈디렉토리로 이동
/usr/local$ cd ~
~$

# 다시 /usr/local로 이동
$ cd /usr/local
/usr/local$

# 상대경로: 하위 디렉토리인 bin으로 이동
/usr/local$ cd bin
/usr/local/bin$

# 상대경로: 다시 상위 디렉토리인 local로 이동
/usr/local/bin$ cd ..
/usr/local$

# 상대경로: 상위 디렉토리로 갔다가 그 아래 include/X11로 이동
/usr/local$ cd ../include/X11
/usr/include/X11$

# 상대경로: 상위 디렉토리로 두 번 이동
/usr/include/X11$ cd ../..
/usr$

# 사용자 홈 디렉토리를 상대경로를 이용하여 이동 (ian은 자신의 사용자명으로 대체)
/usr$ cd ../home/ian
~$
```



## ls (list)
: 현재 경로에 있는 파일을 확인하는 명령어  
> **ls [경로] [옵션]**

```bash
$ cd ~
# 파일 목록 출력
~$ ls
  Desktop    Downloads         Music     Public  Templates  workplace
  Documents  examples.desktop  Pictures  snap    Videos

# 자세한 정보 출력
~$ ls -l
# 파일타입+권한 링크수 사용자 파일크기 수정시간 이름
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Desktop
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Documents
  drwxr-xr-x 3 ian ian 4096  8월 14 16:54 Downloads
  -rw-r--r-- 1 ian ian 8980  8월 14 15:29 examples.desktop
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Music
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Pictures
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Public
  drwxr-xr-x 3 ian ian 4096  8월 14 17:02 snap
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Templates
  drwxr-xr-x 2 ian ian 4096  8월 14 15:48 Videos
  drwxrwxr-x 3 ian ian 4096  8월 14 17:04 workplace

# 숨김 파일까지 출력
~$ ls -a
  .              examples.desktop  .PyCharmCE2018.2
  ..             .gconf            snap
  .bash_history  .gnupg            .sudo_as_admin_successful
  .bash_logout   .ICEauthority     Templates
  .bashrc        .java             Videos
  .cache         .local            workplace
  .config        .mozilla          .Xauthority
  .dbus          Music             .xinputrc
  Desktop        Pictures          .xsession-errors
  .dmrc          .presage          .xsession-errors.old
  Documents      .profile
  Downloads      Public

# 숨김 파일까지 자세히 출력
~$ ls -al

# 특정 경로 안의 내용 출력
~$ ls /usr
  bin  games  include  lib  local  locale  sbin  share  src
```



## mkdir (make directory)
: 디렉토리를 만드는 명령어  
> **mkdir [디렉토리명]**

```bash
# workplace 라는 디렉토리 만들기
~$ mkdir workplace
~$ cd workplace
~/workplace$ cd ..

# 존재하는 폴더 다시 만들면 에러
~$ mkdir workplace
> mkdir: cannot create directory ‘workplace’: File exists

# 하위 폴더까지 폴더 경로 생성
~$ mkdir -p workplace/foo/bar

# -p 옵션을 줄 시 기존 폴더가 있어도 에러를 출력하지 않음
~$ mkdir -p workplace/foo/bar
~$ mkdir -p workplace
```



## rm (remove)

: 파일 삭제
> **rm [옵션] [파일명 or 패턴]**

```bash
# 임시 파일/디렉토리 생성
~$ cd workplace
~/workplace$ mkdir dirA dirB dirC
~/workplace$ touch fileA fileB fileC
~/workplace$ ls
dirA  dirB  dirC  fileA  fileB  fileC

# 파일 삭제
~/workplace$ rm fileA
~/workplace$ ls
dirA  dirB  dirC  fileB  fileC

# 패턴으로 파일 삭제
~/workplace$ rm file*
~/workplace$ ls
dirA  dirB  dirC

# 디렉토리 삭제: -r 옵션
~/workplace$ rm -r dirA
~/workplace$ ls
dirB  dirC

# 패턴으로 디렉토리 삭제: -r 옵션
~/workplace$ rm -r dir*
~/workplace$ ls
 

# 임시 파일/디렉토리 다시 생성
~$ cd workplace
~/workplace$ mkdir dirA dirB
~/workplace$ touch fileA fileB

# 현재 디렉토리 아래 모든 파일/디렉토리 삭제
~/workplace$ rm -r *
```



## cp (copy)

: 파일 복사 

> **cp [복사대상] [목적지]**

```bash
~$ cd ~/workplace
~/workplace$ rm -r *
# apple 이란 디렉토리 생성
~/workplace$ mkdir apple
# banann 란 파일 생성
~/workplace$ touch banana

# 파일을 다른 이름으로 복사: cp [파일] [파일]
~/workplace$ cp banana grape
~/workplace$ ls
apple  banana  grape

# 파일을 이름 그대로 다른 경로로 복사: cp [파일] [디렉토리 경로]
~/workplace$ cp banana apple 

# 파일을 다른 경로에 다른 이름으로 복사: cp [파일] [파일 경로]
~/workplace$ cp banana apple/banana2
~/workplace$ ls apple
banana  banana2

# 디렉토리를 새 이름으로 복사: cp -r [대상 디렉토리] [새 디렉토리]
~/workplace$ cp -r apple cherry
~/workplace$ ls
apple  banana  cherry  grape

# 디렉토리를 다른 경로 아래 복사: cp -r [대상 디렉토리] [기존 디렉토리 경로]
~/workplace$ cp -r apple cherry
# 기존의 apple을 덮어쓰거나 새로운 apple(1)을 만드는 것이 아니라 apple 아래에 cherry 복사
~/workplace$ ls
apple  banana  cherry  grape
~/workplace$ ls cherry
banana  banana2  cherry
```



## mv (move)

: 파일 이동
> **mv [이동대상] [목적지]**

```bash
~$ cd ~/workplace
~/workplace$ rm -r *
~/workplace$ mkdir apple
~/workplace$ touch banana
~/workplace$ ls
apple  banana

# 파일/디렉토리 이름 변경: mv [기존 파일명] [새 파일명]
~/workplace$ mv apple cherry
~/workplace$ mv banana grape
~/workplace$ ls
cherry  grape

# 파일/디렉토리 위치 이동: mv [파일명] [기존 디렉토리 경로]
# mv는 -r 옵션을 붙이지 않아도 폴더를 이동할 수 있다.
~/workplace$ mkdir apple
~/workplace$ mv grape cherry
~/workplace$ mv apple cherry
~/workplace$ ls cherry
apple  grape
```



## find

: 현재 디렉토리 아래 있는 모든 파일과 디렉토리를 검색

> **find [검색경로] [옵션] [입력인자]**

```bash
# 파일, 폴더 생성
~$ cd ~/workplace
~/workplace$ rm -r *
~/workplace$ mkdir -p apple/banana/cherry/foo
~/workplace$ mkdir -p grape/kiwi/foo
~/workplace$ touch apple/banana/cherry/foo/bar1
~/workplace$ touch grape/kiwi/foo/bar2
~/workplace$ ls
apple  grape

# foo 파일 검색
~/workplace$ find -name foo
./grape/kiwi/foo
./apple/banana/cherry/foo

# bar로 시작하는 파일 검색
~/workplace$ find -name bar*
./grape/kiwi/foo/bar2
./apple/banana/cherry/foo/bar1

# 검색 경로 지정
~/workspace$ find grape -name foo
grape/kiwi/foo

# 수정일자, 파일 크기 등으로 검색하는 옵션도 있으나 이름 검색이 가장 많이 쓰임
```



## which

: 터미널에서 명령을 내리다 보면 이 명령어의 실행파일이 어디있는지 궁금할 때가 있다. 혹은 같은 명령어의 여러가지 버전이 설치됐거나 같은 명령어가 여러 위치에 설치돼있어서 현재 쉘에서 실행되는 명령어가 어디서 실행되는지 알아야할 때도 있다. 이럴때 `which` 명령어를 통해 현재 쉘에서 해당 명령어의 실제 실행파일 경로를 확인할 수 있다. 

> which [command]

```bash
$ which ls
/bin/ls

$ which cp mv
/bin/cp
/bin/mv

# 현재 사용되고 있는 파이썬의 위치 확인
$ which python3
/usr/bin/python3

# 파이썬3의 세부 버전 확인
$ ls -l /usr/bin/python3
lrwxrwxrwx 1 root root 9  8월 13 17:20 /usr/bin/python3 -> python3.6

# 파이썬3의 다른 버전 확인
$ ls -l /usr/bin/python3*
```



## echo, cat

: 파일 내용 확인, 파일 생성, 파일 내용 추가

> 새 파일 생성하여 내용 입력, 기존 파일있다면 덮어쓰기  
> **echo or cat "내용" > 파일명**   
> 기존 파일 있다면 붙여쓰기, 없다면 새 파일 생성  
> **echo or cat "내용" >> 파일명**  

더 자세한 사용방법은 이곳 참조: <http://withcoding.com/109>

```bash
# 텍스트 출력
~$ cd ~/workplace
~/workplace$ echo hello world
hello world

# 환경 변수 출력
~/workplace$ echo $PATH
/home/ian/bin:/home/ian/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

# echo를 이용한 간단한 텍스트 파일 생성 >
~/workplace$ echo "hello sweetie!" > mango
# cat을 이용한 파일 내용 출력
~/workplace$ cat mango
hello sweetie!

# echo를 이용한 파일 내용 추가 >>
~/workplace$ echo "I like mango" >> mango
~/workplace$ cat mango
hello sweetie!
I like mango

# cat을 이용한 간단한 텍스트 파일 생성
~/workplace$ cat > apple
apple is sweet!
(ctrl+D로 편집모드 나가기)

# 파일 내용 확인
~/workplace$ cat apple
> apple is sweet!

# cat을 이용한 파일 내용 추가
~/workplace$ cat >> apple
but I prefer banana
(ctrl+D)

# 파일 내용 확인
~/workplace$ cat apple
> apple is sweet!
  but I prefer banana
```



## chmod

: 파일 권한 설정, 참고자료 <https://ko.wikipedia.org/wiki/Chmod>

리눅스에서 파일 권한 설정은 매우 중요하다. 인터넷에서 다운받은 파일이 실행이 안되는 경우가 많은데 이럴때 권한 설정을 해주면 실행이 된다. 명령어는 다음과 같이 사용한다.

> $ chmod [권한 옵션] [파일 or 패턴]

권한 옵션은 세 가지 요소로 구성된다. 이들을 조합하여 원하는 사용자에게 특정 권한을 지정할 수 있다.

- 레퍼런스

  | reference | 설명                   |
  | --------- | ---------------------- |
  | u         | 사용자 (파일의 소유자) |
  | g         | 사용자와 같은 그룹     |
  | o         | 다른 사람들            |
  | a         | 모든 사람              |

- 연산자

  | operator | 설명                             |
  | -------- | -------------------------------- |
  | +        | "reference"에 "mode" 권한 추가   |
  | -        | "reference"에서 "mode" 권한 삭제 |
  | =        | "reference"에 "mode" 권한 지정   |

- mode

  | mode | 설명      |
  | ---- | --------- |
  | r    | 쓰기 권한 |
  | w    | 읽기 권한 |
  | x    | 실행 권한 |

상세 사용법은 예시를 통해 알아보자.

```bash
~$ cd ~/workplace
~/workplace$ rm -r *
# 파일 생성하고 권한 확인
~/workplace$ echo "hello apple" > apple
~/workplace$ ls -l
-rw-rw-r-- 1 ian ian 12  8월 14 23:01 apple
# => 사용자는 읽기/쓰기 권한 있음

# 사용자의 읽기/쓰기 권한 제거
~/workplace$ chmod u-rw apple
~/workplace$ ls -l
----rw-r-- 1 ian ian 12  8월 14 23:01 apple

# 읽기 시도 실패
~/workplace$ cat apple
cat: apple: Permission denied
# 쓰기 시도 실패 
~/workplace$ echo "apple is not fruit" >> apple
bash: apple: Permission denied

# 사용자의 읽기/쓰기 권한 복원
~/workplace$ chmod u+rw apple
~/workplace$ cat apple
hello apple
~/workplace$ echo "apple is not fruit" >> apple
~/workplace$ cat apple
hello apple
apple is not fruit

# 현재 폴더의 모든 파일, 디렉토리, 하위 디렉토리까지 재귀적(recursively)으로 권한설정
~/workplace$ chmod -R u+rw *
```



## Shell Script

이제까지 배운 명령어들을 연결해서 한번에 실행시킬 수 있는 쉘 스크립트를 작성해보자. 쉘 스크립트가 별게 아니고 터미널에서 치는 명령어들을 파일에 미리 써놨다가 한번에 실행시키는 것이다. 좀 더 깊이 들어가면 for 나 if 같은 제어문을 활용한 프로그램도 짤 수 있다. 쉘 스크립트는 리눅스 부팅과정에서도 많이 사용되고 있다. 대표적으로 `~/.bashrc`나 `~/.profile` 등이 있다.

```bash
$ gedit myscript.sh

# 스크립트 작성

ls
mkdir apple
echo "I love banana" > banana
cp banana apple
find -name "bana*"

# 스크립트 저장 후 닫기

# 스크립트가 실행 가능하도록 권한 설정
$ chmod a+x myscript.sh
# 스크립트 실행
$ ./myscript.sh
```



# 2. Package Management

데비안 계열의 리눅스에서는 **apt**란 명령어로 패키지를 관리할 수 있다. Ubuntu 14까지는 **apt-** 로 시작하는 여러 명령어들이 있었는데 (apt-get, apt-cache 등) 이들 중에서 자주 쓰는 기능만 모아 **apt** 명령어 하나로 대부분의 기능을 할 수 있게 만들었다. 물론 기존의 **apt-** 로 시작하는 명령어들도 남아있다. 



## 패키지 업데이트, 패키지 목록보기

```bash
# 저장소 패키지 목록 업데이트
$ sudo apt update

# 설치된 패키지 최신 버전 업그레이드
$ sudo apt upgrade

# 설치된 패키지 목록
$ sudo apt list --installed

# 업그레이드 가능한 패키지 목록
$ sudo apt list --upgradable
```



## 패키지 검색

```bash
# 패키지 저장소의 패키지 검색
# sudo apt search [package name or pattern]
$ sudo apt search unity-tweak-*

# 설치된 패키지 검색
# sudo apt list --installed [package name or pattern]
$ sudo apt list --installed gedit*
Listing... Done
gedit/bionic-updates,now 3.28.1-1ubuntu1.2 amd64 [installed,automatic]
gedit-common/bionic-updates,bionic-updates,now 3.28.1-1ubuntu1.2 all [installed,automatic]
```



## 패키지 설치 및 삭제

```bash
# 패키지 설치 (예시: unity-tweak-tool)
# sudo apt install [package names]
$ sudo apt install python3-dev python3-pip python3-numpy

# 패키지 삭제
# sudo apt remove [package names]
$ sudo apt remove python3-numpy

# 패키지 완전 삭제 (설정파일까지 삭제)
# sudo apt purge [package names]
$ sudo apt purge python3-numpy

# 패키지 삭제 후에는 쓰이지 않게된 의존 패키지들을 삭제한다.
$ sudo apt autoremove
```



## 저장소 추가

우분투를 설치하면 기본적으로 우분투의 저장소의 패키지만 검색 가능하다. 하지만 저장소 목록에 PPA (Personal Package Archive)를 추가하면 사설 저장소의 패키지도 설치할 수 있다. 일반적으로 우분투 저장소의 패키지는 안정(stable) 버전의 패키지만 올라오기 때문에 해당 소프트웨어의 최신 버전이 아닌 경우가 많다. 혹은 호환성 문제로 구버전의 리눅스에는 일부러 낮은 버전의 패키지만 올려놓기도 한다. 그래서 패키지가 기본 저장소에 있더라도 최신버전을 쓰고 싶거나 기본 저장소에 없는 패키지를 설치하고 싶을 때는 해당 패키지의 개발자의 PPA를 추가하여 그곳의 최신 패키지를 설치한다.  

**Note**: Ubuntu 16까지는 `add-apt-repository`를 실행한 후 `apt update`를 해줘야 PPA의 패키지 목록을 받을 수 있었지만 Ubuntu 18부터는 `add-apt-repository`를 실행하면 자동으로 `apt update`까지 실행한다.

```bash
# atom editor 설치 예시 (기본 저장소에 없는 패키지)
$ sudo add-apt-repository ppa:webupd8team/atom
$ sudo apt install atom

# python 3.7 설치 (기본 저장소 버전은 3.6)
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt search python3.*-dev

# 추가한 저장소 삭제
$ sudo add-apt-repository --remove ppa:deadsnakes/ppa
```


