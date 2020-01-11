---
layout: post
title:  "Docker with GUI Tutorial (Sep. 2019 Update)"
date:   2019-06-25 09:00:01
categories: tools

---

## Motivation

연구 때문에 [ORB-SLAM](https://github.com/raulmur/ORB_SLAM) 이라는 걸 돌려봐야 했는데 오래된 코드라 테스트 환경이 무려 Ubuntu 12.04 or 14.04로 되어있을 뿐만 아니라 구버전 우분투에서만 작동하는 ROS Hydro or Indigo까지 dependency로 잡혀있어서 **Ubuntu 14.04를 설치해야 하는 상황**이 되었다. 이거 하자고 옛날 우분투를 시스템에 새로 깐다거나 VirtualBox로 깔긴 싫고 이럴 때 쓰라고 만든 **Docker**를 처음 써보기로 했다. 그런데 쓰다보니 GUI도 돌려야해서 본 포스트는 크게 "도커 사용법"과 도커에서 "GUI 사용법" 크게 두 가지 내용을 다룬다.



## Docker 란?

무식한 내가 떠드는 것 보다는 이미 훌륭한 분들이 잘 정리한 블로그를 보는게 나을듯 하다. 특히 "초보를 위한 도커 안내서"는 기본 개념부터 사용법까지 상세하게 잘 정리되어 있으니 초보자라면 처음부터 다 읽어보면 좋다.~~(저걸 다 읽으면 GUI 빼곤 이 포스트를 읽을 필요가 없습니다;;)~~
- [도커란 무엇인가?(WHAT IS DOCKER?)](http://avilos.codes/infra-management/virtualization-platform/docker/what-is-docker/)
- [도커란 무엇인가? / Docker 컨테이너 / Docker 이미지 / LXC / 가상화](http://dev.youngkyu.kr/32)
- [초보를 위한 도커 안내서 - 도커란 무엇인가?](<https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html>)
- [초보를 위한 도커 안내서 - 설치하고 컨테이너 실행하기](<https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html>)
- [초보를 위한 도커 안내서 - 이미지 만들고 배포하기](<https://subicura.com/2017/02/10/docker-guide-for-beginners-create-image-and-deploy.html>)



한 줄 요약: 고정량의 하드디스크와 메모리를 할당하지 않고 컨테이너 안의 격리된 프로세스로 돌아가는 가상 머신   

핵심 개념

> 컨테이너: 컨테이너는 격리된 공간에서 프로세스가 동작하는 기술입니다. 가상화 기술의 하나지만 기존방식과는 차이가 있습니다. 기존의 가상화 방식은 주로 OS를 가상화하였습니다. 우리에게 익숙한 VMware나 VirtualBox같은 가상머신은 호스트 OS위에 게스트 OS 전체를 가상화하여 사용하는 방식입니다. 이 방식은 여러가지 OS를 가상화(리눅스에서 윈도우를 돌린다던가) 할 수 있고 비교적 사용법이 간단하지만 무겁고 느려서 운영환경에선 사용할 수 없었습니다. ... (중략)... 이를 개선하기 위해 프로세스를 격리 하는 방식이 등장합니다. 리눅스에서는 이 방식을 리눅스 컨테이너라고 하고 단순히 프로세스를 격리시키기 때문에 가볍고 빠르게 동작합니다. CPU나 메모리는 딱 프로세스가 필요한 만큼만 추가로 사용하고 성능적으로도 거어어어어의 손실이 없습니다.
> 
> 이미지: 이미지는 컨테이너 실행에 필요한 파일과 설정값등을 포함하고 있는 것으로 상태값을 가지지 않고 변하지 않습니다. 컨테이너는 이미지를 실행한 상태라고 볼 수 있고 추가되거나 변하는 값은 컨테이너에 저장됩니다. 같은 이미지에서 여러개의 컨테이너를 생성할 수 있고 컨테이너의 상태가 바뀌거나 컨테이너가 삭제되더라도 이미지는 변하지 않고 그대로 남아있습니다. ... (중략)... 새로운 서버가 추가되면 미리 만들어 놓은 이미지를 다운받고 컨테이너를 생성만 하면 됩니다. 한 서버에 여러개의 컨테이너를 실행할 수 있고, 수십, 수백, 수천대의 서버도 문제없습니다. [출처](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)
> 



### Docker의 장점

도커를 한동안 빡세게 써보니 차암~ 좋은 시스템이다. 

- 다수의 여러가지 운영체제를 효율적으로 동시에 돌릴 수 있다.
- 미리 CPU, storage, memory를 할당하지 않고 여러 시스템이 유연하게 자원을 나눠가진다. 
- 새로운 이미지나 컨테이너를 만드는게 매우 쉽고 빠르고 자원효율적이다.

동시 여러개의 웹서버를 띄우는데 많이 쓰인다지만 나한테는 ROS나 C++ 개발에 특히 유용해 보인다. 파이썬이야 가상환경으로 dependency를 쉽게 관리할 수 있지만 ROS나 C++ 라이브러리의 경우 어쩔수 없이 시스템에 설치해야 하는데 이런 저런 패키지를 설치하다보면 리눅스가 걸레가 되기 마련이라 반년만 지나도 포맷하고 싶은 욕구가 솟아오른다. 호스트 시스템에는 최소한의 패키지만 설치해 놓고 새로운 프로젝트를 시작할 때마다 전용 도커 이미지를 만들면 호스트 시스템을 깔끔하게 관리할 수 있다.



### Docker Image 상속과 공유

도커의 가장 멋진 점은 이미지의 상속과 공유 시스템이다. 내가 만약 Ubuntu 16.04 이미지에 내가 원하는 패키지를 설치한 `imageA`를 만들고자 할때 Ubuntu 16.04 설치부터 하지 않아도 된다. 이미 도커에서 (정확히는 Docker Hub) 제공하는 `Ubuntu 16.04` 이미지가 있고 나는 그 이미지를 다운로드 받은 후 이를 **상속**받아 거기부터 특정 패키지를 설치한 새로운 이미지 `imageA`를 만들수 있다. `imageA`에서 다른 패키지를 추가 설치한 새로운 이미지 `imageB`를 만들고자하면 `imageA`를 상속받아 추가 패키지만 설치하면 된다.  

즉 어떤 이미지를 만들던 바닥부터 설치할 필요없이 기존에 있는 이미지들을 검색 후 자신의 목적에 가장 가까운 이미지를 받아서 바로 쓰거나 아니면 거기에 세부설정을 추가하여 자신에게 꼭 맞는 이미지를 효율적으로 만들어낼 수 있다. 이때 반드시 필요한 것이 **도커 이미지들을 공유할 수 있는 Docker Hub** 시스템이다. <https://hub.docker.com/> 이곳에서 자신에게 필요한 기반 이미지를 찾으면 `docker pull`이란 명령어로 이미지를 다운받을 수 있다. 그리고 이를 상속하여 변경한 새로운 이미지를 Docker Hub에 공유할 수도 있다.  

도커 이미지에서 더욱 놀라운 점은 이미지의 설치과정을 기술한 `Dockerfile`만 공유해도 다른 곳에서 똑같은 이미지를 만들어낼 수 있다는 것이다. Docker Hub에 있는 이미지를 받을 때는 빌드된 이미지 자체를 받지만 github이나 개인간에 도커 이미지를 공유할 때는 이미지를 빌드할 수 있는 텍스트 파일인 `Dockerfile`만 받아서 직접 빌드하면 된다.  

이처럼 이미지의 상속과 공유시스템으로 원하는 이미지를 편리하게 만들어 낼 수 있으니 너도나도 도커를 쓰는것 같다. 다만 초기에 개념을 이해하고 도커 명령어와 `Dockerfile` 작성법을 익히려면 시간이 좀 필요하다.



## Docker 설치

설치 방법은 공식 홈페이지에 잘 나와있다.
[우분투 설치](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/)
[윈도우 설치](https://docs.docker.com/v17.09/docker-for-windows/install/)

내가 설치한 Ubuntu 18.04 기준으로 요약하면 다음과 같다.
```bash
# remove old versions
$ sudo apt remove docker*
$ sudo apt autoremove

# install required pakages
$ sudo apt update
$ sudo apt install apt-transport-https ca-certificates curl software-properties-common

# add docker's official GPG key
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# verify the key
$ sudo apt-key fingerprint 0EBFCD88
pub   4096R/0EBFCD88 2017-02-22
      Key fingerprint = 9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid                  Docker Release (CE deb) <docker@docker.com>
sub   4096R/F273FCD8 2017-02-22

# add repository
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
		$(lsb_release -cs) stable"
$ sudo apt update
$ sudo apt install docker-ce
```


