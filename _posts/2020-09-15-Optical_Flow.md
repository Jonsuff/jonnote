

# Optical Flow

- Optical Flow란 두개의 연속된 비디오 프레임 사이에 이미지 객체의 동작 패턴을 말한다. 이때 객체의 움직임 패턴은 객체 자체가 움직이는 경우와 카메라가 움직이는 경우가 있다.
- 구현에 핵심이 되는 함수 : from OpenCV, cv2.calcOpticalFlowPyrLK()



### 응용 가능한 기술

- 움직읨을 통한 구조 분석
- 비디오 압축
- **Video Stabilization** : 영상이 흔들렸거나 블러가 된 경우 깨끗한 영상으로 처리하는 기술



### 구현 방법

- Optical Flow를 구현하기 위해서는 다음과 같은 가정이 필요하다.
  1. 객체의 픽셀 intensity는 연속된 프레임 속에서 변하지 않는다.
  2. 이웃한 픽셀은 비슷한 움직임을 보인다.
- cv2.calcOpticalFlowPyLK() : Lucas-Kanade 방법을 이용한 Optical Flow 계산을 제공한다. Optical Flow의 구현 로직은 다음과 같다.
  1. 비디오 이미지에서 추적할 포인트를 결정하기 위해 cv2.goodFeaturesToTrack()을 이용한다.
  2. 비디오에서 첫 번째 프레임을 취하고 Shi-Tomasi 코너 검출을 수행한다. 그 후 이들 점에 대해 Lucas-Kanade Optical Flow를 이용해 점들을 추적한다.
  3. cv2.calcOpticalFlowPyrLK()의 인자로 이전 프레임, 이전 검출 포인트들, 그리고 다음 프레임을 전달한다.

