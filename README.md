# [SEG/DET-06] Pstage2Serving()

기술스택: AWS, Flask, Nginx, Pytorch, React, Typescript    
발표시간: 14:00 - 14:20   
발표트랙: TRACK 1   
캠퍼 ID 와 이름 : T1006_권태확, T1010_박경환, T1066_박경환, T1184_전주영, T1242_김상훈        
프로젝트(대회): SEG/DET

## 서비스 소개

---

<img src = "https://user-images.githubusercontent.com/54058621/124644782-9686a300-decd-11eb-9c20-d52af19b5cb9.gif" width = "50%" height = "50%">


- 사진 속 인물의 성별, 나이, 마스크 착용여부 등을 시각적으로 감별할 수 있는 서비스
- 배포 : https://mask-cv.netlify.app/ (현재 GPU 서버는 비용 문제로 닫혀있어 실시간 서비스를 이용할 수 없습니다.)
- 시연 영상 원본 (high frame): https://youtu.be/K9-53AvAGzk

## 기획 의도

---

- P STAGE를 통해 얻은 딥러닝 개발 경험을 실 서비스에 접목시켜보기 위해 사이드 프로젝트를 진행했습니다.
- P STAGE의 이미지 분류, 객체 인식, 의미 분할 태스크를 통합하여 웹서비스를 만들고자 했고, 이 서비스 구현에 필요한 역량들을 찾아보았습니다.
- 데이터셋 생성, 딥러닝 모델 구현, 프론트엔드, 백엔드 등 서빙에 필요한 모든 파이프라인을 팀 NJYS가 고민하고, 구현하고, 개선해보면서 엔지니어링 능력을 향상시키고자 했습니다.

## 팀 설명

---

- 팀 `NJYS`(No Job, Yes Stress)는 5개의 원석들이 `YJNS`(Yes Job, No Stress)가 되기 위해 결성한 팀입니다.
- 21년 2월 중순부터 시작해 `365개`의 코딩 문제, `5개`의 전공과목 공부를 통해 기본기를 쌓았고, 여기에 부스트캠프 AI TECH에서 배운 `딥러닝`을 통해 팀원 전부 뛰어난 `기본기`를 장착했습니다.

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-16__11.59.18.png" width = "60%">

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-16__12.00.18.png" width = "60%">

---

## 팀원 소개

---

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/2.png" width = "30%">

### **권태확**

- 뛰어난 모델 리서치 능력과 모델 라이브러리 활용 능력으로 빠르게 SOTA 모델 구현을 주로 담당.
- [우직하게 성장하는 개발자, 권태확입니다.](https://www.notion.so/136ba9b6ad474e389d313c931277d7c5)

팀원 한줄평
#TaskSolver#python_master
#갑.분.조(갑자기 분위기 조력자)#역시싸피
#태확을믿으면돼

---

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/3.png" width = "30%">

### **김규빈**

- 팀을 구성하고 프로젝트 방향 설정.
- 딥러닝 모델링에서의 디버깅을 주로 담당.
- [적극적인 플레이어, 김규빈입니다](https://www.notion.so/c6cdec8fbcd341cab804de1acdc5a905)

팀원 한줄평
#리더십#지치지않는남자#추진력왕
#확실함 #알고리즘강사

---

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/4.png" width = "30%">

### **김상훈**

- 프론트엔드 개발과 다양한 모델 서치 능력으로 학습과 UI를 담당.
- [다양한 시도를 좋아하는 개발자, 김상훈입니다.](https://www.notion.so/9551e05e9667404cb8bbfb006cc7aff9)

팀원 한줄평
#미친CS역량#진정한nerd#코딩하는철학자
#진정한융합인재#프론트_믿고맡긴다구

---

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/1.png" width = "30%">

### **박경환**

- 모델 백엔드 서버 구현 및 딥러닝 모델 디버깅 담당.
- [해결을 좋아하는 개발자, 박경환입니다.](https://www.notion.so/ca780e6a0fff46bd87a9f099a67de276)

팀원 한줄평
#미친러닝커브#묵묵히_맡은일을_진행(개잘핵) 
#온화한실력자#뭔가_많이_해오심(?!)
#백엔드_믿고맡긴다구

---

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/5.png" width = "30%">

### **전주영**

- 프론트엔드 개발, 딥러닝 파라미터 튜닝 실험 및 현 프로젝트 정리.

팀원의 한줄평
#노션왕#Practical리더십#다재다능
#의견충돌시_교통정리왕#센스킹
#벌써취뽀#올라운더능력자

---

# NJYS가 통합한 딥러닝 프로젝트

---

- P STAGE 1의 이미지 분류
- P STAGE 3의 이미지 감지 및 분할
- P STAGE 4의 이미지 분류 경량화

---
## Side Project: Mask-CV-App
### 마스크 착용 여부, 성별, 나이 정보 추출 프로젝트

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/KakaoTalk_Photo_2021-06-16-15-15-44.png" width = "30%">

- p stage 1 모델을 활용한 분류기

<img src ="%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/KakaoTalk_Image_2021-06-16-23-09-54.png" width = "30%">

- P stage 3 동안 학습한 내용을 통해 개발한 새로운 모델. Detection + Instance segmentation + classification.

### 기획 의도(Pstage2Serving())

---

- P STAGE를 통해 얻은 딥러닝 개발 경험을 실 서비스에 접목시켜보기 위해 사이드 프로젝트를 진행했습니다.
- P STAGE의 이미지 분류, 객체 인식, 의미 분할 태스크를 통합하여 웹서비스를 만들고자 했고, 이 서비스 구현에 필요한 역량들을 찾아보았습니다.
- 데이터셋 생성, 딥러닝 모델 구현, 프론트엔드, 백엔드 등 서빙에 필요한 모든 파이프라인을 팀 NJYS가 고민하고, 구현하고, 개선해보면서 엔지니어링 능력을 향상시키고자 했습니다.

### 프로젝트 이슈

---

- 프로젝트 방향성 이슈
    - 서비스를 만드는 과정에서 다양한 의견이 나와 프로젝트 방향성을 잡는데 시간이 소요됨.
    - 앱을 구현할 지 웹을 구현할 지, P STAGE 태스크로 서비스를 만들지, 아예 관련 없는 태스크로 서비스를 만들지 등 의견을 조율하는 과정에서 서로의 역량을 파악할 수 있었음.


- 프론트엔드 이슈
    - React, Typescript, Material-UI, D3.js, axios, React-Query 등 이용
    - 협업을 통해 컴포넌트를 어떻게 구성 할 지 결정하는 과정에서 더 좋은 구조가 무엇인지 깊게 고민할 수 있었음
    - 서버와 통신을 통해 JSON 형식의 데이터만 받고 이미지 위에 결과를 렌더링 하였음
    - Material-UI를 이용해 프로토타입때와 완전히 다른 UI 를 개발하였음.


- 백엔드 이슈
    - Heroku main memory 한계와 모델 구현 환경과 모델 서빙 환경에 대한 차이가 있었음
    - AWS로 변경했고 Nginx, gunicorn, flask를 통해 전체적인 웹서버를 구축하여 웹서버에 대한 이해도가 증가함.
    - Django와 flask를 전부 사용해봄으로써 python web framework에 대해 이해함.


- 데이터셋 이슈
    - P STAGE 1의 데이터셋에는 bounding box , mask area 정보가 없고, 마스크 착용 여부, 성별, 나이 정보만 존재함
    - bounding box와 mask area 정보를 생성하기 위해 외부데이터로 모델을 학습시킨 후 수도라벨링으로 데이터셋을 확장.

- 모델링 이슈
    - 구현하기 쉬운 모델 3개를 사용할 지 구현하기 어려운 모델 1개를 사용할 지 결정해야 했음
        - class분류 결과(마스크 착용 여부, 성별, 나이 총 12개 클래스), bbox에 대한 정보, mask의 segmentation 정보
    - Mask R-CNN을 구현하는 과정에서 Feature Extraction, Region Proposal Network, ROI Pooling, Classification Head, Bounding Box Head, Mask Segmentation Head에 대한 이해도가 올라감


- Response JSON 파일 구조
    - 결과 값의 좌표 정보만 서버에서 전송해주고 클라이언트에서 렌더링 하기 위해서 데이터 형식에 대한  클라이언트, 서버 담당의 소통이 중요하게 작용.


- 서비스 이슈
    - inference time이 GPU를 사용하면 5fps, CPU를 사용하면 0.3fps 가 소모되어 현재 시간을 줄이기 위한 방법 모색 중.

### Dataset

---

- P STAGE 1 dataset + Kaggle dataset

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-16__10.50.38.png" width = "30%">

\+

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-16__10.49.40.png" width = "30%">

→ NJYS dataset

<img src = "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-16__10.48.51.png" width = "80%">

### Tech Stack

---

#### Front-End

- React.js + Typescript + Material UI 및 여러 오픈소스 활용

    <img src ="https://user-images.githubusercontent.com/54058621/124643839-87532580-decc-11eb-9c8e-fd4fb46af4a7.png" width = "70%">

#### Deep Learning

- Mask R-CNN

    <img src= "https://miro.medium.com/max/3840/1*9jlM5QjbTt46gH91RWgzAQ.png" width = "70%">

#### Back-End

- Nginx + Gunicorn + Flask + Pytorch

    <img src= "%5BSEG%20DET-06%5D%20Pstage2Serving()%20e761bfaedce84c8d87bc224f68f4686f/_2021-06-15__11.27.48.png" width = "70%">

---

## Github Repository

- P stage 3 실험 코드 및 wrap-up report : [bcaitech1/p3-ims-obd-njys](https://github.com/bcaitech1/p3-ims-obd-njys)

- Side Project : [NJYS](https://github.com/NJYS/Mask-CV-App)
