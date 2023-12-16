# rokafnet
2023 국방 AI 경진대회(MAICON) 본선 대회 RokafNet 팀의 제출 코드입니다.

## 대회 상세
- 부문: 군 장병 부문
- 과제: **음성인식** 기술을 활용한 효과적인 지휘관의 전술 명령 전달   
  (**Automatic Speech Recognition**)
- Train Data: 음성(data), 전사 텍스트(label) (14000개)
- Test Data: 음성 (6000개)
- Performance measure: 1-CER (문장부호, 띄어쓰기 미포함. [0, 1] 구간 cliping)

## 팀원 소개 및 주요 역할
- 김진수: Data Preprocess 및 Train/Predict 코드 작성
- 박성준: EDA, 개별 data/label 심층 분석, ppt 제작
- 박진영: Data Postprocess 코드 작성, 모델/논문 검색
- 이민준: EDA, 모델/논문 검색, 환경 설정, 실험 관리
  
## 모델 설명
### Overview
ASR 분야 SOTA 모델이며, 한국어에 대한 성능이 뛰어나고 소음에 robust한 특성을 갖는 whisper를 모델로 선정하였다. 큰 소리의 인위적 소음이 합성된 data 특성에 주목하여, 전처리 과정에서는 train data의 noise를 활용한 denoising을 수행했다. 후처리 시에는 맞춤법 교정에 초점을 맞췄는데, 이때는 data의 label 종류가 한정적임을 적극적으로 활용하여 시간을 줄이고 CER을 0.1 가량 향상시켰다.

### Baseline
[seastar105/whisper-small-ko-zeroth](https://huggingface.co/seastar105/whisper-small-ko-zeroth)
- parameters: 244M

### Hyperparameter & Optimizers
- Batch size: 4
- Gradient accumulation steps: 8
- Optimizer: AdamW(lr: 1e-5)
- Scheduler: cosine schedule with warmup(warmup steps: 500)
- eval&save steps: 500
- metric: CER

### Data preprocess
- **Label cleaning**: 문장부호, 띄어쓰기 제거
- Remove outlier: 음성 파일 길이와 전사 텍스트 길이 간 비정상적 차이 존재하는 데이터 제거
- Filter data by length: Train data의 경우 30초 미만 데이터만 사용
- **Denoise**: train data 내 noise 추출 후 [noisereduce](https://github.com/timsainb/noisereduce/tree/master) library 활용하여 주어진 data에 최적화된 denoising 수행
- Create Dataset: huggingface 제공 feature extractor, tokenizer 이용 데이터 변환
  - feature extractor: audio(wav file) -> log-mel spectrogram 변환
  - tokenizer: label text를 tokenizing

### Data postprocess
- Remove duplicates: raw prediction에서 관측되는 텍스트 반복 제거
- **Proofreading**: train data의 label 이용 prediction 문법 교정

### LB Score
- Public: **0.8661**(1st)
- Private: **0.8636**(2nd)

[실험관리 페이지(Notion)](https://www.notion.so/f85c0389cb8e40b89bb4ac0c8c088c78)

## 파일 구조
```
(project)
+-- code
|   +-- preprocess.ipynb
|   +-- code.ipynb: Train/Predict code
|   +-- postprocess.ipynb
|   +-- main.ipynb: merged code
|   +-- main.py: terminal 내 실행을 위해 main.ipynb를 변환한 py 파일
+-- files (not uploaded)
|   +-- noise.wav: preprocess 시 사용되는 noise file (train data에서 추출)
+-- requirements.txt: 환경 파일
```

## 실행 방법
project directory 이동
```shell
# requirements.txt 내 library 설치
$ pip install -r requirements.txt
```
```shell
# code directory 이동 후 main.py 실행
$ cd code
$ python main.py
```
