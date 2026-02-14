# 직원 퇴사 예측 (Employee Attrition Prediction)

IBM HR Analytics 데이터셋을 활용한 직원 퇴사 예측 분석 프로젝트입니다.

## 프로젝트 개요

직원 퇴사는 기업에 큰 비용을 초래합니다. 이 프로젝트는 머신러닝을 활용하여 퇴사 가능성이 높은 직원을 사전에 식별하고, 퇴사의 핵심 요인을 분석하여 실행 가능한 리텐션 전략을 제안합니다.

## 데이터셋

- **출처**: [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)
- **규모**: 1,470명 직원 x 35개 변수
- **타겟 변수**: Attrition (퇴사 여부, Yes/No)

## 분석 프로세스

| 단계 | 노트북 | 내용 |
|------|--------|------|
| 1. EDA | `01_eda.ipynb` | 데이터 탐색, 분포 분석, 퇴사자 특성 파악 |
| 2. 전처리 | `02_preprocessing.ipynb` | 피처 인코딩, 스케일링, 피처 엔지니어링 |
| 3. 모델링 | `03_modeling.ipynb` | 다중 모델 학습, 교차검증, 성능 비교 |
| 4. 인사이트 | `04_insights.ipynb` | 피처 중요도 분석, 비즈니스 제언 |

## 기술 스택

- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **모델링**: scikit-learn, XGBoost, LightGBM
- **환경**: Python 3, Jupyter Notebook

## 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/<username>/attrition-prediction.git
cd attrition-prediction

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 데이터셋 다운로드 (Kaggle API 필요)
kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/raw/ --unzip

# 5. Jupyter Notebook 실행
jupyter notebook
```

## 프로젝트 구조

```
attrition-prediction/
├── data/
│   ├── raw/                   # 원본 데이터셋
│   └── processed/             # 전처리된 데이터
├── notebooks/
│   ├── 01_eda.ipynb           # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb # 데이터 전처리
│   ├── 03_modeling.ipynb      # 모델 학습 & 평가
│   └── 04_insights.ipynb      # 비즈니스 인사이트
├── src/
│   ├── data_loader.py         # 데이터 로딩 유틸리티
│   ├── preprocessing.py       # 전처리 함수
│   └── evaluation.py          # 모델 평가 함수
├── outputs/
│   ├── figures/               # 시각화 이미지
│   └── models/                # 학습된 모델
├── requirements.txt
└── README.md
```
