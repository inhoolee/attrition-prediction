# 직원 퇴사 예측 (Employee Attrition Prediction)

IBM HR Analytics 데이터셋을 활용한 직원 퇴사 예측 분석 프로젝트입니다.

## 프로젝트 개요

직원 퇴사는 기업에 큰 비용을 초래합니다. 이 프로젝트는 머신러닝을 활용하여 퇴사 가능성이 높은 직원을 사전에 식별하고, 퇴사의 핵심 요인을 분석하여 실행 가능한 리텐션 전략을 제안합니다.

## 정의

### 퇴사 기준

- 계약직 종료 포함?
- 자발적 퇴사 / 비자발적 퇴사 구분

### 예측 시점

- 징후가 인지되었을 때 액션이 가능한 시점

#### 연차에 따른 구분

- 1년차 이내
- 2~3년차
- 3년차 이상
- 10년차 이상

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
- **환경**: Python 3, Jupyter Notebook, uv, Jupytext

## 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/<username>/attrition-prediction.git
cd attrition-prediction

# 2. uv 설치 (미설치 시)
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 가상환경 생성 + 의존성 설치
uv sync

# 4. 데이터셋 다운로드 (Kaggle API 필요)
kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/raw/ --unzip

# 5. Jupyter Notebook 실행
uv run jupyter notebook

# 6. 노트북(.ipynb <-> .py) 동기화
uv run jupytext --sync notebooks/*.ipynb
```

## 프로젝트 구조

```
attrition-prediction/
├── data/
│   ├── raw/                   # 원본 데이터셋
│   └── processed/             # 전처리된 데이터
├── notebooks/
│   ├── 01_eda.ipynb           # 탐색적 데이터 분석
│   ├── 01_eda.py              # Jupytext pair (py:percent)
│   ├── 02_preprocessing.ipynb # 데이터 전처리
│   ├── 02_preprocessing.py    # Jupytext pair (py:percent)
│   ├── 03_modeling.ipynb      # 모델 학습 & 평가
│   ├── 03_modeling.py         # Jupytext pair (py:percent)
│   ├── 04_insights.ipynb      # 비즈니스 인사이트
│   └── 04_insights.py         # Jupytext pair (py:percent)
├── src/
│   ├── data_loader.py         # 데이터 로딩 유틸리티
│   ├── preprocessing.py       # 전처리 함수
│   └── evaluation.py          # 모델 평가 함수
├── outputs/
│   ├── figures/               # 시각화 이미지
│   └── models/                # 학습된 모델
├── pyproject.toml             # uv 의존성/프로젝트 설정
├── uv.lock                    # uv lockfile (uv sync 시 생성)
└── README.md
```
