# 퇴사 예측 프로젝트 (Attrition Prediction)

## 프로젝트 목적
HR 분야 취업용 포트폴리오 프로젝트. IBM HR Analytics 데이터셋을 활용하여 직원 퇴사를 예측하고, 비즈니스 관점의 인사이트를 도출한다.

## 기술 스택
- **언어**: Python 3
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **모델링**: scikit-learn, xgboost, lightgbm
- **환경**: Jupyter Notebook, uv (.venv), Jupytext

## 데이터셋
- **출처**: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- **파일**: `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **규모**: 1,470명 x 35개 컬럼
- **타겟**: `Attrition` (Yes/No, 불균형 비율 약 16:84)
- **주요 피처 카테고리**:
  - 인구통계: Age, Gender, MaritalStatus, DistanceFromHome
  - 직무: Department, JobRole, JobLevel, JobInvolvement
  - 보상: MonthlyIncome, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel
  - 만족도: EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance
  - 경력: TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, NumCompaniesWorked
  - 기타: OverTime, PerformanceRating, TrainingTimesLastYear
- **제거 대상 컬럼**: EmployeeCount(상수), StandardHours(상수), Over18(상수), EmployeeNumber(ID)

## 분석 파이프라인
1. **EDA** (`notebooks/01_eda.ipynb`): 데이터 탐색, 분포 확인, 타겟 변수 분석
2. **전처리** (`notebooks/02_preprocessing.ipynb`): 인코딩, 스케일링, 피처 엔지니어링
3. **모델링** (`notebooks/03_modeling.ipynb`): 다중 모델 학습, 교차검증, 하이퍼파라미터 튜닝
4. **인사이트** (`notebooks/04_insights.ipynb`): 피처 중요도, SHAP, 비즈니스 제언

## 코딩 컨벤션
- 노트북 마크다운 셀은 **한글**로 작성
- 코드 주석은 한글 허용, 변수/함수명은 영어
- 함수에는 docstring 포함 (한글 설명)
- 노트북 셀 구조: 마크다운 헤더 → 코드 셀 → 결과 해석 마크다운
- `src/` 모듈의 공통 함수를 노트북에서 import하여 재사용
- 시각화 저장: `outputs/figures/`에 PNG로 저장
- 노트북은 `.ipynb`와 `.py`(Jupytext `py:percent`) 페어로 관리
- 노트북 수정 후 `uv run jupytext --sync notebooks/*.ipynb`로 동기화

## 목표 독자
- HR 담당자 / 비기술 면접관
- 비즈니스 관점의 해석과 실행 가능한 제언을 강조
- 기술적 디테일보다 "왜 퇴사하는가", "어떻게 방지할 수 있는가"에 초점

## 프로젝트 구조
```
attrition-prediction/
├── data/raw/          # 원본 CSV (gitignore)
├── data/processed/    # 전처리된 데이터
├── notebooks/         # 분석 노트북 4개
├── src/               # 재사용 유틸리티 모듈
├── outputs/figures/   # 시각화 이미지
└── outputs/models/    # 학습된 모델 (gitignore)
```
