# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02. 데이터 전처리 & 피처 엔지니어링
#
# EDA에서 파악한 내용을 바탕으로 모델 학습에 적합한 형태로 데이터를 변환한다.
#
# **전처리 파이프라인:**
# 1. 상수/ID 컬럼 제거
# 2. 타겟 변수 이진 변환
# 3. 범주형 변수 인코딩 (이진 → Label, 다범주 → One-Hot)
# 4. 피처 엔지니어링 (도메인 기반 파생 변수)
# 5. 다중공선성 처리
# 6. 학습/테스트 분할 (Stratified)
# 7. 수치형 변수 스케일링 (학습 데이터 기준 fit)
# 8. 전처리된 데이터 저장

# %%
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data_loader import load_raw_data, PROCESSED_DIR
from src.plot_style import apply_plot_style

apply_plot_style(figsize=(12, 6))

# %matplotlib inline

# %% [markdown]
# ## 2.1 데이터 로드 및 불필요한 컬럼 제거

# %%
df = load_raw_data()
print(f"원본 데이터: {df.shape}")

# 상수 컬럼 + ID 컬럼 제거
drop_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df = df.drop(columns=drop_cols)
print(f"제거 후: {df.shape}")
print(f"제거된 컬럼: {drop_cols}")

# %% [markdown]
# ## 2.2 결측치 확인

# %%
missing = df.isnull().sum()
print(f"총 결측치: {missing.sum()}")
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("→ 결측치 없음. 별도 처리 불필요.")

# %% [markdown]
# ## 2.3 타겟 변수 변환
#
# Attrition (Yes/No) → 이진값 (1/0)으로 변환한다.

# %%
df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
print(f"Attrition 분포:\n{df['Attrition'].value_counts()}")
print(f"\n퇴사율: {df['Attrition'].mean():.1%}")

# %% [markdown]
# ## 2.4 범주형 변수 인코딩
#
# 변수 특성에 따라 인코딩 방식을 구분한다:
# - **이진 변수** (2개 값): Label Encoding (0/1)
# - **다범주 명목 변수** (순서 없음): One-Hot Encoding
# - **순서형 변수** (1~4 스케일): 원본 숫자 유지 (인코딩 불필요)

# %%
# 현재 남은 범주형(문자열) 컬럼 확인
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"범주형 컬럼 ({len(cat_cols)}개):")
for col in cat_cols:
    print(f"  {col}: {df[col].unique().tolist()}")

# %%
# 이진 변수 → Label Encoding
binary_cols = ['OverTime', 'Gender']
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
    print(f"{col}: {dict(zip(range(2), ['No/Female', 'Yes/Male']))}")

print(f"\n이진 인코딩 완료: {binary_cols}")

# %%
# 다범주 명목 변수 → One-Hot Encoding
nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

print(f"One-Hot 대상: {nominal_cols}")
print(f"인코딩 전 컬럼 수: {df.shape[1]}")

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

print(f"인코딩 후 컬럼 수: {df.shape[1]}")
print(f"\n생성된 더미 변수:")
for col in nominal_cols:
    dummies = [c for c in df.columns if c.startswith(col + '_')]
    print(f"  {col} → {dummies}")

# %% [markdown]
# ## 2.5 피처 엔지니어링
#
# EDA에서 발견한 인사이트를 바탕으로 도메인 기반 파생 변수를 생성한다.

# %%
# 1) 근속 대비 승진 속도: 근속연수 대비 마지막 승진까지의 기간
df['PromotionStagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

# 2) 직무 경력 대비 현재 역할 기간: 정체 여부
df['RoleStagnation'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)

# 3) 소득 수준 대비 직급: 같은 직급 내 상대적 보상
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)

# 4) 직장 만족도 종합 지수 (4개 만족도 평균)
satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction',
                     'RelationshipSatisfaction', 'WorkLifeBalance']
df['SatisfactionIndex'] = df[satisfaction_cols].mean(axis=1)

# 5) 이직 빈도: 총 경력 대비 회사 수
df['JobHopRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)

print("생성된 파생 변수:")
new_features = ['PromotionStagnation', 'RoleStagnation', 'IncomePerLevel',
                'SatisfactionIndex', 'JobHopRate']
for feat in new_features:
    print(f"  {feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}")

# %%
# 파생 변수와 퇴사의 상관관계 확인
print("파생 변수 × Attrition 상관계수:")
for feat in new_features:
    r = df[feat].corr(df['Attrition'])
    marker = " ◀" if abs(r) > 0.1 else ""
    print(f"  {feat:25s}: r = {r:+.3f}{marker}")

# %% [markdown]
# ## 2.6 다중공선성 처리
#
# EDA에서 확인된 고상관 쌍:
# - MonthlyIncome ↔ JobLevel (r=0.95) → **JobLevel 제거** (소득이 더 직관적)
# - YearsAtCompany와 높은 상관을 보이는 변수 다수 → 트리 모델은 강건하므로 유지하되, 선형 모델용은 별도 처리 가능

# %%
# JobLevel 제거 (MonthlyIncome과 r=0.95)
df = df.drop(columns=['JobLevel'])
print("JobLevel 제거 완료 (MonthlyIncome과 r=0.95, 사실상 중복)")
print(f"현재 컬럼 수: {df.shape[1]}")

# %% [markdown]
# ## 2.7 학습/테스트 데이터 분할
#
# - **Stratified Split**: 클래스 불균형(16:84)이므로 퇴사 비율을 유지
# - **비율**: 80% 학습 / 20% 테스트
# - **random_state 고정**: 재현성 보장

# %%
X = df.drop(columns=['Attrition'])
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터: {X_train.shape[0]}행 × {X_train.shape[1]}열")
print(f"테스트 데이터: {X_test.shape[0]}행 × {X_test.shape[1]}열")
print(f"\n학습 퇴사 비율: {y_train.mean():.1%}")
print(f"테스트 퇴사 비율: {y_test.mean():.1%}")
print("→ Stratified split으로 비율 유지 확인")

# %% [markdown]
# ## 2.8 수치형 변수 스케일링
#
# - **StandardScaler** (평균=0, 표준편차=1)를 사용
# - **학습 데이터에만 fit** → 테스트 데이터에는 transform만 적용 (데이터 누수 방지)
# - One-Hot 인코딩된 이진 변수(0/1)는 스케일링하지 않음

# %%
# 스케일링 대상: 연속형 수치 변수 (이진/더미 제외)
binary_encoded = ['OverTime', 'Gender']
dummy_cols = [c for c in X_train.columns if '_' in c and X_train[c].nunique() <= 2]
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobSatisfaction',
                'RelationshipSatisfaction', 'WorkLifeBalance', 'JobInvolvement',
                'PerformanceRating', 'StockOptionLevel']
exclude_from_scaling = set(binary_encoded + dummy_cols + ordinal_cols)

scale_cols = [c for c in X_train.columns if c not in exclude_from_scaling]

print(f"스케일링 대상 ({len(scale_cols)}개):")
for col in scale_cols:
    print(f"  {col}")
print(f"\n스케일링 제외 ({len(exclude_from_scaling)}개): 이진/더미/순서형 변수")

# %%
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

print("스케일링 완료 (학습 데이터 기준 fit → 테스트 데이터 transform)")
print(f"\n학습 데이터 스케일링 후 통계 (상위 5개):")
X_train_scaled[scale_cols[:5]].describe().round(2)

# %% [markdown]
# ## 2.9 전처리 결과 최종 확인

# %%
print("=" * 60)
print("전처리 파이프라인 요약")
print("=" * 60)
print(f"원본 데이터:          1,470행 × 35열")
print(f"상수/ID 제거 후:      1,470행 × 31열")
print(f"인코딩 후:            1,470행 × {X.shape[1]}열")
print(f"\n학습 데이터 (X_train): {X_train_scaled.shape}")
print(f"테스트 데이터 (X_test): {X_test_scaled.shape}")
print(f"타겟 (y_train):        {y_train.shape[0]}행, 퇴사율 {y_train.mean():.1%}")
print(f"타겟 (y_test):         {y_test.shape[0]}행, 퇴사율 {y_test.mean():.1%}")
print(f"\n처리 내역:")
print(f"  - 제거: EmployeeCount, StandardHours, Over18, EmployeeNumber, JobLevel")
print(f"  - 이진 인코딩: OverTime, Gender")
print(f"  - One-Hot: BusinessTravel, Department, EducationField, JobRole, MaritalStatus")
print(f"  - 파생 변수 5개: PromotionStagnation, RoleStagnation, IncomePerLevel, SatisfactionIndex, JobHopRate")
print(f"  - 스케일링: StandardScaler ({len(scale_cols)}개 연속형 변수)")
print(f"  - 분할: 80/20 Stratified Split")

# %%
# 피처 목록 출력
print(f"최종 피처 ({X_train_scaled.shape[1]}개):")
for i, col in enumerate(X_train_scaled.columns, 1):
    dtype_tag = "[S]" if col in scale_cols else "[B]" if col in exclude_from_scaling else "[?]"
    print(f"  {i:2d}. {dtype_tag} {col}")
print(f"\n[S]=스케일링, [B]=이진/순서형(원본 유지)")

# %% [markdown]
# ## 2.10 전처리된 데이터 저장
#
# 모델링 노트북에서 사용할 수 있도록 CSV로 저장한다.
# - 스케일링된 버전과 스케일링 안 된 버전 모두 저장 (트리 모델은 원본, 선형 모델은 스케일링 버전 사용)

# %%
import joblib

# 스케일링 안 된 버전 (트리 모델용)
train_raw = X_train.copy()
train_raw['Attrition'] = y_train.values
train_raw.to_csv(PROCESSED_DIR / 'train.csv', index=False)

test_raw = X_test.copy()
test_raw['Attrition'] = y_test.values
test_raw.to_csv(PROCESSED_DIR / 'test.csv', index=False)

# 스케일링 된 버전 (선형 모델용)
train_scaled = X_train_scaled.copy()
train_scaled['Attrition'] = y_train.values
train_scaled.to_csv(PROCESSED_DIR / 'train_scaled.csv', index=False)

test_scaled = X_test_scaled.copy()
test_scaled['Attrition'] = y_test.values
test_scaled.to_csv(PROCESSED_DIR / 'test_scaled.csv', index=False)

# 전체 데이터 (Tableau 대시보드용)
full_clean = pd.concat([train_raw, test_raw], axis=0)
full_clean.to_csv(PROCESSED_DIR / 'attrition_clean.csv', index=False)

# Scaler 저장
joblib.dump(scaler, PROCESSED_DIR / 'scaler.joblib')

# 피처 목록 저장
pd.Series(X_train.columns.tolist()).to_csv(PROCESSED_DIR / 'feature_names.csv', index=False, header=['feature'])

print("저장 완료:")
for f in ['train.csv', 'test.csv', 'train_scaled.csv', 'test_scaled.csv',
          'attrition_clean.csv', 'scaler.joblib', 'feature_names.csv']:
    path = PROCESSED_DIR / f
    size = path.stat().st_size / 1024
    print(f"  {f}: {size:.1f} KB")
