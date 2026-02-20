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
# # 03. 모델 학습 & 평가
#
# 여러 분류 모델을 학습하고 교차검증을 통해 성능을 비교한 뒤,
# 최적 모델을 선정하고 테스트 데이터로 최종 평가한다.
#
# **핵심 설계 결정:**
# - 평가 지표: **Recall과 F1** 중심 (퇴사자 누락 비용 > 오경보 비용)
# - 교차검증: **Stratified 5-Fold** (클래스 불균형 유지)
# - 불균형 처리: **class_weight='balanced'** 기본 적용

# %%
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from src.data_loader import PROCESSED_DIR
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.plot_style import apply_plot_style

apply_plot_style(figsize=(12, 6))

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %matplotlib inline

# %% [markdown]
# ## 3.1 전처리된 데이터 로드

# %%
# 트리 모델용 (원본 스케일)
train = pd.read_csv(PROCESSED_DIR / 'train.csv')
test = pd.read_csv(PROCESSED_DIR / 'test.csv')

X_train = train.drop(columns=['Attrition'])
y_train = train['Attrition']
X_test = test.drop(columns=['Attrition'])
y_test = test['Attrition']

# 선형 모델용 (스케일링)
train_s = pd.read_csv(PROCESSED_DIR / 'train_scaled.csv')
test_s = pd.read_csv(PROCESSED_DIR / 'test_scaled.csv')

X_train_s = train_s.drop(columns=['Attrition'])
y_train_s = train_s['Attrition']
X_test_s = test_s.drop(columns=['Attrition'])
y_test_s = test_s['Attrition']

print(f"학습: {X_train.shape}, 테스트: {X_test.shape}")
print(f"학습 퇴사율: {y_train.mean():.1%}, 테스트 퇴사율: {y_test.mean():.1%}")
print(f"피처 수: {X_train.shape[1]}")

# %% [markdown]
# ## 3.2 베이스라인 모델
#
# 모든 예측을 "재직(0)"으로 하는 더미 분류기의 성능을 확인한다.
# 이 수치가 **Accuracy의 하한선**이자, 모델이 넘어야 할 최소 기준이다.

# %%
# 더미 베이스라인: 전부 0(재직)으로 예측
y_dummy = np.zeros_like(y_test)
dummy_metrics = evaluate_model(y_test, y_dummy)

print("베이스라인 (전부 재직으로 예측):")
for k, v in dummy_metrics.items():
    print(f"  {k}: {v:.3f}")
print()
print(f"→ Accuracy {dummy_metrics['Accuracy']:.1%}이지만 Recall=0, F1=0")
print("→ 퇴사자를 단 한 명도 탐지하지 못함. 이것이 Accuracy만 보면 안 되는 이유.")

# %% [markdown]
# ## 3.3 다중 모델 교차검증 비교
#
# 6개 모델을 5-Fold Stratified CV로 비교한다.
# 모든 모델에 `class_weight='balanced'`를 적용하여 소수 클래스(퇴사)에 높은 가중치를 부여한다.

# %%
models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
        X_train_s, y_train_s  # 스케일링 데이터
    ),
    'Random Forest': (
        RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE),
        X_train, y_train
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        X_train, y_train  # GB는 class_weight 미지원, 이후 sample_weight로 처리
    ),
    'XGBoost': (
        XGBClassifier(
            n_estimators=200, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0
        ),
        X_train, y_train
    ),
    'LightGBM': (
        LGBMClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, verbosity=-1
        ),
        X_train, y_train
    ),
    'SVM (RBF)': (
        SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE),
        X_train_s, y_train_s  # 스케일링 데이터
    ),
}

scoring = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']

cv_results = {}
print(f"{'모델':<25s} {'Accuracy':>9s} {'Recall':>9s} {'Precision':>9s} {'F1':>9s} {'AUC-ROC':>9s}")
print("-" * 75)

for name, (model, X, y) in models.items():
    scores = cross_validate(model, X, y, cv=CV, scoring=scoring, n_jobs=-1)
    result = {metric: scores[f'test_{metric}'].mean() for metric in scoring}
    result_std = {metric: scores[f'test_{metric}'].std() for metric in scoring}
    cv_results[name] = {'mean': result, 'std': result_std}

    print(f"{name:<25s} "
          f"{result['accuracy']:>8.3f}  "
          f"{result['recall']:>8.3f}  "
          f"{result['precision']:>8.3f}  "
          f"{result['f1']:>8.3f}  "
          f"{result['roc_auc']:>8.3f}")

# %%
# CV 결과 시각화
metrics_to_plot = ['recall', 'f1', 'roc_auc', 'precision']
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

for i, metric in enumerate(metrics_to_plot):
    names = list(cv_results.keys())
    means = [cv_results[n]['mean'][metric] for n in names]
    stds = [cv_results[n]['std'][metric] for n in names]

    colors = ['#E74C3C' if m == max(means) else '#3498DB' for m in means]
    bars = axes[i].barh(names, means, xerr=stds, color=colors, alpha=0.8, capsize=3)
    axes[i].set_title(metric.upper(), fontsize=13, fontweight='bold')
    axes[i].set_xlim(0, 1.05)

    for bar, mean in zip(bars, means):
        axes[i].text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{mean:.3f}', va='center', fontsize=9)

plt.suptitle('모델별 교차검증 성능 비교 (빨강: 최고)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../outputs/figures/08_model_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3.4 하이퍼파라미터 튜닝
#
# CV 결과에서 상위 2개 모델(LightGBM, XGBoost)에 대해 GridSearchCV로 튜닝한다.
# - 평가 지표: **F1** (Recall과 Precision의 균형)
# - 리샘플링: Stratified 5-Fold

# %%
# LightGBM 튜닝
lgbm_params = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31],
    'min_child_samples': [10, 20, 30],
}

lgbm_grid = GridSearchCV(
    LGBMClassifier(class_weight='balanced', random_state=RANDOM_STATE, verbosity=-1),
    lgbm_params,
    scoring='f1',
    cv=CV,
    n_jobs=-1,
    verbose=0
)
lgbm_grid.fit(X_train, y_train)

print("LightGBM 최적 파라미터:")
for k, v in lgbm_grid.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Best F1 (CV): {lgbm_grid.best_score_:.4f}")

# %%
# XGBoost 튜닝
xgb_params = {
    'n_estimators': [200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
}

spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=spw, eval_metric='logloss',
                  random_state=RANDOM_STATE, verbosity=0),
    xgb_params,
    scoring='f1',
    cv=CV,
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_train, y_train)

print("XGBoost 최적 파라미터:")
for k, v in xgb_grid.best_params_.items():
    print(f"  {k}: {v}")
print(f"  Best F1 (CV): {xgb_grid.best_score_:.4f}")

# %%
# 튜닝 전후 비교
print(f"{'모델':<20s} {'튜닝 전 F1 (CV)':>16s} {'튜닝 후 F1 (CV)':>16s} {'개선':>8s}")
print("-" * 65)

lgbm_before = cv_results['LightGBM']['mean']['f1']
xgb_before = cv_results['XGBoost']['mean']['f1']

print(f"{'LightGBM':<20s} {lgbm_before:>15.4f}  {lgbm_grid.best_score_:>15.4f}  {lgbm_grid.best_score_ - lgbm_before:>+7.4f}")
print(f"{'XGBoost':<20s} {xgb_before:>15.4f}  {xgb_grid.best_score_:>15.4f}  {xgb_grid.best_score_ - xgb_before:>+7.4f}")

# 최종 모델 선택
if lgbm_grid.best_score_ >= xgb_grid.best_score_:
    best_model = lgbm_grid.best_estimator_
    best_name = 'LightGBM (tuned)'
else:
    best_model = xgb_grid.best_estimator_
    best_name = 'XGBoost (tuned)'

print(f"\n→ 최종 선택: {best_name} (F1={max(lgbm_grid.best_score_, xgb_grid.best_score_):.4f})")

# %% [markdown]
# ## 3.5 최종 모델 — 테스트 데이터 평가
#
# **중요**: 테스트 데이터는 이 시점에서 처음 사용된다.
# 교차검증은 학습 데이터 내에서만 수행했으므로, 테스트 성능은 일반화 성능의 추정치이다.

# %%
# 테스트 예측
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 평가 지표
test_metrics = evaluate_model(y_test, y_pred, y_prob)

print(f"최종 모델: {best_name}")
print("=" * 50)
for k, v in test_metrics.items():
    print(f"  {k:>12s}: {v:.4f}")

print(f"\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['재직', '퇴사']))

# %%
# 혼동 행렬
fig = plot_confusion_matrix(y_test, y_pred, title=f'{best_name} — 테스트 데이터')
plt.savefig('../outputs/figures/09_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 해석
tn, fp, fn, tp = __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix(y_test, y_pred).ravel()
print(f"TN(재직→재직): {tn}  |  FP(재직→퇴사 오경보): {fp}")
print(f"FN(퇴사→재직 누락): {fn}  |  TP(퇴사→퇴사 탐지): {tp}")
print(f"\n→ {tp}명의 퇴사자를 사전에 탐지, {fn}명은 놓침")
print(f"→ {fp}명에게 불필요한 개입이 발생하지만 비용은 낮음")

# %%
# ROC Curve & Precision-Recall Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='#E74C3C', lw=2, label=f'{best_name} (AUC={roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec, prec, color='#3498DB', lw=2, label=f'{best_name}')
axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('../outputs/figures/10_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3.6 피처 중요도 분석
#
# 최종 모델이 어떤 변수를 중요하게 사용했는지 확인한다.
# 이 결과는 04_insights 노트북에서 비즈니스 해석에 활용된다.

# %%
# 피처 중요도 추출
importances = best_model.feature_importances_
feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# 상위 15개 시각화
top_n = 15
fig, ax = plt.subplots(figsize=(10, 8))
top = feat_imp.head(top_n).sort_values('importance', ascending=True)
colors = ['#E74C3C' if imp > feat_imp['importance'].quantile(0.9) else '#3498DB'
          for imp in top['importance']]
ax.barh(top['feature'], top['importance'], color=colors)
ax.set_title(f'{best_name} — 피처 중요도 Top {top_n}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('../outputs/figures/11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Top 10 피처:")
for i, row in feat_imp.head(10).iterrows():
    print(f"  {row['feature']:30s}  {row['importance']:.4f}")

# %% [markdown]
# ## 3.7 전체 모델 테스트 성능 비교

# %%
# 모든 모델을 학습 데이터로 fit → 테스트 평가
all_test_results = {}

for name, (model, X_tr, y_tr) in models.items():
    model.fit(X_tr, y_tr)
    X_te = X_test_s if name in ['Logistic Regression', 'SVM (RBF)'] else X_test
    y_p = model.predict(X_te)
    y_pp = model.predict_proba(X_te)[:, 1]
    all_test_results[name] = evaluate_model(y_test, y_p, y_pp)

# 튜닝된 모델 추가
all_test_results[best_name] = test_metrics

# 결과 테이블
result_df = pd.DataFrame(all_test_results).T
result_df = result_df.sort_values('F1', ascending=False)
print("전체 모델 테스트 성능 비교:")
print(result_df.round(4).to_string())

# %%
# 테스트 성능 시각화
fig, ax = plt.subplots(figsize=(12, 6))
metrics_plot = ['Recall', 'F1', 'AUC-ROC', 'Precision']
x = np.arange(len(result_df))
width = 0.2

for i, metric in enumerate(metrics_plot):
    bars = ax.bar(x + i * width, result_df[metric], width, label=metric, alpha=0.85)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(result_df.index, rotation=30, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.legend()
ax.set_title('모델별 테스트 성능 비교', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/12_all_models_test.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3.8 모델 및 결과 저장

# %%
# 최종 모델 저장
model_path = PROCESSED_DIR.parent.parent / 'outputs' / 'models'
model_path.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_path / 'best_model.joblib')
print(f"모델 저장: outputs/models/best_model.joblib")

# 피처 중요도 저장 (Tableau 대시보드용)
feat_imp.to_csv(PROCESSED_DIR / 'feature_importance.csv', index=False)
print(f"피처 중요도 저장: data/processed/feature_importance.csv")

# 예측 확률 저장 (Tableau 대시보드용)
full_data = pd.read_csv(PROCESSED_DIR / 'attrition_clean.csv')
X_full = full_data.drop(columns=['Attrition'])
pred_prob = best_model.predict_proba(X_full)[:, 1]
predictions = pd.DataFrame({
    'Predict_Prob': pred_prob,
    'Risk_Level': pd.cut(pred_prob, bins=[0, 0.3, 0.6, 1.0], labels=['저위험', '중위험', '고위험'])
})
predictions.to_csv(PROCESSED_DIR / 'predictions.csv', index=False)
print(f"예측 확률 저장: data/processed/predictions.csv")

# 전체 테스트 결과 저장
result_df.round(4).to_csv(PROCESSED_DIR / 'model_comparison.csv')
print(f"모델 비교 저장: data/processed/model_comparison.csv")

# %% [markdown]
# ## 3.9 모델링 요약
#
# ### 최종 모델 선택 근거
# - 6개 모델을 Stratified 5-Fold CV로 비교
# - LightGBM과 XGBoost에 대해 GridSearchCV 튜닝
# - **F1 기준** 최적 모델 선정 (Recall과 Precision의 균형)
#
# ### 불균형 처리
# - 모든 모델에 `class_weight='balanced'` 또는 `scale_pos_weight` 적용
# - Accuracy(84%)가 아닌 **Recall과 F1**로 평가
#
# ### 다음 단계
# - 04_insights에서 피처 중요도를 비즈니스 관점으로 해석
# - 고위험 세그먼트 프로파일링
# - 실행 가능한 HR 제언 도출
