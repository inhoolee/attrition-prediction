"""모델 평가 함수 모듈"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from src.plot_style import apply_plot_style


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """분류 모델의 주요 평가 지표를 계산한다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_prob: 양성 클래스 예측 확률 (AUC 계산용)

    Returns:
        평가 지표 딕셔너리
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list[str] | None = None, title: str = '') -> plt.Figure:
    """혼동 행렬을 시각화한다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        labels: 클래스 레이블 목록
        title: 차트 제목

    Returns:
        matplotlib Figure 객체
    """
    if labels is None:
        labels = ['재직 (0)', '퇴사 (1)']

    apply_plot_style(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title or '혼동 행렬', fontsize=14, fontweight='bold')
    return fig
