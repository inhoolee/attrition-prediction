"""모델 평가 함수 모듈"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """분류 모델의 주요 평가 지표를 계산한다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블

    Returns:
        accuracy, precision, recall, f1 점수를 담은 딕셔너리
    """
    pass


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> plt.Figure:
    """혼동 행렬을 시각화한다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        labels: 클래스 레이블 목록

    Returns:
        matplotlib Figure 객체
    """
    pass
