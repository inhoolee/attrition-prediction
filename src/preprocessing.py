"""데이터 전처리 함수 모듈"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categoricals(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """범주형 변수를 인코딩한다.

    Args:
        df: 입력 데이터프레임
        columns: 인코딩할 컬럼 목록. None이면 object 타입 컬럼 자동 선택

    Returns:
        인코딩된 데이터프레임
    """
    pass


def scale_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """수치형 변수를 스케일링한다.

    Args:
        df: 입력 데이터프레임
        columns: 스케일링할 컬럼 목록. None이면 수치형 컬럼 자동 선택

    Returns:
        스케일링된 데이터프레임
    """
    pass
