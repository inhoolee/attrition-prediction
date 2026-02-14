"""데이터 로딩 유틸리티 모듈"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def load_raw_data() -> pd.DataFrame:
    """원본 HR 데이터셋을 로드한다.

    Returns:
        원본 CSV 데이터프레임
    """
    return pd.read_csv(RAW_DIR / RAW_FILENAME)


def load_processed_data(filename: str = "processed.csv") -> pd.DataFrame:
    """전처리된 데이터셋을 로드한다.

    Args:
        filename: 전처리된 파일명

    Returns:
        전처리된 데이터프레임
    """
    return pd.read_csv(PROCESSED_DIR / filename)
