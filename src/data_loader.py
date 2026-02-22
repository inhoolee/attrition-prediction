"""데이터 로딩 유틸리티 모듈"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# 프로젝트 루트(.env)의 환경변수를 로드한다.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_raw_data_path() -> Path:
    """원본 CSV 경로를 확인하고, 찾지 못하면 안내와 함께 오류를 낸다."""
    env_path = os.getenv("ATTRITION_RAW_CSV")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate

    expected_path = RAW_DIR / RAW_FILENAME
    if expected_path.is_file():
        return expected_path

    # data 디렉토리 내에서 동일 파일명을 탐색한다.
    discovered = sorted(DATA_DIR.rglob(RAW_FILENAME))
    if discovered:
        return discovered[0]

    processed_csvs = sorted(p.name for p in PROCESSED_DIR.glob("*.csv")) if PROCESSED_DIR.exists() else []
    processed_hint = ", ".join(processed_csvs[:5])
    if len(processed_csvs) > 5:
        processed_hint += ", ..."

    message = (
        f"원본 데이터 파일을 찾을 수 없습니다.\n"
        f"- 기대 경로: {expected_path}\n"
        f"- 해결 방법 1: 파일을 위 경로에 배치\n"
        f"- 해결 방법 2: 환경변수 ATTRITION_RAW_CSV로 원본 CSV 절대경로 지정\n"
        f"- Kaggle 다운로드 예시:\n"
        f"  uvx --from kaggle kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset "
        f"-p {RAW_DIR}/ --unzip"
    )
    if processed_hint:
        message += f"\n- 참고: 현재 data/processed에는 다음 CSV가 있습니다: {processed_hint}"

    raise FileNotFoundError(message)


def load_raw_data() -> pd.DataFrame:
    """원본 HR 데이터셋을 로드한다.

    Returns:
        원본 CSV 데이터프레임
    """
    return pd.read_csv(_resolve_raw_data_path())


def load_processed_data(filename: str = "processed.csv") -> pd.DataFrame:
    """전처리된 데이터셋을 로드한다.

    Args:
        filename: 전처리된 파일명

    Returns:
        전처리된 데이터프레임
    """
    return pd.read_csv(PROCESSED_DIR / filename)
