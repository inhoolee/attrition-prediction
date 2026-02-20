# IBM HR 데이터셋 컬럼 설명 (한글)

`IBM HR Analytics Employee Attrition & Performance` 데이터셋의 원본 35개 컬럼을 실무 관점에서 정리한 문서입니다.

## 데이터셋 개요

- 관측 단위: 직원 1명(한 행 = 직원 1명)
- 행/열 크기: 1,470행 x 35컬럼
- 타겟 변수: `Attrition` (`Yes`/`No`)
- 문제 유형: 이진 분류(퇴사 예측)
- 참고: `EmployeeCount`, `StandardHours`, `Over18`는 상수 컬럼이고 `EmployeeNumber`는 식별자 컬럼

## 컬럼 사전

| 컬럼명 | 의미(한글) | 값/단위 예시 | 모델링 권장 |
|---|---|---|---|
| `Age` | 직원 나이 | 정수 (예: 41) | 사용 |
| `Attrition` | 퇴사 여부(타겟) | `Yes`, `No` | 타겟 |
| `BusinessTravel` | 출장 빈도 | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` | 사용 |
| `DailyRate` | 일급 관련 지표 | 정수 | 사용(해석 주의) |
| `Department` | 소속 부서 | `Sales`, `Research & Development`, `Human Resources` | 사용 |
| `DistanceFromHome` | 집-직장 거리 | 정수(마일 기준) | 사용 |
| `Education` | 최종 학력 수준 | 1~5 | 사용(순서형) |
| `EducationField` | 전공 분야 | `Life Sciences`, `Medical`, `Marketing`, `Technical Degree` 등 | 사용 |
| `EmployeeCount` | 직원 수 상수 컬럼 | 상수값 `1` | 제외 |
| `EmployeeNumber` | 직원 고유 ID | 정수 식별자 | 제외 |
| `EnvironmentSatisfaction` | 근무 환경 만족도 | 1~4 | 사용(순서형) |
| `Gender` | 성별 | `Male`, `Female` | 사용 |
| `HourlyRate` | 시급 관련 지표 | 정수 | 사용(해석 주의) |
| `JobInvolvement` | 직무 몰입도 | 1~4 | 사용(순서형) |
| `JobLevel` | 직급 레벨 | 1~5 | 사용(순서형) |
| `JobRole` | 직무 역할 | `Sales Executive`, `Research Scientist`, `Manager` 등 | 사용 |
| `JobSatisfaction` | 직무 만족도 | 1~4 | 사용(순서형) |
| `MaritalStatus` | 결혼 상태 | `Single`, `Married`, `Divorced` | 사용 |
| `MonthlyIncome` | 월 소득 | 정수 | 사용 |
| `MonthlyRate` | 월 급여 관련 지표 | 정수 | 사용(해석 주의) |
| `NumCompaniesWorked` | 이전 근무 회사 수 | 정수 | 사용 |
| `Over18` | 18세 이상 여부 상수 컬럼 | 상수값 `Y` | 제외 |
| `OverTime` | 초과근무 여부 | `Yes`, `No` | 사용 |
| `PercentSalaryHike` | 급여 인상률(%) | 정수 | 사용 |
| `PerformanceRating` | 성과평가 등급 | 3~4 | 사용(분산 작음 주의) |
| `RelationshipSatisfaction` | 대인관계 만족도 | 1~4 | 사용(순서형) |
| `StandardHours` | 표준 근무시간 상수 컬럼 | 상수값 `80` | 제외 |
| `StockOptionLevel` | 스톡옵션 수준 | 0~3 | 사용(순서형) |
| `TotalWorkingYears` | 총 경력 연수 | 정수(년) | 사용 |
| `TrainingTimesLastYear` | 최근 1년 교육 횟수 | 정수 | 사용 |
| `WorkLifeBalance` | 워라밸 수준 | 1~4 | 사용(순서형) |
| `YearsAtCompany` | 현 회사 근속 연수 | 정수(년) | 사용 |
| `YearsInCurrentRole` | 현 직무 수행 연수 | 정수(년) | 사용 |
| `YearsSinceLastPromotion` | 최근 승진 이후 경과 연수 | 정수(년) | 사용 |
| `YearsWithCurrManager` | 현 관리자와 함께 근무한 연수 | 정수(년) | 사용 |

## 코드값 해석(순서형/등급형)

| 컬럼 | 코드값 의미 |
|---|---|
| `Education` | 1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor |
| 만족도 계열 (`EnvironmentSatisfaction`, `JobSatisfaction`, `RelationshipSatisfaction`) | 1: Low, 2: Medium, 3: High, 4: Very High |
| `JobInvolvement` | 1: Low, 2: Medium, 3: High, 4: Very High |
| `WorkLifeBalance` | 1: Bad, 2: Good, 3: Better, 4: Best |
| `PerformanceRating` | 3: Excellent, 4: Outstanding |
| `StockOptionLevel` | 0~3 (수준 증가) |

## 전처리 체크리스트

1. 제외 컬럼 제거: `EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`
2. 타겟 변환: `Attrition` (`Yes`=1, `No`=0)
3. 범주형 인코딩: 이진 컬럼(`Gender`, `OverTime` 등)과 다범주 컬럼(`Department`, `JobRole` 등) 분리 처리
4. 스케일링: 거리/소득/연수 계열 수치형 컬럼은 모델 특성에 맞게 표준화
5. 불균형 대응: `Attrition` 불균형(약 16:84) 고려해 평가지표(F1, PR-AUC)와 class weight 검토

## 해석 시 주의사항

- `DailyRate`, `HourlyRate`, `MonthlyRate`는 이름상 급여 지표지만 해석에 제한이 있어 `MonthlyIncome`과 함께 검증적으로 해석하는 것이 안전합니다.
- 이 데이터셋은 시계열 이력 데이터가 아니라 스냅샷 성격이므로, 인과관계 해석보다 위험 신호 탐지 관점이 적합합니다.
