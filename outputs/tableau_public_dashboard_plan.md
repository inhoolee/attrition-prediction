# Tableau Public 공개용 대시보드 기획안

## 1) 목표

- 대시보드 제목: `Why Employees Leave: Attrition Risk & Retention Plan`
- 공개 채널: Tableau Public
- 대상 독자: 채용 담당자, HR 리더, 비기술 이해관계자
- 핵심 메시지:
1. 퇴사는 특정 부서/직무/근속 구간에 집중된다.
2. 주요 드라이버(야근, 소득, 성장 정체, 만족도)가 명확하다.
3. 위험군을 선별하면 리텐션 개입 우선순위를 정할 수 있다.

## 2) 공개용 주의사항 (필수)

- Tableau Public에 올린 워크북/데이터는 비공개가 아니며 누구나 접근 가능하다.
- Tableau Public에서는 데이터 다운로드가 가능할 수 있으므로 민감정보/PII를 포함하면 안 된다.
- Tableau Public는 워크북당 데이터 행 수 제한(15M)이 있다.

## 3) 데이터 소스 설계

### 메인 소스

- 파일: `data/processed/attrition_clean.csv`
- 크기: 1,470행 x 49열
- 특징: 모델링용 인코딩 컬럼(원-핫 + 파생변수) 포함

### 보조 소스

- 파일: `data/processed/predictions.csv`
- 컬럼: `Predict_Prob`, `Risk_Level`
- 용도: 위험도 분류/스코어 시각화

### 비교/설명 소스

- 파일: `data/processed/feature_importance.csv`
- 파일: `data/processed/model_comparison.csv`

## 4) Tableau 계산 필드 (권장)

### 공통 KPI

```text
Headcount = COUNT([Attrition])
Attrition Count = SUM([Attrition])
Attrition Rate = AVG([Attrition])
High Risk Count = SUM(IIF([Risk_Level]="고위험",1,0))
High Risk Rate = AVG(IIF([Risk_Level]="고위험",1,0))
```

### 레이블 복원 (인코딩 컬럼 -> 사람 친화형)

```text
Attrition Label = IF [Attrition]=1 THEN "Yes" ELSE "No" END
Gender Label = IF [Gender]=1 THEN "Male" ELSE "Female" END
OverTime Label = IF [OverTime]=1 THEN "Yes" ELSE "No" END
```

```text
Department Label =
IF [Department_Sales]=1 THEN "Sales"
ELSEIF [Department_Research & Development]=1 THEN "R&D"
ELSE "HR" END
```

```text
BusinessTravel Label =
IF [BusinessTravel_Travel_Frequently]=1 THEN "Travel_Frequently"
ELSEIF [BusinessTravel_Travel_Rarely]=1 THEN "Travel_Rarely"
ELSE "Non-Travel" END
```

```text
MaritalStatus Label =
IF [MaritalStatus_Married]=1 THEN "Married"
ELSEIF [MaritalStatus_Single]=1 THEN "Single"
ELSE "Divorced" END
```

### 구간화

```text
Age Band =
IF [Age] < 30 THEN "<30"
ELSEIF [Age] < 40 THEN "30s"
ELSEIF [Age] < 50 THEN "40s"
ELSE "50+" END
```

```text
Income Band =
IF [MonthlyIncome] < 3000 THEN "<3K"
ELSEIF [MonthlyIncome] < 6000 THEN "3K-6K"
ELSEIF [MonthlyIncome] < 10000 THEN "6K-10K"
ELSE "10K+" END
```

```text
Tenure Band =
IF [YearsAtCompany] <= 1 THEN "0-1y"
ELSEIF [YearsAtCompany] <= 2 THEN "1-2y"
ELSEIF [YearsAtCompany] <= 5 THEN "3-5y"
ELSEIF [YearsAtCompany] <= 10 THEN "6-10y"
ELSE "10y+" END
```

## 5) 대시보드 구성 (3개 탭)

## Tab A. Executive Overview

- 목적: 한 화면에서 조직 퇴사 상태 요약
- 권장 시트:
1. KPI 카드 5개: Headcount, Attrition Count, Attrition Rate, High Risk Count, High Risk Rate
2. 부서별 퇴사율 막대: `Department Label` x `Attrition Rate`
3. 직무별 퇴사율 막대: 직무 복원 필드 x `Attrition Rate`
4. 근속구간별 퇴사율: `Tenure Band` x `Attrition Rate`
5. 연령대별 퇴사율: `Age Band` x `Attrition Rate`
- 필터: Department, Gender, OverTime

## Tab B. Driver Deep Dive

- 목적: 퇴사 유발 요인을 설명 가능한 형태로 제시
- 권장 시트:
1. Feature Importance Top 10: `feature_importance.csv`
2. 야근 여부별 퇴사율: `OverTime Label` x `Attrition Rate`
3. 소득구간별 퇴사율: `Income Band` x `Attrition Rate`
4. 만족도 지표 비교: `SatisfactionIndex`, `JobInvolvement`, `WorkLifeBalance`
5. 출장빈도 x 결혼상태 Heatmap: `BusinessTravel Label` x `MaritalStatus Label` with `Attrition Rate`
- 툴팁 문구 예시:
`퇴사율 {AVG([Attrition]):.1%} | 인원 {COUNT([Attrition])}`

## Tab C. Risk & Action

- 목적: 개입 우선순위와 액션 포인트 제시
- 권장 시트:
1. Risk Scatter: X=`YearsAtCompany`, Y=`MonthlyIncome`, Color=`Risk_Level`, Size=`Predict_Prob`
2. 위험군 프로파일 테이블: Risk_Level별 인원, 평균소득, 평균만족도
3. 부서 액션 매트릭스: X=`Attrition Rate`, Y=`COUNT([Attrition])`, Color=`Department Label`
4. 모델 성능 카드: `model_comparison.csv`의 Recall/F1/AUC 하이라이트
- 액션 텍스트 카드:
1. 야근 집단 관리
2. 초기 근속(0-2년) 온보딩 강화
3. 저소득 + 고위험군 선제 면담

## 6) 레이아웃/디자인 가이드

- Desktop 기준 크기: `1366 x 768` 또는 `1400 x 900`
- 모바일 레이아웃 별도 구성 (핵심 KPI + 2개 차트만)
- 색상:
1. Attrition Yes: `#E74C3C`
2. Attrition No: `#3498DB`
3. Risk High/Med/Low: `#E74C3C / #F39C12 / #27AE60`
- 가독성:
1. 시트당 핵심 질문 1개만
2. 축 정렬/단위(%, 명) 일관 유지
3. 과도한 필터 노출 금지 (성능/UX 저하)

## 7) 퍼블리시 전 체크리스트

1. 원본 데이터에 개인식별정보(이름, 이메일, 사번 원본) 미포함 확인
2. Tooltip/Caption에 민감정보 노출 여부 확인
3. 워크북 설명에 프로젝트 목적/데이터 출처/한계 명시
4. 썸네일 시트는 KPI + 핵심 차트 1개가 보이도록 설정
5. 성능 점검: 불필요한 필터/고마크 시트 제거

## 8) Tableau Public 설명문 템플릿

```text
Employee attrition dashboard built on the IBM HR Analytics dataset.
This project highlights where attrition risk concentrates, what factors are most associated with churn, and how HR can prioritize retention actions.
All data used here is publicly shareable and non-sensitive.
```

## 9) 참고 문서

- Tableau Public 공개 주의사항: https://help.tableau.com/current/pro/desktop/en-us/publish_workbooks_tableaupublic.htm
- Tableau Public FAQ (행 수 제한/공개 범위): https://help.tableau.com/current/pro/desktop/en-us/public_faq.htm
- 시각화 베스트 프랙티스: https://help.tableau.com/current/blueprint/en-us/bp_visual_best_practices.htm
- 시트/필터 성능 팁: https://help.tableau.com/current/pro/desktop/en-us/perf_visualization.htm
