# AI+X: 딥러닝 2026-1 기말 프로젝트 G30조

---
## Title
**고객 인구통계학적 특성에 따른 블랙프라이데이 소비 행동 예측 모델링**  
**(Predicting Black Friday Consumer Behavior Based on Customor Demographics)**

---
## Members
신소재공학부 2020045078 송진서  
신소재공학부 2021091476 김민철  
신소재공학부 2023084575 서민석  
신소재공학부 2023027492 이경현  

---
## Table of Contents  
1. Introduction  
   1.1. Motivation  
   1.2. Reasearch Question  
2. Body  
   2.1. Datasets  
   2.2. Methodology  
   2.3. Analaysis & Evaluation
3. Conclusion  
   3.1. Trial and error  
   3.2. Conclusion  

---
## 1. Introduction

## 1.1. Motivation

**실시간 소비 데이터의 시각화와 호기심**  
매년 '무신사 블랙프라이데이' 같은 국내외 대형 이커머스 플랫폼에서는 행사 기간 동안 실시간 누적 매출액과 판매 흐름을 대중에게 투명하게 공개하고 있다.  
특히 '무신사'의 경우 2022년부터 2025년까지의 블프 기간동안의 매출액을 살펴보면 2135억 원, 3083억 원, 3654억 원, 3685억 원으로 소비자들의 뜨거운 관심과 압도적인 매출을 보여준다.  
이처럼 동시간대에 역동적으로 변하는 매출 지표를 보면서 과연 소비자들이 인구통계학적 특성에 따라 어떻게 반응하는 지 데이터 과학적 호기심이 생겨 이 주제를 선정하게 되었다.

## 1.2. Reasearch Question


<br>

---

## 2.Body

## 2.1. Datasets

본 프로젝트에서는 Kaggle의 [Black Friday Sale 데이터셋](https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale)을 활용한다. 데이터셋은 소비자의 인구통계학적 특성 정보와 상품 카테고리, 그리고 최종 구매 금액을 포함한 총 12개의 칼럼(Features)으로 구성되어 있다. 본 feature들은 크게 고객 식별 / 상품 식별 / 인구통계학적 특성 / 상품 특성 / 예측 목표 변수로 분류하였다.
<br>
<br>
### 데이터셋 변수 정의 (Feature Definition)

| 변수명 (Feature) | 설명 | 분류 |
| :--- | :--- | :--- |
| **User_ID** | 사용자 고유 식별 번호 | 고객 식별 |
| **Product_ID** | 상품 고유 식별 번호 | 상품 식별 |
| **Gender** | 성별 (M: 남성, F: 여성) | 인구통계학적 특성 |
| **Age** | 연령대별 그룹 (0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+) | 인구통계학적 특성 |
| **Occupation** | 직업군 마스킹 코드 (0 ~ 20 종류) | 인구통계학적 특성 |
| **City_Category** | 거주 도시 유형 (A, B, C 지역) | 인구통계학적 특성 |
| **Stay_In_Current_City_Years** | 현재 도시에 거주한 기간 (연 수, 4+는 4년 이상) | 인구통계학적 특성 |
| **Marital_Status** | 결혼 여부 (0: 미혼, 1: 기혼) | 인구통계학적 특성 |
| **Product_Category_1** | 상품 주 카테고리 | 상품 특성 |
| **Product_Category_2** | 상품 부 카테고리 | 상품 특성 |
| **Product_Category_3** | 상품 하위 카테고리 | 상품 특성 |
| **Purchase** | **[Target]** 고객이 해당 상품에 지출한 금액 | 예측 목표 변수 |
<br>


## 데이터 특이사항 (Data Note)
본 데이터셋은 일부 주요 변수 'Occupation', 'Product_Catgory_1', 'Product_Catgory_2', 'Product_Catgory_3'이 실제 명칭 대신 0~20 사이의숫자 코드로 마스킹 처리되어 있다.
<br>
<br>
<br>

## 2.2. Methodology
 
Machine Learning은 라이브러리로 'Scikit-learn'을 사용했고, 방식은 'Random Forest'와 'HistGB'방식을 사용했다. Deep Learning은 라이브러리로 'PyTorch'를 사용했고, 방식은 'Deep DNN'과 'Wide NN'방식을 사용했다.

 **각 방식별 작동방식 및 특징**  
 
**1. Scikit-learn, Random Forest (머신러닝)**  
•	**작동 방식**: 수백 개의 '결정 트리(Decision Tree)'를 각기 다른 데이터 샘플로 학습시킨 뒤, 그 결과들의 평균(회귀)이나 다수결(분류)로 최종 답을 내는 '앙상블(Ensemble)' 모델.  
•	**특징**: 데이터의 일부가 잘못되어도 전체적인 결과가 흔들리지 않아 안정적이다.

**2. Scikit-learn, HistGB (머신러닝 - Histogram-based Gradient Boosting)**  
•	**작동 방식**: 데이터를 구간(Histogram)으로 나누어 처리하여 학습 속도를 획기적으로 높인 Gradient Boosting 모델이으로, 이전 트리의 오차를 다음 트리가 보완하며 학습하는 방식.  
•	**특징**: 대용량 데이터에서 속도가 미친 듯이 빠르고 성능도 거의 최상위권이다.
   
**3. PyTorch, Deep DNN (딥러닝)**  
•	**작동 방식**: 입출력 사이의 은닉층(Hidden Layer)을 여러 층 깊게 쌓아서 데이터의 복잡하고 추상적인 특징을 찾아내는 방식.  
•	**특징**: 레이어가 깊을수록 데이터 사이의 아주 미묘하고 복잡한 상관관계를 스스로 찾아낸다.

**4. PyTorch, Wide NN (딥러닝)**  
•	**작동 방식**: 층을 깊게 쌓기보다는, 한 층에 수많은 뉴런(폭)을 넓게 배치하는 방식.  
•	**특징**: 각 변수들의 직접적인 영향력을 잘 학습해. 'Wide & Deep' 모델의 한 축으로, 기억력(Wide)이 좋다.

<br>
<br>

## 2.3. Analaysis & Evaluation

## 2.4. Analysis & Evaluation (실험 결과 분석 및 평가)

본 프로젝트에서는 일관성 있는 실험 환경을 구축하기 위해 `GLOBAL_SEED = 42`로 고정하였으며, 총 3가지 시나리오 기반의 소스 코드(`code_1`, `code_2`, `code_3`)를 실행하여 4대 AI 아키텍처(Random Forest, LightGBM, PyTorch Deep DNN, PyTorch Wide NN)의 성능 및 변수 영향도를 다각도로 비교 분석하였습니다.

---

### [실험 1] 아키텍처별 자율 최적화 학습 성능 대조 (`code_1`)

* **실험 목적**: 각 인공지능 모델이 주어진 하이퍼파라미터 환경 내에서 스스로 변수별 가중치(Feature Importance)를 최적화하여 도달할 수 있는 최종 예측 성능을 대조한다.
* **성능 지표**: $R^2$ Score(데이터 설명력, 높을수록 우수) 및 RMSE(평균 예측 오차, 낮을수록 우수)

#### ① 최종 예측 성능 대조 분석

<img width="1400" height="750" alt="final 성능대조" src="https://github.com/user-attachments/assets/15911165-879c-43b6-9a81-7583ed2da90f" />

* **분석 결과**: 머신러닝 기반의 **LightGBM이 $R^2$ 14.79%, RMSE $971,040**을 기록하며 가장 우수한 성능을 보여주었습니다. 뒤를 이어 딥러닝 기반의 PyTorch Deep DNN($R^2$ 13.24%)이 견고한 성능을 도출했습니다. 

#### ② 아키텍처별 내부 가중치 분석 (Feature Importance)
<img width="1500" height="1000" alt="아키텍처별 가중치 부여값" src="https://github.com/user-attachments/assets/5b26df9f-c55d-438a-bb60-35e196659d4a" />


* **핵심 인사이트**: 모델들이 자율적으로 학습한 가중치를 시각화한 결과, **LightGBM(91.3%)**과 **PyTorch Deep DNN(91.1%)**은 구매 금액 예측에 있어 **`City_Category`** 변수에 극단적으로 높은 의존성을 보였습니다. 반면, Random Forest와 PyTorch Wide NN은 비교적 변수들을 다각도로 반영하는 성향을 보였으나 최종 성능은 하락했습니다.

---

### [실험 2] 공통 변수 가중치 강제 주입 조건 하의 성능 비교 (`code_2`)

* **실험 목적**: 모델 고유의 변수 가중치 최적화 기능을 배제하고, 모든 아키텍처에 **동일한 Feature Importance 가중치를 강제로 고정 주입**했을 때, 순수한 알고리즘 아키텍처 체력만으로 유의미한 성능 차이가 발생하는지 검증합니다.

#### ① 4대 아키텍처 강제 고정 주입 공통 변수 가중치
<img width="1000" height="600" alt="공통가중치 부여값" src="https://github.com/user-attachments/assets/1053cfe3-8727-439c-9dc8-8fb26bf7a656" />


* **주입 조건**: 실험 1에서 지배적인 영향력을 보인 `City_Category`(48.7%), `Gender`(14.4%), `Gender_Age`(12.4%) 순으로 통제된 가중치 배율을 고정 주입했습니다.

#### ② 동일 가중치 상태의 최종 성능 대조
<img width="1400" height="750" alt="공통가중치 성능비교" src="https://github.com/user-attachments/assets/c698af8b-3aab-4121-a95a-0818d8cff720" />


* **분석 결과**: 놀랍게도 가중치를 통제하자 **PyTorch Deep DNN이 $R^2$ 17.15%, RMSE $957,520**으로 치고 나가며 자율 최적화 때보다 더 높은 전체 1위의 성능을 기록했습니다. 
* **인사이트**: 비선형적인 다층 신경망(Deep DNN) 구조는 도메인 지식이나 통제된 특성(Feature)이 정교하게 가이드될 때, 트리 기반 머신러닝보다 데이터의 복잡한 상관관계를 추론하는 능력이 극대화됨을 증명합니다.

---

### 🎯 [실험 3] `City_Category` 가중치 100% 부여 단독 학습 실험 (`code_3`)

* **실험 목적**: 앞선 실험들에서 핵심 지표로 작용한 거주 도시 정보(`City_Category`)의 영향력을 극단적으로 스케일업(100% 부여)하여, 오직 **이 단 하나의 Feature만을 가지고 예측을 수행할 때** 아키텍처별 방어력을 확인합니다.

#### ① City_Category 단독 학습 시 아키텍처별 최종 성능 대조
<img width="1400" height="750" alt="city100성능대조" src="https://github.com/user-attachments/assets/71141eb3-518c-4f57-97f1-8434e3a47888" />


* **분석 결과**: 단 하나의 변수만으로 학습을 강제했음에도 불구하고, LightGBM($R^2$ 13.69%)과 Random Forest($R^2$ 13.66%)가 놀라운 수준으로 성능을 방어해 냈습니다.
* **인사이트**: 
  1. 본 블랙프라이데이 데이터셋의 예측 타겟(`Purchase`)은 **소비자가 어떤 도시(A, B, C)에 거주하느냐에 따라 소비 규모의 정체성이 상당 부분 결정된다**는 데이터의 내재적 특성을 방증합니다.
  2. 트리 기반 알고리즘(ML)은 단일 범주형 변수의 분기(Split) 연산만으로도 수치형 타겟의 기댓값을 매우 정교하게 맵핑해내는 강인함(Robustness)을 보여준 반면, 딥러닝 아키텍처는 단일 변수 환경에서 상대적으로 취약한 모습을 보였습니다.


---

## 3. Conclusion  
## 3.1. Trial and error  
##### (1) 데이터셋 변경 ①
본 프로젝트는 초기에 [Retail Black Friday Sales 데이터셋](https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset)을 활용하여 '평상시'와 '블랙프라이데이' 시즌 간의 소비자 행동 변화를 비교 분석하고자 함. 그러나 초기데이터 탐색 및 시각화 과정에서 기존 데이터셋의 한계점을 발견하여 프로젝트 방향을 수정하게 됨.  

* **① 할인율별 구매 발생 확률 분포**: 평상시와 블프 시즌 간 할인율별 밀도 함수가 비정상적으로 일치.  
* **② 상품 카테고리별 구매 선택 확률**: `Accessories`, `Beauty`, `Electronics` 등 서로 다른 카테고리임에도 불구하고 모든 항목이 약 10% 내외의 비슷한 선택 확률을 보임.  
* **③ 연령대별 구매 발생 확률** : 전 영령대에 걸쳐 구매건수 비중이 정확히 20% 수준으로 균등하게 분배되어 있음.   

위 세 가지 분석 결과로, 해당 데이터셋은 실제 소비자의 역동적인 구매 이력을 기록한 데이터가 아닌, **학습 및 예측 모델링 연습을 위해 통계적 패키지로 무작위 생성된 가상 데이터**인 것으로 판단되었고, 인구통계학적으로 데이터셋을 분석하기에 부적합하다고 판단하여 데이터셋 변경을 결정함.

<br>

<img width="1512" height="600" alt="trial and error 1" src="https://github.com/user-attachments/assets/c4ea8b37-d99e-4f1a-b849-12591659e3a4" />


##### (2) 데이터셋 변경 ②

그 다음으로 [Christmas Sales and Trends 데이터셋](https://www.kaggle.com/datasets/ibikunlegabriel/christmas-sales-and-trends)을 활용하여 분석을 시도함. 해당 데이터셋은 11월과 12월의 크리스마스 시즌, 블랙프라이데이 시즌, 그리고 그 외의 평상시 기간을 모두 포함하고 있어, 크리스마스 데이터를 필터링한 후 '평상시'와 '블랙프라이데이' 간의 인구통계학적 소비 패턴을 비교하기에 적합해 보였음.

<br>

<img width="1645" height="943" alt="image" src="https://github.com/user-attachments/assets/4aaf9260-4bcc-45a5-a94f-460a8c88c8c8" />

<br>

해당 데이터를 기반으로 초기 분석을 진행한 결과, 카테고리별 매출 분포나 건당 평균 매출액 등에서 기간별로 유의미한 데이터 차이를 발견하는 성과를 거두기도 함.

<br>

<img width="1617" height="907" alt="image" src="https://github.com/user-attachments/assets/78a02c68-1478-463a-9ceb-09485918acb4" />

<br>

그러나 세부 데이터 탐색(EDA) 과정에서 다음과 같은 치명적인 한계점들을 다시 한번 직면하게 됨.

* **① 일률적인 할인율 규칙**: 모든 상품의 할인율이 10%로 고정되어 있어, 할인 금액이나 할인율 변화에 따른 소비자의 반응 및 유의미한 상관관계를 도출할 수 없었음.
* **② 인구통계학적 변별력 부족**: 연령대별 평균 구매 금액을 제외하고는, 성별이나 나이 등 핵심 인구통계학적 특성에 따른 시즌별 구매 행동 변화를 관찰하기 어려웠음.
* **③ 가상 규칙의 비현실성**: 블랙프라이데이나 크리스마스 당일을 제외한 모든 평상시 기간의 할인율을 일괄적으로 0%로 상정해 둔 데이터 구조를 발견함. 이는 상시 할인이 존재하는 실제 이커머스 시장의 매출 데이터와 괴리가 너무 크다고 판단됨.

위 세 가지 한계점으로 인해, 해당 데이터셋 역시 실제 소비자의 역동적인 구매 패턴을 반영하지 못하는 한계가 명확하다고 결론지었음. 이에 따라 분석의 신뢰도를 높이기 위해 더 정교하고 정제된 현재의 최종 데이터셋으로 재변경을 결정하게 됨.

<br>
<br>

## 3.2. Conclusion  
<br>
<br>

