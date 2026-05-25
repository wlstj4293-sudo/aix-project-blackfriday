# AI+X: 딥러닝 2026-1 기말 프로젝트 G30조
고객 인구통계학적 특성에 따른 블랙프라이데이 소비 행동 예측 모델링  
(Predicting Black Friday Consumer Behavior Based on Customor Demographics)

---
## Table of Contents
1. Title
2. Members
3. Motivation
4. Datasets
5. Methodology
6. Evaluation & Analaysis
7. Related Work
8. Conclusion

---
## 1.Title
**고객 인구통계학적 특성에 따른 블랙프라이데이 소비 행동 예측 모델링**  
**(Predicting Black Friday Consumer Behavior Based on Customor Demographics)**

---
## 2. Members
신소재공학부 2020045078 송진서  
신소재공학부 2021091476 김민철  
신소재공학부 2023084575 서민석  
신소재공학부 2023027492 이경현  

---
## 3. Motivation
**실시간 소비 데이터의 시각화와 호기심**  
매년 '무신사 블랙프라이데이' 같은 국내외 대형 이커머스 플랫폼에서는 행사 기간 동안 실시간 누적 매출액과 판매 흐름을 대중에게 투명하게 공개하고 있다.  
특히 '무신사'의 경우 2022년부터 2025년까지의 블프 기간동안의 매출액을 살펴보면 2135억 원, 3083억 원, 3654억 원, 3685억 원으로 소비자들의 뜨거운 관심과 압도적인 매출을 보여준다.  
이처럼 동시간대에 역동적으로 변하는 매출 지표를 보면서 과연 소비자들이 인구통계학적 특성에 따라 어떻게 반응하는 지 데이터 과학적 호기심이 생겨 이 주제를 선정하게 되었다.

---
## 4. Datasets

본 프로젝트에서는 Kaggle의 [Black Friday Sale 데이터셋](https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale)을 활용한다. 데이터셋은 소비자의 인구통계학적 특성 정보와 상품 카테고리, 그리고 최종 구매 금액을 포함한 총 12개의 칼럼(Features)으로 구성되어 있다. 본 feature들은 크게 고객 식별 / 상품 식별 / 인구통계학적 특성 / 상품 특성 / 예측 목표 변수로 분류하였다.

### 📋 데이터셋 변수 정의 (Feature Definition)

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

## 데이터 특이사항 (Data Note)
본 데이터셋은 일부 주요 변수 'Occupation', 'Product_Catgory_1', 'Product_Catgory_2', 'Product_Catgory_3'이 실제 명칭 대신 0~20 사이의숫자 코드로 마스킹 처리되어 있다.

---
## 5. Methodology
 
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


