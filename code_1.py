from pathlib import Path
import pandas as pd  # 파이썬에서 excel과 같은 역할
import numpy as np  # 수학, 과학, 행렬 계산 역할
import matplotlib.pyplot as plt  # 그래프 작성 역할
import time  # 각 모델별 연산(학습) 시간을 초 단위로 정밀 측정하기 위한 내장 라이브러리

# 1. 사이킷런 (머신러닝 전처리, 모델, 평가 지표)
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 검증용으로 분할
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 문자열 인코딩 및 데이터 표준화
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor  # ML 모델 2종
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가를 위한 오차(RMSE) 및 설명력(R2) 지표
from sklearn.inspection import permutation_importance  # 모델의 블랙박스를 열어 변수별 가중치를 공정하게 계산
from sklearn.base import BaseEstimator, RegressorMixin  # 파이토치 딥러닝 모델을 사이킷런 형태로 감싸기 위한 기본 객체

# 2. 파이토치 (인공신경망 딥러닝 아키텍처 구현)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# 🔒 [재현성 보장] 난수(Seed) 고정 파트
# 코드를 언제, 어디서 돌려도 완전히 동일한 학습 결과와 평가 수치가 나오도록 통제합니다.
# (단, 컴퓨터 하드웨어 스케줄링에 따라 '학습 소요 시간'은 미세하게 달라집니다.)
# =========================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- [그래프 환경 세팅] 맥북 화면에서 한글이 깨지지 않도록 태블로 스타일 미니멀 서식 지정 ---
plt.rcParams['font.family'] = 'AppleGothic'  # 맥북 기본 한글 폰트 적용
plt.rcParams['axes.unicode_minus'] = False   # 그래프 축에 마이너스(-) 부호 깨짐 방지
plt.ion()                                    # 인터랙티브 모드 활성화 (그래프 창 제어 원활화)

# Apple 실리콘(M1, M2, M3 등) 칩셋의 GPU 가속 장치(mps)가 있으면 쓰고, 없으면 CPU를 사용합니다.
pt_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🍏 PyTorch 연산 장치: {pt_device}")

# =========================================================================
# 📊 [데이터 로드 및 유저 프로필 기반 전처리]
# =========================================================================
print("📊 블랙프라이데이 데이터셋을 로드하고 있습니다...")
current_dir = Path(__file__).resolve().parent
file_path = current_dir / 'BFtrain.csv'
if not file_path.exists():
    file_path = current_dir / 'archive' / 'BFtrain.csv'

raw_df = pd.read_csv(file_path)
raw_df.columns = raw_df.columns.str.strip()  # 컬럼명 앞뒤에 혹시 모를 공백 제거

# [데이터 압축/집계] 유저 한 명이 여러 번 구매한 내역을 하나로 묶어 '총 구매 금액'을 구합니다.
user_df = raw_df.groupby('User_ID').agg({
    'Gender': 'first', 'Age': 'first', 'Occupation': 'first',
    'City_Category': 'first', 'Stay_In_Current_City_Years': 'first', 'Marital_Status': 'first',
    'Purchase': 'sum'  # 한 유저의 모든 구매 금액을 합산 (타겟 변수 고도화)
}).reset_index()

# [교차 피쳐 생성] '성별+나이', '나이+도시' 등 문자열을 조합해 AI가 복합적인 타겟층 맥락을 이해하도록 돕습니다.
user_df['Gender_Age'] = user_df['Gender'] + "_" + user_df['Age']
user_df['Age_City'] = user_df['Age'] + "_" + user_df['City_Category']
user_df['Gender_City'] = user_df['Gender'] + "_" + user_df['City_Category']

# [라벨 인코딩] AI가 이해하지 못하는 텍스트 데이터('M', 'F', '26-35' 등)를 컴퓨터용 숫자(0, 1, 2...)로 변환합니다.
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 
                    'Gender_Age', 'Age_City', 'Gender_City']
for col in categorical_cols:
    le = LabelEncoder()
    user_df[col] = le.fit_transform(user_df[col].astype(str))

# 예측에 사용할 9개의 특성(X)과 맞추고자 하는 정답 변수인 총 구매 금액(y)을 설정합니다.
feature_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 
                'Marital_Status', 'Gender_Age', 'Age_City', 'Gender_City']
X = user_df[feature_cols].values
y = user_df['Purchase'].values.reshape(-1, 1)

# 전체 데이터를 학습용(80%)과 성능 검증용(20%) 데이터셋으로 쪼갭니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_SEED)

# [데이터 표준화] 변수마다 단위(단위 수치 범위)가 다르면 모델이 왜곡되므로, 평균 0, 표준편차 1인 정규분포 형태로 맞춥니다.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 타겟 변수(구매 금액)도 숫자가 너무 크기 때문에 학습 안정성을 위해 표준화합니다.
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_test_original = y_test.flatten()  # 나중에 오차 계산 시 원래 달러 단위와 비교하기 위해 1차원 배열로 보관

# 결과를 차곡차곡 담아 대시보드에 넘겨줄 딕셔너리들
results_r2 = {}
results_rmse = {}
results_time = {}        # 모델별 학습 소요 시간 저장용
feature_importances = {} # 모델별 변수 가중치 저장용

# =========================================================================
# 🛡️ [Scikit-learn 엄격한 검사 우회용 포장 클래스] 
# 파이토치로 만든 딥러닝 모델은 구조가 달라서 사이킷런의 '변수 중요도 검사 함수'에 넣으면 에러가 납니다.
# 이를 해결하기 위해 파이토치 모델을 마치 사이킷런의 회귀 모델인 것처럼 보이게 겉면을 포장(Wrap)해주는 클래스입니다.
# =========================================================================
class PyTorchSklearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, torch_model, device):
        self.torch_model = torch_model
        self.device = device
        self._estimator_type = "regressor" # 사이킷런 검사기를 속이기 위한 태그 정보
        
    def fit(self, X, y):
        return self # 딥러닝 학습은 하단에서 따로 진행하므로 구조만 유지합니다.
        
    def predict(self, X):
        self.torch_model.eval() # 예측 모드 전환 (배치 정규화 등 일시 정지)
        with torch.no_grad():   # 메모리 절약을 위해 기울기(Gradient) 계산 비활성화
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.torch_model(X_tensor).cpu().numpy()
        return preds.flatten()  # 사이킷런 규격에 맞춰 1차원 배열 형태로 예측값 반환

# =========================================================================
# 🌲 [Model 1] 머신러닝 ① - Random Forest (의사결정나무 앙상블)
# =========================================================================
print("\n🌲 [ML 1/2] Scikit-learn Random Forest 학습 중...")
start_time = time.time()  # ⏱️ 순수 모델 학습 시간 측정 시작

rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=GLOBAL_SEED, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_scaled.flatten())

results_time['ML: Random Forest'] = time.time() - start_time  # ⏱️ 소요 시간 저장

# 표준화된 수치로 예측된 값을 다시 원래 우리가 아는 '달러($) 단위'로 되돌립니다.
rf_pred = rf_model.predict(X_test_scaled).reshape(-1, 1)
rf_pred_org = scaler_y.inverse_transform(rf_pred).flatten()

# R2 스코어(%)와 RMSE(달러 오차)를 계산하여 저장합니다.
results_r2['ML: Random Forest'] = r2_score(y_test_original, rf_pred_org) * 100
results_rmse['ML: Random Forest'] = np.sqrt(mean_squared_error(y_test_original, rf_pred_org))
feature_importances['ML: Random Forest'] = rf_model.feature_importances_  # 트리 자체 모델이 제공하는 변수 가중치 저장

# =========================================================================
# 🚀 [Model 2] 머신러닝 ② - Hist Gradient Boosting (LightGBM 스타일 부스팅)
# 데이터를 구간별로 나누어(Histogram-based) 속도가 엄청나게 빠르고 성능이 강력한 최신 ML 알고리즘입니다.
# =========================================================================
print("🚀 [ML 2/2] Scikit-learn Hist Gradient Boosting 학습 중...")
start_time = time.time()  # ⏱️ 학습 시간 측정 시작

hgb_model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, random_state=GLOBAL_SEED)
hgb_model.fit(X_train_scaled, y_train_scaled.flatten())

results_time['ML: HistGB (LightGBM형)'] = time.time() - start_time  # ⏱️ 소요 시간 저장

hgb_pred = hgb_model.predict(X_test_scaled).reshape(-1, 1)
hgb_pred_org = scaler_y.inverse_transform(hgb_pred).flatten()

results_r2['ML: HistGB (LightGBM형)'] = r2_score(y_test_original, hgb_pred_org) * 100
results_rmse['ML: HistGB (LightGBM형)'] = np.sqrt(mean_squared_error(y_test_original, hgb_pred_org))

print("🔍 HistGB 모델 가중치 분석 중...")
# 부스팅 모델은 내장 가중치 기능이 약하므로, 데이터를 인위적으로 셔플하여 오차가 얼마나 커지는지 확인하는 공정한 가중치 연산 기법을 씁니다.
hgb_imp = permutation_importance(hgb_model, X_test_scaled, y_test_scaled.flatten(), random_state=GLOBAL_SEED)
feature_importances['ML: HistGB (LightGBM형)'] = hgb_imp.importances_mean / np.sum(hgb_imp.importances_mean) # 합계가 1(100%)이 되도록 정규화

# =========================================================================
# ⚡ 딥러닝(PyTorch) 공통 데이터 배치 및 파이프라인 구성
# =========================================================================
X_train_pt = torch.FloatTensor(X_train_scaled).to(pt_device)
y_train_pt = torch.FloatTensor(y_train_scaled).to(pt_device)
X_test_pt = torch.FloatTensor(X_test_scaled).to(pt_device)

# 미니배치 학습 시 멀티스레드 환경에서도 난수가 튀지 않도록 제어하는 보안 가이드 워커 함수
def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
g = torch.Generator().manual_seed(GLOBAL_SEED)
# 데이터를 256개씩 잘라서 딥러닝 모델에 던져주는 미니배치 로더 구축
pt_loader = DataLoader(TensorDataset(X_train_pt, y_train_pt), batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=g)

# =========================================================================
# 🔥 [Model 3] 딥러닝 ① - PyTorch Deep DNN (깊고 조밀한 신경망)
# 입력층부터 128 -> 64 -> 32 -> 1 순서로 계단식으로 깊게 파고드는 전형적인 심층 인공신경망입니다.
# 중간에 배치 정규화(BatchNorm1d)를 끼워 넣어 학습이 폭주하거나 멈추는 현상을 방지했습니다.
# =========================================================================
print("🔥 [DL 1/2] PyTorch Deep DNN 학습 중...")
class DeepDNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1) # 최종 출력은 구매 금액 예측값 1개
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(GLOBAL_SEED)
deep_model = DeepDNN(X_train.shape[1]).to(pt_device)
optimizer_deep = optim.Adam(deep_model.parameters(), lr=0.005) # 가중치를 업데이트할 최적화 알고리즘
criterion = nn.MSELoss() # 평균제곱오차 기준의 손실 함수

start_time = time.time()  # ⏱️ 딥러닝 순수 에포크 학습 시간 측정 시작
deep_model.train()
for epoch in range(30): # 30번 반복 학습
    for bx, by in pt_loader:
        optimizer_deep.zero_grad() # 이전 루프의 잔여 가중치 기울기 리셋
        loss = criterion(deep_model(bx), by) # 예측값과 실제값 오차 계산
        loss.backward() # 역전파 연산으로 에러를 뒤로 전달
        optimizer_deep.step() # 가중치 전격 업데이트
results_time['DL: PyTorch Deep DNN'] = time.time() - start_time  # ⏱️ 소요 시간 저장

deep_model.eval()
with torch.no_grad():
    deep_pred = deep_model(X_test_pt).cpu().numpy()
deep_pred_org = scaler_y.inverse_transform(deep_pred).flatten()

results_r2['DL: PyTorch Deep DNN'] = r2_score(y_test_original, deep_pred_org) * 100
results_rmse['DL: PyTorch Deep DNN'] = np.sqrt(mean_squared_error(y_test_original, deep_pred_org))

print("🔍 Deep DNN 모델 가중치 분석 중...")
# 위에서 선언한 Wrapper 포장지를 사용해 파이토치 모델을 사이킷런 함수에 안전하게 통과시킵니다.
deep_wrapper = PyTorchSklearnWrapper(deep_model, pt_device)
deep_imp = permutation_importance(deep_wrapper, X_test_scaled, y_test_scaled.flatten(), random_state=GLOBAL_SEED)
feature_importances['DL: PyTorch Deep DNN'] = deep_imp.importances_mean / np.sum(deep_imp.importances_mean)

# =========================================================================
# 🧠 [Model 4] 딥러닝 ② - PyTorch Wide NN (단층형 넓은 신경망)
# 은닉층을 깊게 파고들지 않는 대신, 첫 번째 레이어의 뉴런 수를 512개로 대폭 늘려
# 데이터를 한 번에 아주 넓고 방대하게 펼쳐서 복잡한 비선형 관계를 캐치하는 구조입니다.
# =========================================================================
print("🧠 [DL 2/2] PyTorch Wide NN 학습 중...")
class WideNN(nn.Module):
    def __init__(self, input_dim):
        super(WideNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1) # 곧바로 출력층으로 연결되는 구조
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(GLOBAL_SEED)
wide_model = WideNN(X_train.shape[1]).to(pt_device)
optimizer_wide = optim.Adam(wide_model.parameters(), lr=0.005)

start_time = time.time()  # ⏱️ 학습 시간 측정 시작
wide_model.train()
for epoch in range(30):
    for bx, by in pt_loader:
        optimizer_wide.zero_grad()
        loss = criterion(wide_model(bx), by)
        loss.backward()
        optimizer_wide.step()
results_time['DL: PyTorch Wide NN'] = time.time() - start_time  # ⏱️ 소요 시간 저장

wide_model.eval()
with torch.no_grad():
    wide_pred = wide_model(X_test_pt).cpu().numpy()
wide_pred_org = scaler_y.inverse_transform(wide_pred).flatten()

results_r2['DL: PyTorch Wide NN'] = r2_score(y_test_original, wide_pred_org) * 100
results_rmse['DL: PyTorch Wide NN'] = np.sqrt(mean_squared_error(y_test_original, wide_pred_org))

print("🔍 Wide NN 모델 가중치 분석 중...")
wide_wrapper = PyTorchSklearnWrapper(wide_model, pt_device)
wide_imp = permutation_importance(wide_wrapper, X_test_scaled, y_test_scaled.flatten(), random_state=GLOBAL_SEED)
feature_importances['DL: PyTorch Wide NN'] = wide_imp.importances_mean / np.sum(wide_imp.importances_mean)


# =========================================================================
# 🎹 [시각화 1] 모델별 종합 성능 및 연산 시간 대조 스펙트럼 (대시보드 1)
# 세련된 태블로(Tableau) 스타일을 모방하여 미니멀하고 직관적으로 커스텀 튜닝한 차트입니다.
# =========================================================================
models_list = list(results_r2.keys())
r2_values = list(results_r2.values())
rmse_values = list(results_rmse.values())
time_values = list(results_time.values())
tableau_grayscale = ['#7F8C8D', '#5D6D7E', '#2C3E50', '#1A252F'] # 태블로 특유의 고급스러운 모노톤 헥사코드

# 가로로 3개의 독립된 격자 그래프(1행 3열)를 담을 거대한 도화지를 생성합니다.
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7.5), facecolor='#FFFFFF')
fig1.suptitle('🚨 [대시보드 1] AI 아키텍처별 최종 성능 및 학습 소요 시간 대조 분석', fontsize=15, fontweight='bold', color='#111111', y=0.96)

# 3개의 그래프 공통 스타일 미니멀화 작업 루프 (투박한 기본 테두리를 날립니다.)
for ax in axes1:
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)     # 상단 테두리 제거
    ax.spines['right'].set_visible(False)   # 우측 테두리 제거
    ax.spines['left'].set_color('#E0E0E0')   # 좌측 테두리는 연한 그레이
    ax.spines['bottom'].set_color('#E0E0E0') # 하단 테두리 연한 그레이
    ax.grid(axis='y', linestyle=':', alpha=0.5, color='#BBBBBB') # 은은한 Y축 배경 가이드 점선 배치

# 📊 차트 ①: R2 Score (데이터 설명력 모델 성적표)
bars1 = axes1[0].bar(models_list, r2_values, color=tableau_grayscale, width=0.45)
axes1[0].set_title('① 모델별 데이터 설명력 (R2 Score - 높을수록 우수)', fontweight='bold', fontsize=11, pad=15)
axes1[0].set_ylabel('정확성 수치 (%)', fontsize=10)
axes1[0].set_ylim(0, max(r2_values) + 3) # 숫자가 겹치지 않도록 Y축 상단 마진 확보
axes1[0].set_xticks(range(len(models_list)))
axes1[0].set_xticklabels(models_list, rotation=20, ha='right') # 텍스트가 겹치지 않게 20도 회전 수평 정렬

# 막대 상단에 구체적인 정확도 수치 텍스트 매핑
for bar in bars1:
    yval = bar.get_height()
    axes1[0].text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval:.2f}%', va='bottom', ha='center', fontweight='bold', fontsize=9)

# 📊 차트 ②: RMSE (평균 오차 금액 단위 분석)
bars2 = axes1[1].bar(models_list, rmse_values, color=tableau_grayscale, width=0.45)
axes1[1].set_title('② 평균 예측 오차 수준 (RMSE - 낮을수록 우수)', fontweight='bold', fontsize=11, pad=15)
axes1[1].set_ylabel('평균 오차 금액 ($)', fontsize=10)
axes1[1].set_ylim(0, max(rmse_values) + 80000)
axes1[1].set_xticks(range(len(models_list)))
axes1[1].set_xticklabels(models_list, rotation=20, ha='right')

# 막대 상단에 컴마가 포함된 달러($) 포맷 텍스트 매핑
for bar in bars2:
    yval = bar.get_height()
    axes1[1].text(bar.get_x() + bar.get_width()/2, yval + 10000, f'${yval:,.0f}', va='bottom', ha='center', fontweight='bold', fontsize=9)

# 📊 차트 ③: Time (하드웨어 연산 효율성 평가)
bars3 = axes1[2].bar(models_list, time_values, color=tableau_grayscale, width=0.45)
axes1[2].set_title('③ 모델별 학습 소요 시간 (Time - 낮을수록 효율적)', fontweight='bold', fontsize=11, pad=15)
axes1[2].set_ylabel('소요 시간 (초)', fontsize=10)
axes1[2].set_ylim(0, max(time_values) + (max(time_values) * 0.15))
axes1[2].set_xticks(range(len(models_list)))
axes1[2].set_xticklabels(models_list, rotation=20, ha='right')

# 막대 상단에 소수점 2자리 초 단위 텍스트 매핑
for bar in bars3:
    yval = bar.get_height()
    axes1[2].text(bar.get_x() + bar.get_width()/2, yval + (max(time_values) * 0.02), f'{yval:.2f}초', va='bottom', ha='center', fontweight='bold', fontsize=9)

fig1.tight_layout() # 컴포넌트 간 여백 자동 최적화 정렬
fig1.subplots_adjust(top=0.85, bottom=0.20, wspace=0.28) # 상단 메인 타이틀 및 좌우 차트 간격 미세 조정


# =========================================================================
# 🎹 [시각화 2] 4대 아키텍처별 변수 중요도 분석 (대시보드 2)
# 각 인공지능 알고리즘 내부가 유저 프로필 변수들을 어떤 중요도 비율로 해석했는지 대조 분석합니다.
# =========================================================================
print("\n🔮 모델별 변수 가중치(Feature Importance) 대시보드를 구축하고 있습니다...")
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), facecolor='#FFFFFF') # 2행 2열 멀티 차트 도화지 구성
axes2 = axes2.flatten() # 2차원 매트릭스 배열을 1차원 순차 배열로 풀어 루프 돌리기 편하게 변환
fig2.suptitle('💡 [대시보드 2] 아키텍처 내부 가중치 분석 (Feature Importance - 세로형)', fontsize=15, fontweight='bold', color='#111111', y=0.97)

for i, model_name in enumerate(models_list):
    ax = axes2[i]
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.grid(axis='y', linestyle=':', alpha=0.4, color='#BBBBBB')
    
    importances = feature_importances[model_name] * 100  # 직관성을 위해 영향력 합산 수치를 백분율(%)로 환산
    
    # 가중치가 높은 순서대로 변수들을 역순 정렬(내림차순)하는 인덱스 추출 파트
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[idx] for idx in sorted_idx]
    sorted_vals = importances[sorted_idx]
    
    # 정렬된 가중치 순서대로 세로 막대 배치
    bars = ax.bar(sorted_features, sorted_vals, color=tableau_grayscale[i], width=0.5, edgecolor='#FFFFFF', alpha=0.9)
    
    ax.set_title(f'📌 {model_name}', fontweight='bold', fontsize=11, color='#222222', loc='left')
    ax.set_ylabel('변수 영향력 비율 (%)', fontsize=9, color='#555555')
    ax.set_ylim(0, max(sorted_vals) + 8)
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=25, ha='right', fontsize=9) # 글자가 겹치지 않게 우하향 25도 회전
    
    # 의미 있는 수치(0.5% 초과 가중치)를 가진 변수들 위에 퍼센트 수치 기재
    for bar in bars:
        yval = bar.get_height()
        if yval > 0.5:  
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', va='bottom', ha='center', fontsize=8, fontweight='bold', color='#444444')

fig2.tight_layout()
fig2.subplots_adjust(top=0.90, bottom=0.12, hspace=0.35, wspace=0.22)

plt.show(block=True) # 파이썬 백그라운드 연산이 완전히 끝나도 팝업 창이 꺼지지 않고 화면에 강제 고정 유지