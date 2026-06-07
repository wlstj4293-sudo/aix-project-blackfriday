from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time  # ⏱️ 시간 측정을 위한 내장 라이브러리 추가

# 1. 머신러닝 및 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. 딥러닝 (PyTorch 2종 아키텍처)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================================================================
# 🔒 [재현성 보장] 글로벌 난수 및 시드 잠금
# =========================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(GLOBAL_SEED)

pt_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🍏 PyTorch 연산 장치: {pt_device}")

# =========================================================================
# 🔤 [폰트 세팅] 글자 오류 완벽 차단 포맷
# =========================================================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['AppleGothic', 'sans-serif']
plt.ion()

# =========================================================================
# 📊 [데이터 로드 및 순수 프로필 전처리]
# =========================================================================
print("📊 블랙프라이데이 데이터셋을 로드하고 있습니다...")
current_dir = Path(__file__).resolve().parent
file_path = current_dir / 'BFtrain.csv'
if not file_path.exists():
    file_path = current_dir / 'archive' / 'BFtrain.csv'

raw_df = pd.read_csv(file_path)
raw_df.columns = raw_df.columns.str.strip()

user_df = raw_df.groupby('User_ID').agg({
    'Gender': 'first', 'Age': 'first', 'Occupation': 'first',
    'City_Category': 'first', 'Stay_In_Current_City_Years': 'first', 'Marital_Status': 'first',
    'Purchase': 'sum'
}).reset_index()

user_df['Gender_Age'] = user_df['Gender'] + "_" + user_df['Age']
user_df['Age_City'] = user_df['Age'] + "_" + user_df['City_Category']
user_df['Gender_City'] = user_df['Gender'] + "_" + user_df['City_Category']

categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 
                    'Gender_Age', 'Age_City', 'Gender_City']
for col in categorical_cols:
    le = LabelEncoder()
    user_df[col] = le.fit_transform(user_df[col].astype(str))

feature_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 
                'Marital_Status', 'Gender_Age', 'Age_City', 'Gender_City']
X = user_df[feature_cols].values
y = user_df['Purchase'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=GLOBAL_SEED)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
y_test_original = y_test.flatten()

# =========================================================================
# ⛓️ [City_Category 100% 가중치 강제 마스킹 알고리즘]
# =========================================================================
target_feature = 'City_Category'
target_idx = feature_cols.index(target_feature)

controlled_importance = np.zeros(len(feature_cols))
controlled_importance[target_idx] = 1.0  # City_Category에게 지분 100% 몰아주기

X_train_controlled = X_train_scaled * controlled_importance
X_test_controlled = X_test_scaled * controlled_importance

results_r2 = {}
results_rmse = {}
results_time = {}  # ⏱️ 모델별 학습 소요 시간 저장용 딕셔너리 추가

# =========================================================================
# 🌲 [Model 1] 머신러닝 ① - Random Forest
# =========================================================================
print(f"\n🌲 [ML 1/2] {target_feature} 100% 기반 Random Forest 학습 중...")
start_time = time.time()  # ⏱️ 시간 측정 시작

rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=GLOBAL_SEED, n_jobs=-1)
rf_model.fit(X_train_controlled, y_train_scaled.flatten())

results_time['ML: Random Forest'] = time.time() - start_time  # ⏱️ 소요 시간 저장

rf_pred = rf_model.predict(X_test_controlled).reshape(-1, 1)
rf_pred_org = scaler_y.inverse_transform(rf_pred).flatten()

results_r2['ML: Random Forest'] = r2_score(y_test_original, rf_pred_org) * 100
results_rmse['ML: Random Forest'] = np.sqrt(mean_squared_error(y_test_original, rf_pred_org))

# =========================================================================
# 🚀 [Model 2] 머신러닝 ② - Hist Gradient Boosting
# =========================================================================
print(f"🚀 [ML 2/2] {target_feature} 100% 기반 Hist Gradient Boosting 학습 중...")
start_time = time.time()  # ⏱️ 시간 측정 시작

hgb_model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, random_state=GLOBAL_SEED)
hgb_model.fit(X_train_controlled, y_train_scaled.flatten())

results_time['ML: HistGB (LightGBM형)'] = time.time() - start_time  # ⏱️ 소요 시간 저장

hgb_pred = hgb_model.predict(X_test_controlled).reshape(-1, 1)
hgb_pred_org = scaler_y.inverse_transform(hgb_pred).flatten()

results_r2['ML: HistGB (LightGBM형)'] = r2_score(y_test_original, hgb_pred_org) * 100
results_rmse['ML: HistGB (LightGBM형)'] = np.sqrt(mean_squared_error(y_test_original, hgb_pred_org))

# =========================================================================
# ⚡ 딥러닝 통제 데이터 공통 준비
# =========================================================================
X_train_pt = torch.FloatTensor(X_train_controlled).to(pt_device)
y_train_pt = torch.FloatTensor(y_train_scaled).to(pt_device)
X_test_pt = torch.FloatTensor(X_test_controlled).to(pt_device)

def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
g = torch.Generator().manual_seed(GLOBAL_SEED)
pt_loader = DataLoader(TensorDataset(X_train_pt, y_train_pt), batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=g)

# =========================================================================
# 🔥 [Model 3] 딥러닝 ① - PyTorch Deep DNN
# =========================================================================
print(f"🔥 [DL 1/2] {target_feature} 100% 기반 PyTorch Deep DNN 학습 중...")
class ControlledDeepDNN(nn.Module):
    def __init__(self, input_dim):
        super(ControlledDeepDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(GLOBAL_SEED)
deep_model = ControlledDeepDNN(X_train.shape[1]).to(pt_device)
optimizer_deep = optim.Adam(deep_model.parameters(), lr=0.005)
criterion = nn.MSELoss()

start_time = time.time()  # ⏱️ 시간 측정 시작
deep_model.train()
for epoch in range(30):
    for bx, by in pt_loader:
        optimizer_deep.zero_grad()
        loss = criterion(deep_model(bx), by)
        loss.backward()
        optimizer_deep.step()
results_time['DL: PyTorch Deep DNN'] = time.time() - start_time  # ⏱️ 소요 시간 저장

deep_model.eval()
with torch.no_grad():
    deep_pred = deep_model(X_test_pt).cpu().numpy()
deep_pred_org = scaler_y.inverse_transform(deep_pred).flatten()

results_r2['DL: PyTorch Deep DNN'] = r2_score(y_test_original, deep_pred_org) * 100
results_rmse['DL: PyTorch Deep DNN'] = np.sqrt(mean_squared_error(y_test_original, deep_pred_org))

# =========================================================================
# 🧠 [Model 4] 딥러닝 ② - PyTorch Wide NN
# =========================================================================
print(f"🧠 [DL 2/2] {target_feature} 100% 기반 PyTorch Wide NN 학습 중...")
class ControlledWideNN(nn.Module):
    def __init__(self, input_dim):
        super(ControlledWideNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x): return self.net(x)

torch.manual_seed(GLOBAL_SEED)
wide_model = ControlledWideNN(X_train.shape[1]).to(pt_device)
optimizer_wide = optim.Adam(wide_model.parameters(), lr=0.005)

start_time = time.time()  # ⏱️ 시간 측정 시작
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


# =========================================================================
# 🎹 [최종 시각화] 성능 대조 및 시간 대조 분석 그래프 (3열 확장)
# =========================================================================
print("\n🖥️ 세로형 성능 및 시간 대조 대시보드를 생성하고 있습니다...")

models_list = list(results_r2.keys())
r2_values = list(results_r2.values())
rmse_values = list(results_rmse.values())
time_values = list(results_time.values())  # 시간 데이터 추가
tableau_grayscale = ['#7F8C8D', '#5D6D7E', '#2C3E50', '#1A252F']

# 원래 첫 번째 코드의 형태인 1행 3열 구조로 변경 및 전체 가로 폭 확장(14 -> 18)
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7.5), facecolor='#FFFFFF')
fig1.suptitle(f'🚨 [실험] {target_feature} 가중치 100% 부여 시 아키텍처별 최종 성능 및 학습 소요 시간 대조', 
             fontsize=14, fontweight='bold', color='#111111', y=0.96)

for ax in axes1:
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.tick_params(axis='both', colors='#444444', labelsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5, color='#BBBBBB')

# 📊 차트 ①: R2 Score 수직 그래프
bars1 = axes1[0].bar(models_list, r2_values, color=tableau_grayscale, width=0.45)
axes1[0].set_title(f'① {target_feature} 단독 학습 시 설명력 (R2 Score)', fontweight='bold', fontsize=11, pad=15)
axes1[0].set_ylabel('정확성 수치 (%)', fontsize=9)
axes1[0].set_ylim(min(0, min(r2_values) - 5), max(r2_values) + 5)
axes1[0].set_xticks(range(len(models_list)))
axes1[0].set_xticklabels(models_list, rotation=20, ha='right')

for bar in bars1:
    yval = bar.get_height()
    axes1[0].text(bar.get_x() + bar.get_width()/2, yval + (0.3 if yval >= 0 else -1.5), f'{yval:.2f}%', va='bottom', ha='center', fontweight='bold', fontsize=9)

# 📊 차트 ②: RMSE 오차 수직 그래프
bars2 = axes1[1].bar(models_list, rmse_values, color=tableau_grayscale, width=0.45)
axes1[1].set_title(f'② {target_feature} 단독 학습 시 오차 수준 (RMSE)', fontweight='bold', fontsize=11, pad=15)
axes1[1].set_ylabel('평균 오차 금액 ($)', fontsize=9)
axes1[1].set_ylim(0, max(rmse_values) + 80000)
axes1[1].set_xticks(range(len(models_list)))
axes1[1].set_xticklabels(models_list, rotation=20, ha='right')

for bar in bars2:
    yval = bar.get_height()
    axes1[1].text(bar.get_x() + bar.get_width()/2, yval + 10000, f'${yval:,.0f}', va='bottom', ha='center', fontweight='bold', fontsize=9)

# 📊 차트 ③: Time 효율성 수직 그래프 (시간 칸 추가 및 복원)
bars3 = axes1[2].bar(models_list, time_values, color=tableau_grayscale, width=0.45)
axes1[2].set_title('③ 모델별 학습 소요 시간 (Time - 낮을수록 효율적)', fontweight='bold', fontsize=11, pad=15)
axes1[2].set_ylabel('소요 시간 (초)', fontsize=9)
axes1[2].set_ylim(0, max(time_values) + (max(time_values) * 0.15))
axes1[2].set_xticks(range(len(models_list)))
axes1[2].set_xticklabels(models_list, rotation=20, ha='right')

for bar in bars3:
    yval = bar.get_height()
    axes1[2].text(bar.get_x() + bar.get_width()/2, yval + (max(time_values) * 0.02), f'{yval:.2f}초', va='bottom', ha='center', fontweight='bold', fontsize=9)

fig1.tight_layout()
fig1.subplots_adjust(top=0.85, bottom=0.20, wspace=0.28)

# 오직 하나의 대시보드 창만 화면에 유지
plt.show(block=True)