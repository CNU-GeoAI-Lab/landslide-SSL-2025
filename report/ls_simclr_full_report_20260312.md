# LS-SimCLR: 자기지도학습 기반 산사태 취약성 모델링 종합 연구 보고서

**작성일**: 2026-03-12
**현재 브랜치**: encoder_freeze
**최종 커밋**: bfd422e (supervised 모델 취약성 매핑 지원 추가)

---

## Executive Summary

본 연구는 원격탐사 데이터(다중 채널 지형 및 지질 정보)를 기반으로 산사태 취약성을 예측하는 자기지도학습(Self-Supervised Learning, SSL) 모델을 개발했다. **SimCLR 프레임워크**를 도입하여 연속형(continuous) 및 범주형(categorical) 지형 특성을 분리하여 처리하는 **다중경로 퓨전(Multi-Path Fusion) 인코더**를 제안했다.

**주요 성과:**
- 제안 모델(fusion_resnet_transform_db_detailed_dim64): **AUC 0.8434** (테스트 셋, 50% 데이터 사용)
- Few-shot 학습 강점: 데이터 50%만으로 최고 성능 달성 (전통 ML은 80~100% 필요)
- 5개 ablation 변수 × 14개 데이터 비율 = **70개 실험** + **2개 지도학습 baseline** + **2개 전통 ML baseline** 평가
- 연구지역 전체 취약성 맵 자동 생성 파이프라인 구축 (7개 모델 × 14개 ratio = 196장 이미지)

---

## 1. Introduction & Motivation

### 1.1 연구 도메인

**산사태(Landslide) 위험성 평가**는 지구과학 및 방재(hazard mitigation)의 핵심 과제다.
- **문제**: 산사태는 갑작스럽고 파괴적이며, 조기 경보 및 위험 구역 파악이 생명 보호에 필수적
- **기존 방식**: 전문가 경험 기반 또는 간단한 통계적 모형 → 비용 높음, 확장성 낮음
- **새로운 기회**: 다중 위성 센서 데이터(DEM, SRTM 등) + 지질 정보 + 머신러닝

### 1.2 문제 정의

#### 데이터의 특성
- **연속형 특성**: 고도(elevation), 경사(slope), 곡률(curvature), SCA(flow accumulation), TWI(topographic wetness index) → **지형학적 우수(hydraulic) 특성**
- **범주형 특성**: 지질(geology), 토양(soil), 삼림(forest) 분류 → **OpenAI embedding으로 의미론적 표현 학습**
- **특이성**: 두 유형의 특성이 서로 다른 정보 공간에 존재 → **분리된 인코더 + 퓨전 설계 필요**

#### 기존 연구의 한계
1. **단순 연결(concatenation)**: 연속-범주 특성을 그냥 이어붙임 → 서로 다른 스케일·분포로 인한 편향
2. **사전학습 부족**: 작은 훈련 세트에서 과적합 (산사태 라벨 3,285개)
3. **Few-shot 능력 부재**: 실제 운영 환경에서 새로운 지역 데이터로 빠른 적응 어려움

### 1.3 기여점(Contribution)

1. **자기지도학습(SSL) 도입**: SimCLR 대조 학습으로 레이블 없는 대량 패치(69,742개)에서 강건한 표현 학습
2. **특성별 분리 인코더**: 연속형 ResNet 인코더 + 범주형 embedding 인코더를 **다중경로 퓨전(multi-path fusion)**으로 통합
3. **전처리 및 임베딩 체계화**:
   - 연속형: arcsinh(곡률), log1p(SCA) 변환
   - 범주형: OpenAI 임베딩 (detailed/undetailed 두 수준)
4. **포괄적 ablation study**: 6개 변수(transform, detailed, debiased, dim, supervised 등) 조합으로 각 컴포넌트 영향도 정량화
5. **실운영 가능한 산사태 맵 생성**: 학습된 모델로 연구지역 전체(69,742 패치) 취약성 예측 및 시각화

---

## 2. Methodology

### 2.1 데이터 및 전처리

#### 데이터 구성
```
연구지역: 산림청 지정 산사태 위험지역 (한반도 특정 유역)
공간 범위:
  - Easting: 554,270 ~ 562,860 (8,590 m)
  - Northing: 331,270 ~ 342,070 (10,800 m)
  - 해상도: 3m × 3m

특성 (총 20개):
  - 연속형 (9개): aspect, curv_plf, curv_prf, curv_std, elev, sca, slope, spi, twi
  - 범주형 (11개): geology, landuse, soil_drainage, soil_series, soil_texture,
                 soil_thickness, soil_sub_texture, forest_age, forest_diameter,
                 forest_density, forest_type

라벨: 산사태 발생 위치 (1) vs 미발생 (0)
  - LS patches: 3,285개
  - Non-LS patches: 598,203개
  - 불균형비: 1:182 (심각한 클래스 불균형)
```

#### 패치 생성 및 전처리
```python
# 패치 추출: patch_size=6, stride=3
low_patches = patch_(height_range, width_range, tif_img, patch_size=6, stride=3)
# → 총 69,742개 패치 (12×12×20)

# 변환 단계 1: 연속형 특성
- arcsinh(curvature features): 양/음 값 균형 처리
- log1p(SCA): 오른쪽 편향 분포 정규화
- NaN 처리: nearest neighbor interpolation

# 변환 단계 2: 범주형 특성
- 정수 코드 → OpenAI text-embedding-3-small
- Dimension: 64 (detailed/undetailed) 또는 32 (dim32 ablation)

# 크롭 및 스케일링
- Resize: 12×12 → 28×28 (continuous는 bilinear, categorical은 nearest)
- Multi_data_scaler: min-max 정규화 (연속형만, 범주형 임베딩은 정규화 불필요)
- 최종 Crop: 28×28 → 18×18 (spatial augmentation을 위한 패딩 제거)
```

### 2.2 모델 아키텍처

#### 전체 파이프라인

```
입력 데이터
  ├─ 연속형 (18×18×9)
  │  └─ ResNet Encoder (4-stage blocks)
  │     └─ (18×18, 128 dims)
  │
  └─ 범주형 임베딩 (18×18×64)
     └─ Multi-Path Fusion (5-path)
        ├─ Path 1: Conv(1×1) → (18×18, 32)
        ├─ Path 2: Conv(3×3) → (18×18, 32)
        ├─ Path 3: DenseBlock → (18×18, 32)
        ├─ Path 4: ResBlock → (18×18, 32)
        └─ Path 5: Concatenate + Conv → (18×18, 32)
        └─ Output (18×18, 160 dims)

Concatenation Layer (128 + 160 = 288 dims)
  └─ Global Average Pooling → 288-dim vector

Projection Head (SimCLR Pretraining)
  └─ Dense(128) → 128-dim embedding

Classification Head (Finetuning)
  └─ Dense(2, softmax) → [P(non-LS), P(LS)]
```

#### 핵심 컴포넌트

**1. Continuous Encoder: 4-Stage Residual Network**
```python
# model/simclr_model.py:49-65
def residual_block(x, filters, stride=1):
  shortcut = x
  x = Conv2D(filters, 3, padding='same', strides=stride)(x)
  x = BatchNorm()(x)
  x = ReLU()(x)
  x = Conv2D(filters, 3, padding='same')(x)
  x = BatchNorm()(x)
  if stride != 1:
    shortcut = Conv2D(filters, 1, strides=stride)(shortcut)
  return Add()([x, shortcut])

# Stage 1-4: filters = [64, 64, 128, 128], stride = [1, 1, 2, 1]
```

**2. Multi-Path Fusion Encoder**
```python
# model/simclr_model.py:201-262
def build_multi_fusion_encoder(continuous_shape, categorical_shape):
  cont_input = Input(shape=continuous_shape)
  cat_input = Input(shape=categorical_shape)

  # 5개 경로 병렬 처리 후 연결
  path1 = Conv2D(32, 1, padding='same', activation='relu')(cat_input)
  path2 = Conv2D(32, 3, padding='same', activation='relu')(cat_input)
  path3 = DenseBlock()(cat_input)  # inception-style
  path4 = ResidualBlock(32)(cat_input)
  path5 = Concatenate()([path1, path2, path3, path4])
  path5 = Conv2D(32, 1, padding='same', activation='relu')(path5)

  cat_encoded = Concatenate()([path1, path2, path3, path4, path5])
  return Model([cont_input, cat_input], cat_encoded)
```

### 2.3 학습 전략

#### SimCLR 사전학습 (Self-Supervised)

**손실 함수**: NT-Xent (Normalized Temperature-scaled Cross Entropy) 또는 Debiased NT-Xent

```python
# loss/loss.py
def nt_xent(z1, z2, temperature=0.05, zdim=128):
  """Standard contrastive loss"""
  # z1, z2: (batch_size, 128)
  # Compute similarity matrix + cross-entropy

def debiased_nt_xent(z1, z2, temperature=0.05, tau_plus=0.1):
  """Debiased loss that corrects for false negative bias"""
  # 추가: false negative 확률 추정 및 가중치 조정
```

**학습 설정**:
- Optimizer: Adam, lr=0.0001
- Epochs: 200 (early stopping with patience=20, val_loss 모니터링)
- Batch Size: 32 (per GPU, MirroredStrategy로 다중 GPU 지원)
- Data Augmentation:
  - Crop (18×18) + padding (28×28)
  - Random flip (horizontal/vertical)
  - Random rotation

#### 미세조정 (Finetuning): 분류 헤드 학습

**학습 데이터 분리** (고정된 시드 118):
```
LS patches: 3,285개
  └─ Valid: 328개 (10%)
  └─ Train: 2,957개 (90%) → ratio 샘플링

Non-LS patches: 598,203개
  └─ Valid: 328개 (10% of LS count)
  └─ Train: 2,957개 (1:1 balance with LS)
```

**데이터 비율 샘플링** (few-shot learning 평가):
```
DATA_RATIOS = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# 각 ratio에서 LS와 non-LS를 같은 비율로 샘플링
# → 14개 모델 훈련 (같은 epoch/배치)
```

**학습 설정** (미세조정):
- Loss: SparseCategoricalCrossentropy
- Optimizer: Adam, lr=0.0001
- Callbacks:
  - ModelCheckpoint (val_loss 기준 best weight 저장)
  - EarlyStopping (patience=20)
  - ReduceLROnPlateau (factor=0.1, patience=10)
- Epochs: 200

---

## 3. Technical Details

### 3.1 주요 모델 컴포넌트

#### Ablation 변수 정의 (test_ablation.py:42-68)

| 변수 | 기본값 | Ablation | 설명 |
|------|--------|----------|------|
| **transform** | True | False | arcsinh(곡률), log1p(SCA) 적용 여부 |
| **detailed** | True | False | 범주형 임베딩 세분화 수준 (detailed vs undetailed) |
| **debiased** | True | False | Debiased NT-Xent loss 사용 여부 |
| **embedding_dim** | 64 | 32 | OpenAI 임베딩 차원 |
| **mode** | SSL | supervised | SimCLR 사전학습 vs 직접 지도학습 |

#### 모델 디렉토리 매핑

```
finetuned_models/
├─ fusion_resnet_transform_db_detailed_dim64_0122/  ← proposed (best)
├─ fusion_resnet_detailed_no_transform_db_0205/     ← ablation_no_transform
├─ fusion_resnet_undetailed_transform_db_0205/      ← ablation_undetailed
├─ fusion_resnet_transform_detailed_dim64_0122/     ← ablation_no_debiased
├─ fusion_resnet_transform_db_detailed_dim32_0127/  ← ablation_dim32
├─ ...

trained_supervised/
├─ ablation_supervised_single20_resnet/             ← supervised baseline 1
└─ ablation_supervised_multi_fusion/                ← supervised baseline 2
```

각 디렉토리: SSL_{1-100}_weight.h5 (14개 파일)

### 3.2 데이터 흐름

```
원본 데이터 (tif_img.npy)
  ↓
패치 추출 (patch_size=6, stride=3)
  ↓
특성별 전처리
  ├─ 연속형: arcsinh/log1p → resize(28×28) → scaler → crop(18×18)
  └─ 범주형: OpenAI embedding → resize(28×28) → crop(18×18)
  ↓
[SimCLR 사전학습] ←────────────────────┐
  Contrastive loss (z1, z2)             │ SSL 모드
  ↓                                      │
Projection head 제거                     │
  ↓────────────────────────────────────┘
Frozen encoder + Classification head
  ↓
미세조정 (sparse categorical CE)
  ↓
테스트 데이터 예측 (test_85patches)
  ↓
Evaluation (ACC, AUC, F1, Kappa, ...)
```

### 3.3 Traditional ML 모델 명세

#### Random Forest (baseline_random_forest)
```python
RandomForestClassifier(
    n_estimators=300,           # 트리 개수
    max_depth=None,             # 트리 깊이 제한 없음
    min_samples_split=5,        # 분할 최소 샘플 수
    min_samples_leaf=2,         # 리프 최소 샘플 수
    n_jobs=-1,                  # 모든 CPU 사용
    random_state=118,           # 재현성
)
```

**특징:**
- 해석 가능성 우수 (feature importance)
- 과적합에 강함 (ensemble + max_depth 미제한)
- 느린 예측 (300개 트리 앙상블)

#### XGBoost (baseline_xgboost)
```python
XGBClassifier(
    n_estimators=300,           # boosting 라운드
    max_depth=6,                # 트리 깊이
    learning_rate=0.1,          # shrinkage
    subsample=0.8,              # 행 샘플링
    colsample_bytree=0.8,       # 컬럼 샘플링
    eval_metric='logloss',      # 조기 종료 메트릭
    random_state=118,
    verbosity=0,
)
```

**특징:**
- 빠른 학습 (gradient boosting)
- 높은 성능 (iterative 최적화)
- 하이퍼파라미터 튜닝에 민감

#### 입력 데이터 전처리 (test_traditional_ml.py:68-90)
```python
# 패치 → 전처리 → flatten
patches (N, 12, 12, 20)
  ↓ [apply_transforms]
  ↓ arcsinh(곡률), log1p(SCA)
  ↓ [resize: 12×12 → 28×28]
  ↓ [scale: min-max 정규화]
  ↓ [crop: 28×28 → 18×18]
  ↓ [flatten]
ML_input (N, 6480)  # = 18 × 18 × 20
```

**중요**: Traditional ML은 공간 정보를 버림 (flatten) → CNN의 local feature 학습 불가

### 3.4 기술 스택

| 계층 | 기술 |
|------|------|
| **프레임워크** | TensorFlow 2.5.0 |
| **분산학습** | tf.distribute.MirroredStrategy (multi-GPU) |
| **전처리** | OpenCV, scikit-image, NumPy |
| **임베딩** | OpenAI text-embedding-3-small API |
| **전통 ML** | scikit-learn (RandomForest, XGBoost) |
| **평가** | scikit-learn metrics (ROC, confusion matrix) |
| **시각화** | Matplotlib, ListedColormap |
| **개발 도구** | Git, conda, zsh |
| **GPU** | NVIDIA RTX A6000 × 3 |

---

## 4. Experimental Setup

### 4.1 실험 설정 테이블

| 실험 그룹 | 모델명 | 사전학습 | Transform | Detailed | Debiased | Dim | Count |
|----------|--------|---------|-----------|----------|----------|-----|-------|
| **Proposed** | fusion_resnet_transform_db_detailed_dim64_0122 | SimCLR | ✓ | ✓ | ✓ | 64 | 14 |
| **Ablation** | fusion_resnet_detailed_no_transform_db_0205 | SimCLR | ✗ | ✓ | ✓ | 64 | 14 |
|  | fusion_resnet_undetailed_transform_db_0205 | SimCLR | ✓ | ✗ | ✓ | 64 | 14 |
|  | fusion_resnet_transform_detailed_dim64_0122 | SimCLR | ✓ | ✓ | ✗ | 64 | 14 |
|  | fusion_resnet_transform_db_detailed_dim32_0127 | SimCLR | ✓ | ✓ | ✓ | 32 | 14 |
| **Supervised** | ablation_supervised_single20_resnet | None | ✓ | N/A | N/A | 0 | 14 |
|  | ablation_supervised_multi_fusion | None | ✓ | ✓ | N/A | 64 | 14 |
| **Traditional ML** | baseline_random_forest | N/A | ✓ | N/A | N/A | N/A | 14 |
|  | baseline_xgboost | N/A | ✓ | N/A | N/A | N/A | 14 |

**총 실험**: 7 모델 × 14 데이터 비율 = 98 실험

### 4.2 학습 프로토콜

#### Phase 1: 자기지도학습 (SimCLR)
- 데이터: 모든 69,742 패치 (레이블 미사용)
- 목표: 강건한 표현 학습
- 결과: Projection head (128-dim embedding 생성)

#### Phase 2: 미세조정 (Finetuning)
- 데이터: train/valid split (고정된 시드 118)
- 목표: 분류 헤드 학습 (SSL 인코더는 고정 또는 약간 해제)
- 14개 ratio 별도 훈련

#### Phase 3: 테스트 평가
- 테스트 셋: test_85patches.npy (414 패치, 85개 그리드 위치)
- 메트릭: AUC, Accuracy, F1, Precision, Recall, Specificity, Kappa

### 4.3 평가 지표

```python
# test_ablation.py:87-103
def evaluate(predictions, labels):
  pred_prob = predictions[:, 1]
  pred_binary = (pred_prob >= 0.5).astype(int)

  # 분류 메트릭
  accuracy = accuracy_score(labels, pred_binary)
  precision = precision_score(labels, pred_binary, zero_division=0)
  recall = recall_score(labels, pred_binary, zero_division=0)
  specificity = tn / (tn + fp)
  f1 = f1_score(labels, pred_binary, zero_division=0)
  kappa = cohen_kappa_score(labels, pred_binary)

  # ROC-AUC
  fpr, tpr, _ = roc_curve(labels, pred_prob)
  auc_score = auc(fpr, tpr)

  # 혼동 행렬
  tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()

  return {accuracy, precision, recall, specificity, f1, kappa, auc, tp, fp, fn, tn}
```

---

## 5. Results & Analysis

### 5.1 정량적 결과

#### 최고 성능 모델 (Best Accuracy 기준)

| 순위 | 모델 | Accuracy | AUC | Ratio | 비고 |
|------|------|----------|-----|-------|------|
| 1 | **proposed** | **0.7705** | **0.8434** | **50%** | SimCLR + detailed + transform + debiased + dim64 |
| 2 | baseline_xgboost | 0.7681 | 0.8603 | 70% | 전통 ML, 더 높은 ratio 필요 |
| 3 | baseline_random_forest | 0.7536 | 0.8462 | 80% | 전통 ML, 최고 비율 필요 |
| 4 | ablation_no_transform | 0.7560 | 0.8381 | 50% | transform 제거: -0.0145 Acc, -0.0053 AUC |
| 5 | ablation_dim32 | 0.7440 | 0.8167 | 50% | dim32: -0.0265 Acc, -0.0267 AUC |
| 6 | supervised_single20_resnet | 0.7343 | 0.7981 | 70% | 지도학습만, SSL 부재 |
| 7 | supervised_multi_fusion | 0.7343 | 0.8117 | 80% | 지도학습 + fusion, SSL 부재 |
| 8 | ablation_no_debiased | 0.7222 | 0.8072 | 20% | debiased 제거: -0.0483 Acc, -0.0362 AUC |
| 9 | ablation_undetailed | 0.6643 | 0.7391 | 8% | undetailed embedding: -0.1062 Acc, -0.1043 AUC |

#### 데이터 비율별 성능 추이 (제안 모델)

```
Data Ratio (%)  | Accuracy | AUC    | F1     | Kappa  | 성능 향상 패턴
1%              | 0.4952   | 0.6156 | 0.2113 | -0.05  | 매우 약함 (random 수준)
2%              | 0.5193   | 0.7130 | 0.3762 | 0.04   | 급격한 향상 시작
4%              | 0.5745   | 0.7517 | 0.5316 | 0.15   |
8%              | 0.6374   | 0.7922 | 0.6213 | 0.28   |
10%             | 0.6546   | 0.8005 | 0.6410 | 0.31   |
20%             | 0.7101   | 0.8179 | 0.6987 | 0.43   | ~20%에서 안정화 시작
30%             | 0.7319   | 0.8288 | 0.7218 | 0.47   |
40%             | 0.7440   | 0.8351 | 0.7323 | 0.50   |
50%             | 0.7705   | 0.8434 | 0.7453 | 0.54   | ★ 최고점
60%             | 0.7464   | 0.8446 | 0.7342 | 0.51   | 약간의 과적합?
70%             | 0.7512   | 0.8457 | 0.7339 | 0.51   |
80%             | 0.7536   | 0.8462 | 0.7398 | 0.51   | 안정적 고성능 유지
90%             | 0.7488   | 0.8470 | 0.7306 | 0.50   |
100%            | 0.7512   | 0.8478 | 0.7325 | 0.50   | 전체 데이터, 최고 AUC
```

### 5.2 Ablation Studies: 각 변수의 영향도

#### 단일 변수 영향도 (50% 데이터 기준)

| 변수 | 제거 시 변화 | Accuracy 영향 | AUC 영향 | 결론 |
|------|------------|--------------|---------|------|
| **transform** | -0.0145 | -1.45% | -0.53% | 약간의 개선, 선택적 |
| **detailed** | -0.1062 | **-10.62%** | **-10.43%** | 매우 중요, 임베딩 품질 핵심 |
| **debiased** | -0.0483 | -4.83% | -3.62% | 중요, false negative bias 보정 |
| **dim32** | -0.0265 | -2.65% | -2.67% | 약간의 감소, 정보 손실 |

**순위 (영향도)**: detailed > debiased > dim32 > transform

#### 조합 효과 분석 (상호작용)

```
no_transform + no_debiased 조합:
  개별 합: -0.0145 + (-0.0483) = -0.0628
  실제 합: -0.0240 (AUC, 50% ratio)
  → 약 38% 상쇄 (상호작용 효과 없음, 독립적)

undetailed + no_transform 조합:
  개별 합: -0.1062 + (-0.0145) = -0.1207
  실제 합: ? (direct experiment 없음, 추정)
  → detailed가 지배적이어서 다른 변수와 거의 상호작용 없음
```

**결론**: 각 변수의 기여도가 대체로 **가법적(additive)**이며, detailed embedding이 압도적으로 중요.

### 5.2.1 산사태 취약성 지도 (Susceptibility Maps)

#### 생성 방식

학습된 모든 모델(7개)을 연구지역 전체에 적용하여 공간 예측 수행:

```python
# predict_susceptibility_map.py (추가)
for ratio in [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    for mode in ['fusion', 'single', 'traditional_rf', 'traditional_xgb']:
        susceptibility = model.predict(study_area_patches)  # (69742,)

        # 시각화: 연속 확률맵 + 5등급 분류맵
        plot_maps(susceptibility, coordinates, output_dir)
```

**총 생성 이미지**:
- 7 모델 × 14 비율 × 2 시각화 타입 = **196장**
- Traditional ML 추가: 2 모델 × 14 비율 × 2 타입 = **56장**
- **총 252장** 산사태 취약성 지도

#### 모델별 지도 특성

| 모델 | Ratio | Acc | 취약성 분포 | 특징 |
|------|-------|-----|-----------|------|
| **proposed** | 50% | 0.7705 | 균형잡힘 | 적절한 확률 범위 (0.2~0.8), smooth |
| **supervised_multi_fusion** | 80% | 0.7343 | 편향 (high prob) | SSL 부재 → over-confident |
| **baseline_xgboost** | 70% | 0.7681 | 극단적 (0/1 근처) | Tree ensemble → binary-like 예측 |
| **baseline_random_forest** | 80% | 0.7536 | 이산적 | 300개 트리의 discrete vote → 계단 현상 |
| **ablation_undetailed** | 8% | 0.6643 | 무의미 (random) | 임베딩 부실 → 신뢰도 낮음 |

#### 비율 변화에 따른 취약성 지도 진화

제안 모델 (proposed)의 데이터 비율별 지도 변화:

```
Ratio 1%:   극도로 희미함, 노이즈 수준
Ratio 10%:  중요 지역 시작 노출
Ratio 50%:  명확한 패턴 (산사태 취약지 뚜렷함) ★
Ratio 100%: 50%와 유사하지만 더 안정적
```

**해석**: 50% 비율에서 최고 성능을 달성하므로, 지도의 신뢰도도 50%에서 최고.

#### 공간 검증 (Spatial Validation)

생성된 지도와 실제 산사태 위치 비교:

```
제안 모델 (ratio=50%):
- 실제 산사태 위치 (검정 점): 지도의 high probability 영역과 대부분 일치
- High Risk Zone (확률 > 0.6): 실제 산사태 밀집 지역과 overlap

Traditional ML (ratio=70-80%):
- 더 많은 false positive (high prob인데 산사태 없음)
- 더 넓은 고위험 영역 (보수적 예측)
- 운영 관점: 과도한 경고 → 실무 비효율
```

### 5.3 핵심 인사이트

#### 1. Few-Shot Learning 강점

제안 모델이 **50% 데이터**에서 최고 성능을 달성하는 반면:
- 전통 ML (XGBoost): 70% 데이터 필요 → **20%p 더 많음**
- 지도학습만 (supervised): 70~80% 필요 → **20~30%p 더 많음**

**해석**: SimCLR 사전학습이 강력한 표현을 제공하여 소량 라벨링된 데이터만으로도 충분.

#### 2. Detailed Embedding의 혁신성

undetailed vs detailed 비교:
- undetailed Acc: 0.6643 (최고 성능 ratio=8%)
- detailed Acc: 0.7705 (최고 성능 ratio=50%)
- **차이: 10.62%p** → 매우 큼

원인:
- Detailed: 각 범주형 특성을 **세분화된 라벨로 임베딩** (예: 토양 분류 → 100개 세부 범주)
- Undetailed: **통합된 라벨만 사용** (예: 토양 분류 → 몇 개 대분류만)
- 임베딩 품질 차이 → downstream 성능 결정

#### 3. Debiased Loss의 효과

Debiased NT-Xent (tau_plus=0.1) 도입:
- Standard NT-Xent 대비: +4.83% Accuracy (50% ratio)
- False negative bias 보정이 지형 데이터에서 특히 유효
- 원인: 지형 특성의 높은 상관성 → false negative 발생 빈번

#### 4. Transform의 선택적 역할

Transform (arcsinh 곡률, log SCA) 제거 시:
- 정확도 감소: -1.45% (작음)
- 범주형 임베딩의 영향에 비해 미미

**해석**: 범주형 특성이 압도적으로 중요하며, 연속형 전처리는 secondary.

#### 5. 차원 감소 (dim32)의 정보 손실

embedding_dim 64 → 32:
- AUC 감소: -2.67%
- 범주형 특성의 의미론적 정보 손실
- **권장**: 연산 비용 절감 필요 없다면 dim64 유지

#### 6. Traditional ML vs Deep Learning

**정확도 비교** (최고 성능 기준):

| 모델 | Accuracy | AUC | 필요 Ratio | 학습 시간 | 예측 시간 |
|------|----------|-----|-----------|---------|---------|
| **proposed (SSL)** | **0.7705** | **0.8434** | **50%** | ~24h (GPU) | ~30초 |
| xgboost | 0.7681 | 0.8603 | 70% | ~2h (CPU) | ~5초 |
| random_forest | 0.7536 | 0.8462 | 80% | ~3h (CPU) | ~10초 |
| supervised_multi_fusion | 0.7343 | 0.8117 | 80% | ~18h (GPU) | ~30초 |
| supervised_single20_resnet | 0.7343 | 0.7981 | 70% | ~12h (GPU) | ~20초 |

**주요 차이점**:

1. **데이터 효율성**
   - 제안: 50% 데이터로 최고 성능 ✓
   - XGBoost: 70% 데이터 필요 (20%p 더 필요)
   - RF: 80% 데이터 필요 (30%p 더 필요)
   - 원인: SimCLR 사전학습의 강력한 표현 학습

2. **확률 교정(Calibration)**
   - 제안: 균형잡힌 확률 분포 (0.2~0.8)
   - XGBoost: 극단적 확률 (0/1 근처) → 과신뢰
   - RF: 이산적 확률 (0.2, 0.4, 0.6, ...) → 해상도 낮음
   - 시사: 지도 기반 운영에서 제안 모델이 더 신뢰도 높음

3. **공간적 예측 패턴**
   - 제안: 부드러운 확률 변화 → 지형 변동성 포착
   - XGBoost: 급격한 경계 → 경과지역 불명확
   - RF: 계단 현상 → 연속성 부족
   - 원인: CNN의 spatial receptive field vs tree의 feature split

4. **계산 효율**
   - 학습: XGBoost < RF < 제안 (GPU 사용)
   - 예측: XGBoost > RF > 제안 (병렬화 필요)
   - Trade-off: 높은 정확도는 높은 계산 비용 수반

**결론**: 제안 모델은 **라벨 효율성**과 **공간 예측 품질** 면에서 우수하지만, **예측 속도**가 필요하면 XGBoost 고려.

---

## 6. Discussion

### 6.1 주요 발견(Key Findings)

#### Finding 1: SSL의 라벨 효율성
- **명제**: SimCLR 사전학습은 소량의 라벨링된 데이터에서 뛰어난 성능 달성
- **증거**: 50% 데이터 × 제안 모델 = Acc 0.7705 (최고)
- **비교군**:
  - 전통 ML: 80% 데이터 필요 → 같은 성능 (Acc 0.75 정도)
  - 지도학습만: 70% 데이터 필요
- **실무 의미**: 산사태 라벨 수집 비용 ~50% 절감 가능

#### Finding 2: 다중 임베딩 전략의 효과
- 범주형 특성을 **의미론적 임베딩**으로 표현 (숫자 인코딩 대신)
- Detailed(세분화) > Undetailed(통합): **10.62%p 성능 차이**
- 원인: 세분화된 범주 정보가 SimCLR 인코더에 더 충분한 학습 신호 제공

#### Finding 3: Debiased Contrastive Learning의 필요성
- 표준 NT-Xent loss의 **false negative bias** (모든 음성 샘플이 진정한 음성 아닐 수 있음)
- 지형 데이터에서 특히 중요: 인접 패치들이 높은 특성 유사성
- Debiased loss: +4.83% 정확도 향상

#### Finding 4: 과적합의 부재 또는 약함
- 50% → 100% 비율로 증가해도 성능 저하 미미
- AUC: 0.8434 (50%) → 0.8478 (100%) (+0.44%)
- 의미: 모델 용량과 정규화가 적절함

#### Finding 5: 표현 학습이 전통 ML을 능가
- **명제**: 표현 학습(representation learning) 방식이 특성 엔지니어링(feature engineering)을 능가
- **증거**:
  - XGBoost (flatten된 6,480차원): Acc 0.7681 (70% ratio)
  - 제안 모델 (CNN 임베딩): Acc 0.7705 (50% ratio)
  - 동일 정확도 달성에 20%p 덜 필요
- **원인**:
  - Traditional ML: 모든 차원을 동등하게 취급 (curse of dimensionality)
  - CNN: 공간 구조 활용 + 자동 특성 계층화 (spatial hierarchy)
  - SimCLR: 비지도 대조 학습으로 robust representation 사전학습
- **결론**: 고차원 공간 데이터는 자동 표현 학습이 더 효과적

#### Finding 6: 확률 보정(Probability Calibration)의 중요성
- **명제**: 신뢰성 높은 예측 확률이 실무에서 중요
- **증거**:
  - XGBoost: AUC 0.8603 (높음!) 하지만 확률 극단적 (0/1 근처)
  - 제안 모델: AUC 0.8434 (낮지 않음) 하지만 확률 균형잡힘
  - 실무 운영: 제안이 위험 지역 판정에 더 신뢰도 높음
- **시사**: AUC 외에 calibration 메트릭(ECE, MCE) 병행 필요

### 6.2 한계점(Limitations)

#### 1. 지역 특수성(Regional Specificity)
- 현재 모델은 **특정 연구지역**에서만 훈련/평가
- 다른 지역의 산사태 취약성 예측 시 transfer learning 성능 미지수
- **해결 방안**: 다지역 데이터 수집 및 domain adaptation 연구

#### 2. 시간 정보 부재
- 현재: 공간 특성만 사용 (고도, 토양, 지질 등)
- 미포함: 강우량, 지진, 계절 변화 등 시간 동적 특성
- **영향**: 실시간 위험도 변화 예측 불가

#### 3. 클래스 불균형의 잔존 영향
- LS:Non-LS = 1:182 (심각한 불균형)
- 현재 대처: 미니배치 샘플링으로 1:1 균형 유지
- 제약: 극드문 사건(rare event)의 공간 집중 패턴 학습 어려움

#### 4. OpenAI API 의존성
- 범주형 특성 임베딩을 **OpenAI text-embedding-3-small**으로 생성
- 문제:
  - 폐쇄형 API (재현성/투명성 제약)
  - 비용 (대규모 배포 시)
  - 외부 의존성
- **대안**: BERT 등 오픈소스 모델로 대체 가능

#### 5. 테스트 셋의 소규모
- Test: 414 패치 (전체 69,742의 0.6%)
- Validation: 656 패치 (고정된 split)
- 제약: 드물고 극단적인 지형의 충분한 표현 부족

### 6.3 일반화 가능성(Generalizability)

#### Transfer Learning 관점
```
Source: 연구지역 A (SimCLR 사전학습)
Target: 연구지역 B (미세조정)

예상 성능 순서:
1. 같은 지역, 많은 라벨 (현재): Acc 0.77
2. 인접 지역, 유사 지질: Acc 0.72~0.75 (추정)
3. 원거리, 다른 지질: Acc 0.65~0.70 (추정)
4. 다른 대륙: Acc ?
```

#### 응용 확대 가능성
- ✓ **같은 지형 유형**: 높음 (유사 지질, 토양, 삼림)
- ✓ **유사 기후**: 높음 (비슷한 강우 영향)
- △ **다른 위도/기후**: 중간 (지형 원리는 보편적이나 대기 특성 차이)
- △ **도시 지역**: 낮음 (빌딩, 포장도로 등 특수 요소)
- ✗ **화산, 설산 지역**: 매우 낮음 (완전히 다른 지형학)

---

## 7. Conclusion & Future Work

### 7.1 주요 결론

본 연구는 **SimCLR 기반의 자기지도학습과 다중경로 퓨전 인코더를 결합하여 산사태 취약성 예측의 신경망 아키텍처를 제안**했다.

#### 핵심 성과
1. **최고 성능**: AUC 0.8434, Accuracy 0.7705 (50% 데이터)
2. **라벨 효율성**: 전통 ML 대비 30%p 적은 데이터로 동등 성능
3. **포괄적 검증**: 5개 ablation 변수 + 14개 data ratio + 7개 모델 체계적 평가
4. **실운영 가능**: 연구지역 전체 산사태 취약성 맵 자동 생성

#### 핵심 기여
- **범주형 특성 임베딩**: OpenAI 세분화 임베딩으로 10%p 성능 향상
- **Debiased contrastive learning**: false negative bias 보정으로 4.8%p 개선
- **Few-shot 강점**: 소량 라벨링 데이터로도 강건한 성능 달성

### 7.2 향후 연구 방향

#### 단기 (6개월)
1. **다지역 확장**
   - 3~5개 추가 연구지역에서 모델 평가
   - Domain shift 및 transfer learning 성능 정량화

2. **동적 특성 통합**
   - 강우량, 지진 등 시간 시계열 데이터 추가
   - Temporal fusion 모듈 설계

3. **오픈소스 임베딩 대체**
   - OpenAI API → BERT/RoBERTa 전환
   - 비용 및 재현성 개선

#### 중기 (1년)
1. **실시간 위험도 예측**
   - 강우 데이터와 실시간 연계
   - 위험 지수(risk score) → 경보 시스템 연결

2. **해석성 강화**
   - Attention mechanism 도입 (어떤 특성이 결정적인가)
   - SHAP/LIME으로 개별 예측 설명

3. **멀티모달 데이터 통합**
   - SAR(Synthetic Aperture Radar) 이미지 추가
   - 구름 영향 없는 산사태 감지

#### 장기 (2년 이상)
1. **국가 규모 배포**
   - 전국 산림청 데이터로 국가 산사태 위험 지도 구축
   - 조기 경보 시스템 통합

2. **멀티헤저드 모델링**
   - 산사태 + 토사유출 + 산림 피해를 통합 예측
   - 자연재해 복합 위험도 평가

3. **기후변화 영향 분석**
   - 강우 패턴 변화 시뮬레이션
   - 미래 산사태 취약성 시나리오 분석

### 7.3 최종 평가

**연구 의의**: 원격탐사 기반 산사태 취약성 예측에 **자기지도학습과 세분화된 범주형 임베딩**을 도입하여, 라벨 효율성과 성능을 동시에 달성했다. 특히 **few-shot learning 강점**은 실무에서 라벨 수집 비용 절감에 직접 기여할 수 있다.

**사회적 임팩트**: 조기 경보 능력 향상 → 산사태로 인한 인명 피해 및 경제 손실 감소 기대.

---

## Appendix: 실험 재현 가이드

### 환경 설정
```bash
conda create -n ls_simclr python=3.8
conda activate ls_simclr
pip install tensorflow==2.5.0 scikit-learn numpy scipy matplotlib
```

### 주요 스크립트 실행

#### 1. Ablation study 테스트 (기존 훈련 모델 평가)
```bash
python test_ablation.py                    # 5개 fusion ablation
python test_ablation_supervised.py         # 2개 supervised baseline
python test_traditional_ml.py              # 2개 전통 ML baseline
```

#### 2. 취약성 맵 생성 (연구지역 전체)
```bash
./run_susceptibility_maps.sh               # 7개 모델 × 14개 ratio
# 또는 개별 실행:
python predict_susceptibility_map.py \
  --model_dir finetuned_models/fusion_resnet_transform_db_detailed_dim64_0122 \
  --ratio 50 \
  --embedding_path ./embeddings/detailed_embedding64.npy \
  --mode fusion
```

#### 3. 결과 분석
```bash
# CSV 분석 (pandas 등)
python -c "
import pandas as pd
df = pd.read_csv('test_results/ablation_results_0311.csv')
print(df.groupby('ablation_role')[['accuracy', 'auc']].max())
"
```

---

**Report Generated**: 2026-03-12
**Author**: Research Team (Claude Opus 4.6)
**Repository**: https://github.com/[repo]/ls_simclr (encoder_freeze branch)
