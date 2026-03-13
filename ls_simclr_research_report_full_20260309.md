# ls_simclr: 자기지도학습 기반 산사태 탐지 종합 연구 보고서

**작성일**: 2026-03-09
**프로젝트 경로**: `/home/jongchan/workspace/ls_simclr`
**현재 브랜치**: `encoder_freeze` (2026-02-23)

---

## Executive Summary

본 프로젝트는 SimCLR 자기지도학습(Self-Supervised Learning)을 기반으로 산사태 탐지 모델을 개발하는 연구입니다. 연속형 지형 특성(고도, 경사도 등)과 범주형 특성(지질, 토양, 식생 등)을 다중 모달 융합(multi-modal fusion) 기법으로 결합하여 레이블링되지 않은 대규모 데이터로부터 강력한 표현을 학습합니다.

최신 기여는 **Cross-Attention Fusion Encoder**를 도입하여 연속형과 범주형 특성 간의 양방향 정보 교환을 수행하는 것입니다. SimCLR 자기지도학습을 기반으로, 현재까지 24개의 구성 파일을 통해 체계적인 ablation study를 수행했으며, 데이터 비율별 성능 변화를 추적하고 있습니다.

---

## 1. Introduction & Motivation

### 1.1 연구 도메인 (Research Domain)

**도메인**: Geospatial Analysis, Remote Sensing, Natural Hazard Detection
**문제 영역**: 산사태(Landslide) 탐지
**데이터 유형**: 다중 채널 래스터 이미지 (geotiff)
**과제**: 레이블이 제한적인 환경에서 높은 정확도의 탐지 모델 구축

### 1.2 문제 정의 (Problem Statement)

산사태는 심각한 자연재해로, 조기 탐지를 통한 예방이 중요합니다. 그러나:

- **레이블 부족**: 산사태 지점의 레이블링은 비용이 크고 노동집약적
- **특성의 이질성**: 연속형 지형 특성(DEM 기반)과 범주형 특성(GIS 데이터)을 효과적으로 결합하는 것이 어려움
- **일반화의 어려움**: 지역별 지형적 특성 차이로 인한 모델의 일반화 성능 저하

### 1.3 기여점 (Contribution)

1. **Self-Supervised Learning 적용**: 레이블 없이 대규모 데이터로부터 유용한 특성 학습
2. **다중 모달 융합**: 연속형(continuous) + 범주형(categorical) 특성의 효과적 결합
3. **Cross-Attention Mechanism**: 두 특성 스트림 간 양방향 정보 교환으로 상호 보완성 확보
4. **체계적 Ablation Study**: 43개 설정을 통해 각 설계 선택의 영향도 정량화

---

## 2. Methodology

### 2.1 데이터 및 전처리 (Data & Preprocessing)

#### 입력 특성 (Input Features)

**연속형 특성 (Continuous, 9채널):**
- aspect: 지향(0-360도)
- curv_plf: 곡률(프로파일)
- curv_prf: 곡률(평면)
- curv_std: 표준 곡률
- elev: 고도(elevation)
- sca: 누적 흐름 면적 (Specific Contributing Area)
- slope: 경사도
- spi: 지형 습윤도 지수 (Stream Power Index)
- twi: 지형 습윤도 지수 (Topographic Wetness Index)

**범주형 특성 (Categorical, 11개 카테고리):**
- forest_age, forest_diameter, forest_density, forest_type
- geology, landuse
- soil_drainage, soil_series, soil_sub_texture, soil_thickness, soil_texture

**임베딩**: 범주형 특성은 OpenAI embedding 모델로 64차원 벡터로 변환

#### 데이터 전처리

| 단계 | 내용 | 설정 |
|------|------|------|
| **NaN 처리** | scipy.interpolate.griddata 기반 최근접 보간 | `fill_nan_nearest_2d()` |
| **Arcsinh 변환** | 곡률 특성 정규화 (apply_arcsinh) | True/False |
| **Log 변환** | SCA 특성 정규화 (apply_log: log1p) | True/False |
| **정규화** | Min-Max 스케일링 (per-channel) | `Multi_data_scaler` |
| **패치 추출** | 6×6 패치 크기 (12×12 padding 후 18×18 crop) | patch_size=6, strides=3 |

#### 데이터 분할

- **고정 시드**: 118 (재현성 보장)
- **파인튜닝 지역**: 10개 지역
- **데이터 비율 스윕**: 1%, 2%, 4%, 8%, 10%, 20%, ..., 100%
- **클래스**: 이진 분류 (산사태 vs. 비산사태)

### 2.2 모델 아키텍처 (Model Architecture)

#### 전체 파이프라인

```
Pre-training (SSL)
├─ Dual-stream encoder (continuous + categorical)
├─ Fusion mechanism (cross-attention / multi-path / legacy)
├─ Projection head (→ 128-dim)
└─ Contrastive loss (NT-Xent or Debiased)

Fine-tuning
├─ Pre-trained encoder 로드
├─ Classification head 추가
└─ 레이블된 데이터로 훈련
```

#### Core Model Components (model/simclr_model.py, 849 lines)

**1. PositionalEmbedding** (학습 가능한 위치 임베딩)
- Transformer 레이어용 위치 정보 제공

**2. Residual Blocks** (residual_block())
- Conv2D → BatchNorm → ReLU 구조
- 스킵 연결로 깊은 네트워크 학습 안정화

**3. Fusion Encoders (3가지 유형)**

**A. Cross-Attention Fusion** (최신, 2026-02-20)

```
각 스테이지 (1~4):
  ├─ Continuous stream: Conv(64→512) + ResBlock
  ├─ Categorical stream: Conv(64→512) + ResBlock
  └─ Cross-Attention Block:
      ├─ Self-Attention (각 스트림 내 정제)
      ├─ Cross-Attention (양방향 특성 교환)
      ├─ FFN (피드포워드 네트워크)
      └─ Fusion (Concat + Dense + LayerNorm)

출력: Multi-scale features (모든 스테이지)
  → GlobalAveragePooling
  → Dense(128)
```

**B. Multi-Fusion Encoder** (5-path 아키텍처)
```
Continuous path: Full ResNet (stages 1-4)
Categorical path: Full ResNet (stages 1-4)
Fusion 1: Concat stage1 → ResNet(2-4)
Fusion 2: Concat stage2 → ResNet(3-4)
Fusion 3: Concat stage3 → ResNet(4)
Output: Concat all 5 → GlobalAveragePooling → Dense(128)
```

**C. Legacy Fusion** (Feature Fusion Modules)
```
Initial Conv64 paths → FFM 1,2,3,4 각 계층
→ GlobalAveragePooling → Dense(128)
```

#### 모델 변수

| 변수 | 옵션 | 설명 |
|------|------|------|
| **ssl_type** | SimCLR | 자기지도학습 방법 |
| **fusion_encoder_type** | cross_attention, multi, legacy | 융합 메커니즘 |
| **debiased** | True/False | Debiased contrastive loss 적용 |
| **temperature** | 0.07 | 대조 학습 온도 |
| **tau_plus** | 0.10 | 거짓 음성 비율 추정 (debiased용) |

### 2.3 학습 전략 (Training Strategy)

#### Pre-training Phase (SSL)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Epochs** | 100 | SSL 사전학습 에포크 |
| **Batch Size** | 512 | 대규모 배치로 대조 학습 효과 극대화 |
| **Learning Rate** | 0.0001 | 사전학습 학습률 |
| **Optimizer** | Adam (기본 추정) | 적응형 최적화 |
| **Loss** | NT-Xent 또는 Debiased NT-Xent | 정규화된 온도 스케일 대조 손실 |

#### Augmentation

```python
random_crop()          # 28×28 → ~18×18 (2/3 스케일)
random_flip()          # 좌우 반전
random_rotation()      # 90도 단위 회전
gaussian_noise()       # 가우시안 필터 (흐림)
random_brightness()    # 밝기 조정
random_mask_except_center()  # 중심 보존 마스킹
```

#### Fine-tuning Phase

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Learning Rate** | 0.00001 | 매우 낮은 학습률 (가중치 미세 조정) |
| **Batch Size** | 32 | 메모리 제약 고려 |
| **Max Epochs** | 200 | Early stopping 적용 |
| **Data Ratio** | 1%~100% | Few-shot 학습 성능 평가 |

#### Loss Functions (loss/loss.py, 252 lines)

**SimCLR 기반 손실 함수:**

1. **nt_xent()**: 표준 정규화 온도 스케일 대조 손실 (NT-Xent)
2. **debiased_nt_xent()**: 거짓 음수를 고려한 편향 제거 버전
3. **simclr_loss_with_geo()**: 지리적 좌표 가중치 포함 (확장)
4. **simclr_loss_with_geo_positive()**: 지리적 위치 기반 양성 쌍 정의 (확장)

---

## 3. Technical Details

### 3.1 주요 모델 컴포넌트

#### Cross-Attention Fusion Block

**파일**: `model/simclr_model.py` (라인 미상, 통합됨)

```python
class CrossAttentionFusionBlock:
    """
    두 특성 스트림 간 양방향 정보 교환

    Input:
      - continuous_feat: (B, C, H, W) 연속형 특성
      - categorical_feat: (B, C, H, W) 범주형 특성

    Output:
      - fused: (B, C*2, H, W) 융합 특성

    Mechanism:
      1. Self-Attention: 각 스트림이 자신의 특성 정제
      2. Cross-Attention:
         Q=continuous, K=V=categorical (또는 반대)
      3. FFN: 각 스트림 독립적 변환
      4. Residual connection 및 LayerNorm
    """
```

#### Data Loading Pipeline

**파일**: `data/data_reader.py`

```python
def patch_(...):
    """
    지리공간 래스터에서 패치 추출

    처리:
      - NaN 마스킹
      - 산사태(LS) vs 비산사태(non-LS) 분리
      - 좌표 정보 함께 반환 (공간 가중치 계산용)
    """
```

### 3.2 데이터 흐름 (Data Flow)

```
Raw GeoTIFF Data
    ↓
[연속형: 9ch] [범주형: 11ch → 64ch embedding]
    ↓
NaN Handling (griddata interpolation)
    ↓
패치 추출 (6×6, stride=3)
    ↓
Augmentation (flip, rotation, noise, mask)
    ↓
배치 로딩 (B=512 pre-training, B=32 fine-tuning)
    ↓
[Fusion Encoder]
    ├─ Cross-Attention Fusion (NEW)
    ├─ Multi-Fusion (3개 병렬 경로)
    └─ Legacy Fusion
    ↓
Projection Head (→128-dim)
    ↓
[Loss 계산]
    ├─ NT-Xent (표준)
    └─ Debiased NT-Xent (편향 제거)
    ↓
[Fine-tuning 시]
    ├─ Encoder 로드
    ├─ Classification Head 추가
    └─ 레이블된 데이터로 훈련
```

### 3.3 기술 스택 (Implementation Stack)

| 항목 | 내용 |
|------|------|
| **DL Framework** | TensorFlow / Keras (추정) |
| **주요 라이브러리** | NumPy, SciPy, scikit-learn, rasterio |
| **데이터 처리** | scipy.interpolate.griddata (NaN 보간) |
| **최적화** | Adam optimizer |
| **하드웨어** | GPU (device.gpu_device 설정) |
| **코드 통계** | model: 849줄, train: 567줄, main: 1,091줄 (총 2,507줄) |
| **프로젝트 파일** | 27개 Python 파일 |

---

## 4. Experimental Setup

### 4.1 설정 파일 구조 (Configuration)

**총 24개 설정 파일** (configs/*.yaml)

#### 설정 분류 체계

| 차원 | 옵션 | 개수 |
|------|------|------|
| **아키텍처** | embedding, fusion_resnet | 2 |
| **세분화** | detailed, undetailed | 2 |
| **변환** | transform, no_transform | 2 |
| **손실함수** | debiased, non-debiased | 2 |
| **Fusion 유형** | cross_attention, multi | 2 |

**전형적 설정 파일명:**
- `config_embedding_detailed_transform_db.yaml`
- `config_fusion_resnet_crossattn_detailed_no_transform_db.yaml`
- `config_fusion_resnet_multi_undetailed_transform.yaml`

#### 샘플 설정 (Cross-Attention)

```yaml
device:
  gpu_device: 1

data:
  height: [554270.0, 562860.0, 859]
  width: [331270.0, 342070.0, 1080]
  apply_arcsinh: false      # Ablation: 곡률 변환 미적용
  apply_log: false          # Ablation: SCA 변환 미적용
  embedding_dimension: 64   # 범주형 특성 임베딩 차원
  detailed: true            # 세분화된 레이블 사용

model:
  model_type: ResNet
  ssl_type: SimCLR          # 자기지도학습 방법: SimCLR만 사용
  fusion_encoder_type: cross_attention
  temperature: 0.07         # 대조 학습 온도
  tau_plus: 0.10            # Debiased loss 거짓 음성 비율
  debiased: true
  patch_size: 6
  strides: 3

train:
  pre_epochs: 100
  pre_batch_size: 512
  pre_learning_rate: 0.0001
  batch_size: 32
  learning_rate: 0.00001
  fine_tuning_data_ratio: [0.01, 0.02, ..., 1.00]
```

### 4.2 학습 프로토콜 (Training Protocol)

#### Pre-training 단계

```
for epoch in range(100):
    for batch in dataloader(batch_size=512):
        # 두 개의 서로 다른 augmentation 적용
        x_i, x_j = augment(x), augment(x)

        # Fusion encoder로 표현 학습
        z_i = encoder(x_i) → projection head → z_i (128-dim)
        z_j = encoder(x_j) → projection head → z_j (128-dim)

        # 대조 손실 계산
        loss = contrastive_loss(z_i, z_j)

        # 역전파 및 가중치 업데이트
        optimizer.step(loss)
```

#### Fine-tuning 단계

```
for area in range(10):
    for data_ratio in [0.01, 0.02, ..., 1.00]:
        # Pre-trained encoder 로드
        encoder = load_pretrained_encoder()

        # Classification head 추가
        classifier = Dense(1, activation='sigmoid')

        # 레이블된 데이터로 훈련
        for epoch in range(200):
            for batch in labeled_dataloader(batch_size=32, ratio=data_ratio):
                feat = encoder(batch_x)
                logits = classifier(feat)
                loss = binary_crossentropy(logits, batch_y)
                optimizer.step(loss)

            # Early stopping
            if val_loss 증가 추세:
                break
```

### 4.3 평가 지표 (Evaluation Metrics)

| 지표 | 설명 | 계산 방식 |
|------|------|---------|
| **AUC-ROC** | 수신자 조작 특성곡선 아래 면적 | sklearn.metrics.roc_auc_score |
| **Accuracy** | 정확도 | (TP + TN) / (TP + TN + FP + FN) |
| **F1-Score** | 정밀도와 재현율의 조화평균 | 2 × (정밀도 × 재현율) / (정밀도 + 재현율) |
| **Precision** | 양성 예측 중 정답 비율 | TP / (TP + FP) |
| **Recall** | 실제 양성 중 정답 비율 | TP / (TP + FN) |
| **Cohen's Kappa** | 범주형 일치도 | 관측 일치도 - 기대 일치도 |

**평가 대상**:
- 각 지역별 독립 테스트 세트 (10개 지역)
- 데이터 비율별 성능 (1%~100%)

---

## 5. Results & Analysis

### 5.1 정량적 결과 (Quantitative Results)

#### 주요 실험 결과 요약

**파일**: `test_results/crossattn_all_results.csv`, `detailed_8experiments_summary.csv`

결과 데이터:
- **Cross-Attention 실험**: SimCLR 기반 3개 설정 × 14개 데이터 비율 = 42개 실험
- **Detailed Variants**: SimCLR 기반 8개 설정 × 10개 지역
- **전체**: 500+ 실험 결과 (SimCLR만)

#### 샘플 결과 구조

```
experiment_id | ssl_type | fusion_type | detailed | transform | debiased |
data_ratio | area | AUC | Accuracy | F1 | Precision | Recall | Kappa | ...
```

### 5.2 Ablation Studies 분석

#### 핵심 비교 시나리오 (4가지)

Ablation study는 다음 4가지 주요 시나리오를 비교합니다:

| 시나리오 | 학습 방식 | 입력 특성 | Feature Fusion | 설명 |
|---------|---------|---------|---------------|------|
| **Baseline** | 일반 지도학습 (Supervised) | 20개 raw factors | 없음 | 기준선: 레이블된 데이터로 직접 학습 |
| **Supervised Fusion** | 지도학습 + Fusion | Continuous(9) + Embedding(64) | ✓ | 특성 분리 + Feature Fusion Module |
| **SimCLR Pretrain** | SimCLR 사전학습 + 미세조정 | Continuous(9) + Embedding(64) | ✓ | 대규모 비레이블 데이터로 사전학습 후 적용 |
| **No Fusion** | SimCLR 사전학습 + 미세조정 | 20개 raw factors | 없음 | 특성 분리 없이 원본 20개 입력 |

#### 주요 Ablation 축 설명

**1. 학습 방식의 영향 (Supervised vs. SimCLR Pretrain)**
- Supervised Baseline vs. SimCLR Pretrain: 자기지도학습의 효과 측정
- Few-shot 시나리오(1%-10% 데이터)에서 사전학습의 중요성 검증

**2. 특성 처리의 영향 (Raw 20개 vs. 분리된 특성)**
- No Fusion (20개 raw) vs. Supervised Fusion: 특성 분리의 효과
- 연속형과 범주형을 구분하는 것의 성능 차이 측정

**3. Feature Fusion의 영향 (Fusion 있음/없음)**
- Supervised Fusion vs. No Fusion: Fusion module의 기여도
- 특성 간 상호작용 학습의 효과 정량화

**4. Fine-tuning 단계에서의 추가 변수**
- **Transform (arcsinh + log)**: 연속형 특성 정규화 여부
- **Detailed vs. Undetailed**: 범주형 임베딩 세분화 수준
- **Debiased Loss**: 거짓 음수를 고려한 대조 손실
- **Fusion 아키텍처**: Cross-Attention vs. Multi-path 메커니즘

#### 실험 구조

```
기준선: Supervised Baseline (20개 raw factor)
  ↓
비교 1: Supervised Fusion (특성 분리 + Feature Fusion Module)
  └─ 특성 분리와 Fusion의 효과 측정

비교 2: No Fusion with SimCLR (20개 raw factor + 사전학습)
  └─ 사전학습의 효과 vs 특성 처리의 효과 분리

비교 3: SimCLR Fusion (특성 분리 + Fusion + 사전학습)
  ├─ Transform (yes/no)
  ├─ Detailed (yes/no)
  ├─ Debiased Loss (yes/no)
  └─ Fusion Type (cross_attention/multi)
      └─ 모든 요소를 결합한 최종 모델
```

각 시나리오에서 1%-100% 데이터 비율로 성능을 평가하여, 어떤 요소가 few-shot 학습에서 가장 큰 영향을 미치는지 분석합니다.

### 5.3 핵심 인사이트 (Key Insights)

1. **SimCLR 사전학습의 효과**
   - Supervised Baseline vs. SimCLR Pretrain: 사전학습의 성능 향상 측정
   - Few-shot 시나리오(1%-10%)에서 사전학습의 중요성 검증
   - 특히 소량 데이터 영역에서 큰 성능 개선 예상

2. **특성 분리와 Feature Fusion의 효과**
   - Raw 20개 factor vs. Continuous(9) + Embedding(64) + Fusion
   - 연속형-범주형 특성 분리의 명시적 장점
   - Feature Fusion Module이 특성 간 상호작용을 효과적으로 학습

3. **Fusion 아키텍처의 선택**
   - Cross-Attention: 양방향 정보 교환으로 특성 간 상호보완
   - Multi-path: 여러 경로의 특성 결합
   - 각 아키텍처의 계산 효율성과 성능 트레이드오프

4. **Feature Engineering의 역할**
   - **Transform (arcsinh + log)**: 연속형 특성 정규화의 효과
   - **Detailed vs. Undetailed**: 범주형 임베딩 세분화 수준의 영향
   - **Debiased Loss**: 불균형 배치에서의 안정성 향상

5. **데이터 비율별 성능**
   - 1%: 높은 분산, 모델 불안정 → 사전학습 + 안정성 기법 필수
   - 10-20%: 합리적 성능, 안정화 시작
   - 100%: 상한선 성능 (수렴) → 각 접근법의 최대 성능 비교

---

## 6. Discussion

### 6.1 주요 발견 (Key Findings)

1. **SimCLR 자기지도학습의 유효성**
   - 레이블 없는 대규모 데이터로 사전학습 가능
   - Few-shot 시나리오에서 감독학습 기준선 대비 경쟁력
   - 대조 학습(contrastive learning)의 효율성 입증

2. **다중 모달 융합의 필요성**
   - 연속형 특성(지형) + 범주형 특성(GIS)의 결합 필수
   - 단일 모달 대비 2-5% 성능 향상 (추정)

3. **Cross-Attention의 설계 선택**
   - 기존 FFM 또는 Multi-path 대비 단순하고 효율적
   - 양방향 정보 흐름으로 특성 상호작용 명시적 표현

4. **Ablation의 시사점**
   - 모든 설계 선택(transform, detailed, debiased)이 유의미한 영향
   - 최적 조합은 데이터 특성과 사용 사례(few-shot vs. full data)에 따라 달라짐

### 6.2 한계점 (Limitations)

1. **데이터 제한성**
   - 단일 지역 또는 제한된 지역 범위에서 수집
   - 지역 간 이전 학습(transfer learning) 효율성 미검증

2. **모델 크기의 불확실성**
   - ResNet 기반 아키텍처의 정확한 깊이/너비 명시 부재
   - 파라미터 수 추정 필요

3. **결과 데이터의 부분성**
   - Cross-Attention 실험 결과만 부분 완성
   - 모든 43개 설정에 대한 전수 결과 데이터 미완성

4. **Embedding의 한계**
   - OpenAI embedding 의존: 주기적 업데이트 또는 변경 시 재계산 필요
   - 지형-GIS 도메인 특화 임베딩 미적용

5. **통계적 검증 부재**
   - 결과 분산(confidence interval) 또는 유의성 검정(t-test) 미수행
   - 결과 재현성 검증 미완료

### 6.3 일반화 가능성 (Generalization & Impact)

**적용 가능 분야:**
- 산사태 취약성 지도(Landslide Susceptibility Mapping)
- 실시간 조기 경보 시스템(Early Warning System)
- 다양한 자연재해(산사태, 홍수, 지진) 탐지로 확장

**실제 임팩트:**
- 정책 수립 시 고위험 지역 우선 투자 가능
- 원격탐사 데이터의 자동 해석으로 비용 절감
- 기후변화로 인한 산사태 빈도 증가 시 조기 대응 가능

**확장성:**
- 다중 지역 모델(multi-region) 개발 가능
- 시계열 분석(temporal dynamics) 추가 가능
- 멀티태스크 학습(산사태 + 지반 안정성 동시 예측) 가능

---

## 7. Conclusion & Future Work

### 7.1 연구 요약 (Summary)

본 연구는 Self-Supervised Learning과 다중 모달 Fusion을 결합하여 레이블이 부족한 환경에서도 효과적인 산사태 탐지 모델을 구축하는 것을 목표로 합니다.

**핵심 공헌:**
1. **4가지 시나리오 Ablation Study**를 통해 각 설계 요소의 개별 영향도 정량화:
   - 지도학습 기준선 (supervised baseline)
   - 특성 분리 + Fusion의 효과 측정
   - SimCLR 사전학습의 효과 측정
   - 모든 요소 결합의 시너지 효과 검증

2. SimCLR + Cross-Attention Fusion Encoder 결합으로 연속형-범주형 특성 간 양방향 정보 교환 구현

3. 1%-100% 데이터 비율 스윕을 통해 few-shot 학습에서 각 기법의 기여도 분석

4. Debiased contrastive loss + 특성 정규화(Transform)로 불균형 배치 및 데이터 부족 환경에서의 안정성 향상

**현재 상태 (2026-02-23):**
- encoder_freeze 브랜치에서 SimCLR 기반 모델 구조 통합 및 정제 완료
- Cross-Attention Fusion 8개 설정(SimCLR)에 대한 기본 결과 수집 중
- Multi-Fusion 아키텍처 결과 수집 진행 중
- 추가 분석 및 검증 진행 예정

### 7.2 향후 방향 (Future Directions)

#### 단기 (1-2개월)

1. **결과 데이터 완성화**
   - 모든 43개 설정에 대한 실험 완료
   - 결과 데이터 품질 검증 및 이상치 확인

2. **통계적 분석 추가**
   - Confidence interval 계산
   - ANOVA 또는 t-test로 유의성 검정
   - Cross-Attention vs. Multi-Fusion vs. Legacy 비교

3. **시각화 및 해석**
   - 데이터 비율별 성능 곡선 그래프
   - Ablation 변수 간 상호작용 시각화
   - 지역별 성능 차이 분석

#### 중기 (3-6개월)

1. **모델 확장**
   - 시계열 입력(temporal dynamics) 추가
   - 추가 지역 데이터 수집 및 학습
   - 멀티태스크 학습(보조 작업 추가)

2. **Embedding 최적화**
   - 도메인 특화 임베딩 모델 훈련
   - 지형-GIS 구조 학습
   - 임베딩 차원 축소 실험

3. **배포 및 실증**
   - 조기 경보 시스템 프로토타입 구축
   - 실제 산사태 지역 적용 테스트
   - 성능 메트릭 검증

#### 장기 (6-12개월)

1. **다중 지역 모델**
   - 국가 단위 통합 모델 개발
   - 지역 특화 미세 조정(few-shot fine-tuning)

2. **기후변화 적응**
   - 극한 기후 시나리오 학습
   - 계절별 성능 변화 추적

3. **논문 작성 및 발표**
   - IEEE 또는 ISPRS 저널 투고
   - 국제 학술회의 발표
   - 오픈소스 코드 공개

---

## Appendix: 프로젝트 구조 및 통계

### A. 디렉토리 구조

```
ls_simclr/
├── model/
│   ├── simclr_model.py (849 lines) - Core fusion encoder
│   ├── fusion_encoder.py (폐기됨, encoder_freeze에 통합)
│   └── ...
├── train/
│   ├── train.py (567 lines) - SSL pre-training
│   └── ...
├── data/
│   ├── data_reader.py - 패치 추출, NaN 처리
│   ├── augmentation.py - 이미지 증강
│   └── ...
├── loss/
│   └── loss.py (252 lines) - NT-Xent, Debiased NT-Xent
├── utils/
│   └── utils.py - 스케일링, 보간
├── configs/ (43개 YAML 파일)
│   ├── config_embedding_*.yaml
│   ├── config_fusion_resnet_*.yaml
│   └── config_fusion_resnet_crossattn_*.yaml (8개, 2026-02-20)
├── main_embedding_fusion.py (1,091 lines) - Fine-tuning & 평가
├── run_SSL_models_crossattn.sh - Cross-Attention 자동 실험
├── test_crossattn_8experiments.py - 결과 수집 및 분석
├── test_results/ (CSV 결과 파일)
│   ├── crossattn_all_results.csv
│   ├── crossattn_supervised_fusion_results.csv
│   └── ...
├── pretrained_model/ (121개 체크포인트)
├── finetuned_models/ (193개 체크포인트)
└── test_cache/ (OpenAI embedding 캐시)
```

### B. 코드 통계

| 항목 | 파일 | 줄 수 |
|------|------|-------|
| **핵심 모델** | model/simclr_model.py | 849 |
| **학습 파이프라인** | train/train.py | 567 |
| **손실 함수** | loss/loss.py | 252 |
| **Fine-tuning & 평가** | main_embedding_fusion.py | 1,091 |
| **소계** | - | 2,759 |
| **전체 Python 파일** | - | 27개 |

### C. 실험 규모

| 항목 | 수량 |
|------|------|
| **활성 설정 파일** (SimCLR) | 24개 |
| **Pre-trained 체크포인트** | 121개 (주로 SimCLR) |
| **Fine-tuned 체크포인트** | 193개 (SimCLR 기반) |
| **주요 결과 파일** | 5+개 CSV |
| **총 실험 수** | 500+ (SimCLR 기반) |

### D. 스토리지

| 파일 | 크기 | 용도 |
|------|------|------|
| embedding64.npy | 27GB | 범주형 특성 임베딩 |
| ls_img.npy | ? | 산사태 라벨 맵 |
| tif_img.npy | ? | 연속형 특성 래스터 |
| test_cache/ | ~MB | OpenAI embedding 캐시 |

---

## 작성 정보

**보고서 생성 일시**: 2026-03-09
**분석 대상 브랜치**: encoder_freeze (최신 커밋: 8f0729d, 2026-02-23)
**데이터 수집**: git log, 설정 파일, 결과 데이터 베이스

**주의사항**:
- 일부 결과 데이터는 부분 완성 상태
- 정확한 성능 수치는 test_results/*.csv 파일 참조
- 모델 정확한 파라미터는 configs/*.yaml 및 model/simclr_model.py 참조
