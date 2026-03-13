# LS-SimCLR: Self-Supervised Learning for Landslide Susceptibility Mapping

SimCLR 기반 자기지도학습(SSL)을 활용한 산사태 취약성 매핑 프레임워크.
연속형 지형 특성과 OpenAI 텍스트 임베딩 기반 범주형 특성을 융합(Fusion)하여 산사태 발생 가능성을 예측합니다.

## 주요 특징

- **SimCLR 사전학습** + 지도학습 미세조정 파이프라인
- **다중 융합 인코더**: Multi-path, Cross-Attention, Legacy 아키텍처 지원
- **OpenAI 텍스트 임베딩**: 범주형 지형 특성(지질, 토양, 산림 등)을 의미적 임베딩으로 변환
- **Debiased Contrastive Loss**: 지리적 편향을 보정하는 대조 학습
- **Ablation Study**: 변환, 임베딩 차원, 디바이어싱 등 체계적 성능 분석
- **Traditional ML Baseline**: Random Forest, XGBoost와의 비교 실험

## 입력 데이터

20개 지형 특성 (12×12 패치 → 28×28 리사이즈 → 18×18 크롭)

| 유형 | 특성 |
|------|------|
| **연속형** (9개) | aspect, curv_plf, curv_prf, curv_std, elev, sca, slope, spi, twi |
| **범주형** (11개) | geology, landuse, soil_drainage, soil_series, soil_texture, soil_thickness, soil_sub_texture, forest_age, forest_diameter, forest_density, forest_type |

## 프로젝트 구조

```
ls_simclr/
├── main.py                          # 통합 학습 스크립트 (plain/embedding/fusion 모드)
├── predict_susceptibility_map.py    # 산사태 취약성 맵 생성
├── model/
│   └── simclr_model.py              # 모델 아키텍처 (ResNet, ViT, Fusion 인코더)
├── data/
│   ├── data_reader.py               # 패치 데이터 로딩
│   ├── augmentation.py              # SimCLR 데이터 증강
│   └── transform.py                 # 특성 변환 (arcsinh, log)
├── loss/
│   └── loss.py                      # 대조 손실 함수 (debiased 포함)
├── train/
│   ├── train.py                     # 사전학습 로직
│   └── train_supervised_ablation.py # Ablation 지도학습
├── test/
│   ├── test_ablation.py             # Ablation study 평가
│   ├── test_ablation_supervised.py  # 지도학습 baseline 평가
│   ├── test_crossattn_8experiments.py # Cross-attention 8실험 평가
│   └── test_traditional_ml.py       # RF/XGBoost baseline 평가
├── run/
│   ├── run_SSL_models_crossattn.sh  # Cross-attention 실험 일괄 실행
│   ├── run_ablation.sh              # Ablation 학습 실행
│   └── run_susceptibility_maps.sh   # 취약성 맵 일괄 생성
├── embeddings/
│   ├── openai_embedding.py          # OpenAI 텍스트 임베딩 생성
│   └── __init__.py
├── description/
│   ├── detailed_labels.json         # 범주형 특성 상세 라벨
│   └── undetailed_labels.json       # 범주형 특성 간략 라벨
├── configs/                         # YAML 실험 설정 파일
├── utils/
│   └── utils.py                     # 유틸리티 (스케일러, NaN 처리)
├── report/                          # 연구 보고서
└── requirements.txt
```

## 설치

```bash
conda create -n ls_simclr python=3.8
conda activate ls_simclr
pip install -r requirements.txt
```

`.env` 파일에 OpenAI API 키 설정 (임베딩 생성용):
```
OPENAI_API_KEY=your_api_key_here
```

## 사용법

### 학습

`main.py`는 config YAML 파일에서 모드를 자동 감지합니다:

```bash
# Plain 모드: 20채널 raw features, 단일 인코더
python main.py --config configs/config_aug1.yaml

# Embedding 모드: 연속형 + 임베딩 결합, 단일 인코더
python main.py --config configs/config_embedding_transform.yaml

# Fusion 모드: 연속형/범주형 분리 입력, 이중 인코더
python main.py --config configs/config_fusion_resnet_crossattn_detailed_transform.yaml

# 모드 수동 지정 및 파라미터 오버라이드
python main.py --config configs/default.yaml --mode fusion --gpu_device 0 \
    --fine_tuning_data_ratio 0.5 --dir_name my_experiment
```

### 모드 자동 감지 규칙

| Config 특징 | 감지 모드 |
|---|---|
| `fusion_encoder_type` 존재 | `fusion` |
| `embedding_dimension` 존재 (fusion 아님) | `embedding` |
| 그 외 | `plain` |

### 일괄 실험

```bash
# Cross-attention 8실험 일괄 실행
bash run/run_SSL_models_crossattn.sh

# Ablation 지도학습 실험
bash run/run_ablation.sh

# 취약성 맵 일괄 생성
bash run/run_susceptibility_maps.sh
```

### 평가

```bash
# Ablation study 테스트
python test/test_ablation.py

# 지도학습 baseline 테스트
python test/test_ablation_supervised.py

# Cross-attention 8실험 테스트
python test/test_crossattn_8experiments.py

# Traditional ML baseline
python test/test_traditional_ml.py
```

### 취약성 맵 생성

```bash
# Fusion 모델 (proposed)
python predict_susceptibility_map.py --mode fusion \
    --config configs/config_fusion_resnet_crossattn_detailed_transform_db.yaml \
    --model_dir finetuned_models/fusion_resnet_crossattn_detailed_transform_db_0228

# Random Forest baseline
python predict_susceptibility_map.py --mode traditional_rf

# XGBoost baseline
python predict_susceptibility_map.py --mode traditional_xgb
```

## 모델 아키텍처

### Fusion Encoder 유형

| 유형 | 설명 |
|------|------|
| **multi** | 5-path 멀티스케일 융합 (각 깊이에서 concat) |
| **cross_attention** | 양방향 cross-attention (연속형 ↔ 범주형) |
| **legacy** | 매 레이어에서 feature fusion module 적용 |

### 실험 변수 (Ablation)

| 변수 | 값 |
|------|------|
| Embedding 라벨 | detailed / undetailed |
| 데이터 변환 | transform (arcsinh+log) / no_transform |
| 대조 손실 | standard / debiased |
| 임베딩 차원 | 32 / 64 |
| 인코더 유형 | multi / cross_attention |
| SSL 적용 | SimCLR / no_pretrain (지도학습) |

## Config YAML 구조

```yaml
device:
  gpu_device: 0

data:
  apply_arcsinh: true          # 곡률 특성 arcsinh 변환
  apply_log: true              # SCA 특성 log 변환
  embedding_dimension: 64      # 임베딩 차원 (32 or 64)
  detailed: true               # 상세/간략 라벨 선택

model:
  model_type: ResNet           # ResNet / ViT
  ssl_type: SimCLR             # SimCLR / no_pretrain
  fusion_encoder_type: cross_attention  # multi / cross_attention / legacy
  temperature: 0.07            # 대조 손실 온도
  debiased: false              # Debiased contrastive loss

train:
  pre_batch_size: 512          # 사전학습 배치 크기
  pre_epochs: 100              # 사전학습 에폭
  learning_rate: 0.00001       # 미세조정 학습률
  fine_tuning_data_ratio: 1.0  # 학습 데이터 비율 (0.01~1.0)
```

## 평가 지표

- **AUC** (Area Under ROC Curve)
- **Accuracy**, **F1 Score**
- **Precision**, **Recall**, **Specificity**
- **Cohen's Kappa**
- 14단계 데이터 비율별 성능 평가 (1%, 2%, 4%, 8%, 10%, 20%, ..., 100%)
