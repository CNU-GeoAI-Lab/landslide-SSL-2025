# Validation loss가 떨어지지 않는 원인 분석

## main_embedding_fusion.py vs train_supervised_crossattn_compare.py 비교

### 1. 공통점 (차이 아님)
- 동일 모델: cross-attention encoder + build_finetune_fusion_model
- 동일 learning_rate: 0.00001 (config)
- 동일 batch_size: 32
- 동일 seed: 118
- 동일 train/valid split 로직
- MinMax scaler: utils.Multi_data_scaler (동일)
- Categorical: detailed_embedding 사용

### 2. 차이점 및 가능한 원인

| 항목 | main_embedding_fusion (no_pretrain) | train_supervised_crossattn_compare | 영향 |
|------|-------------------------------------|-------------------------------------|------|
| **사전학습** | pretrain 없음 → 전체 가중치 학습 | 동일 | 동일 |
| **Learning rate** | 0.00001 | 0.00001 | 동일. 다만 scratch 학습에 LR이 낮을 수 있음 |
| **데이터 로딩** | np.load(npy) 직접 | 동일 | 동일 |
| **fine_tuning_areas** | config에서 10 | load_config에서 10 | 동일 |
| **실제 사용 데이터** | non_ls, ls from npy | 동일 | 동일 |

### 3. 검토된 가능 원인

1. **Learning rate가 너무 낮음 (0.00001)**
   - Scratch 학습(사전학습 없음)에서는 전체 encoder+head를 학습해야 함
   - 0.00001은 pretrained encoder fine-tuning에 적합한 값
   - 제안: scratch 학습 시 0.0001 ~ 0.001 시도

2. **Continuous factor transform (log, arcsinh)**
   - log1p(sca): sca=0 근처에서 기울기 급변
   - arcsinh(curv): 큰 값에서 saturate, 스케일 왜곡 가능
   - transform 미적용 시 원시값으로 학습하면 분포가 더 안정적일 수 있음

3. **Scaler fit/transform 일관성**
   - scaler는 real_patches(전체 자료)에 fit
   - train/valid는 동일 transform 후 scale
   - transform 적용 여부가 scaler fitting 데이터와 일치해야 함

4. **Data ratio가 낮을 때 (1%, 2% 등)**
   - 극소량 데이터로 학습 시 validation loss가 요동
   - early stopping, reduce_lr 등이 조기 종료 유발 가능
