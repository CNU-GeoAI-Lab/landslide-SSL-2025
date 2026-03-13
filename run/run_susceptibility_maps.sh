#!/bin/bash
cd "$(dirname "$0")/.."
# 모든 ablation + supervised 모델에 대해 14개 ratio별 산사태 취약성 맵 생성
# 각 모델별 별도 폴더에 저장

PYTHON=/home/jongchan/anaconda3/envs/ls_simclr/bin/python
GPU=1
RATIOS="1 2 4 8 10 20 30 40 50 60 70 80 90 100"

# Fusion 모델: model_dir | embedding_path | output_folder_name
FUSION_MODELS=(
    "finetuned_models/fusion_resnet_transform_db_detailed_dim64_0122|./embeddings/detailed_embedding64.npy|proposed"
    "finetuned_models/fusion_resnet_detailed_no_transform_db_0205|./embeddings/detailed_embedding64.npy|ablation_no_transform"
    "finetuned_models/fusion_resnet_undetailed_transform_db_0205|./embeddings/undetailed_embedding64.npy|ablation_undetailed"
    "finetuned_models/fusion_resnet_transform_detailed_dim64_0122|./embeddings/detailed_embedding64.npy|ablation_no_debiased"
    "finetuned_models/fusion_resnet_transform_db_detailed_dim32_0127|./embeddings/detailed_embedding32.npy|ablation_dim32"
    "trained_supervised/ablation_supervised_multi_fusion|./embeddings/detailed_embedding64.npy|supervised_multi_fusion"
)

# Single ResNet 모델: model_dir | output_folder_name
SINGLE_MODELS=(
    "trained_supervised/ablation_supervised_single20_resnet|supervised_single20_resnet"
)

# Traditional ML 모델: mode | output_folder_name
TRADITIONAL_MODELS=(
    "traditional_rf|traditional_random_forest"
    "traditional_xgb|traditional_xgboost"
)

# ===== Fusion 모델 실행 =====
for entry in "${FUSION_MODELS[@]}"; do
    IFS='|' read -r MODEL_DIR EMB_PATH FOLDER_NAME <<< "$entry"
    OUTPUT_DIR="./images/susceptibility_maps/${FOLDER_NAME}"

    echo ""
    echo "======================================================================"
    echo " [FUSION] ${FOLDER_NAME}"
    echo " Model: ${MODEL_DIR}"
    echo " Output: ${OUTPUT_DIR}"
    echo "======================================================================"

    for RATIO in $RATIOS; do
        WEIGHT="${MODEL_DIR}/SSL_${RATIO}_weight.h5"
        if [ ! -f "$WEIGHT" ]; then
            echo "  [SKIP] ratio=${RATIO}% - weight not found"
            continue
        fi
        echo "  [RUN] ratio=${RATIO}%"
        $PYTHON -u predict_susceptibility_map.py \
            --model_dir "$MODEL_DIR" \
            --ratio "$RATIO" \
            --embedding_path "$EMB_PATH" \
            --mode fusion \
            --gpu_device "$GPU" \
            --output_dir "$OUTPUT_DIR" \
            --dpi 300
    done
    echo "  [DONE] ${FOLDER_NAME}"
done

# ===== Single ResNet 모델 실행 =====
for entry in "${SINGLE_MODELS[@]}"; do
    IFS='|' read -r MODEL_DIR FOLDER_NAME <<< "$entry"
    OUTPUT_DIR="./images/susceptibility_maps/${FOLDER_NAME}"

    echo ""
    echo "======================================================================"
    echo " [SINGLE] ${FOLDER_NAME}"
    echo " Model: ${MODEL_DIR}"
    echo " Output: ${OUTPUT_DIR}"
    echo "======================================================================"

    for RATIO in $RATIOS; do
        WEIGHT="${MODEL_DIR}/SSL_${RATIO}_weight.h5"
        if [ ! -f "$WEIGHT" ]; then
            echo "  [SKIP] ratio=${RATIO}% - weight not found"
            continue
        fi
        echo "  [RUN] ratio=${RATIO}%"
        $PYTHON -u predict_susceptibility_map.py \
            --model_dir "$MODEL_DIR" \
            --ratio "$RATIO" \
            --mode single \
            --gpu_device "$GPU" \
            --output_dir "$OUTPUT_DIR" \
            --dpi 300
    done
    echo "  [DONE] ${FOLDER_NAME}"
done

# ===== Traditional ML 모델 실행 =====
for entry in "${TRADITIONAL_MODELS[@]}"; do
    IFS='|' read -r MODE FOLDER_NAME <<< "$entry"
    OUTPUT_DIR="./images/susceptibility_maps/${FOLDER_NAME}"

    echo ""
    echo "======================================================================="
    echo " [TRADITIONAL ML] ${FOLDER_NAME}"
    echo " Mode: ${MODE}"
    echo " Output: ${OUTPUT_DIR}"
    echo "======================================================================="

    for RATIO in $RATIOS; do
        echo "  [RUN] ratio=${RATIO}%"
        $PYTHON -u predict_susceptibility_map.py \
            --ratio "$RATIO" \
            --mode "$MODE" \
            --gpu_device "$GPU" \
            --output_dir "$OUTPUT_DIR" \
            --dpi 300
    done
    echo "  [DONE] ${FOLDER_NAME}"
done

echo ""
echo "======================================================================"
echo "All susceptibility maps generated."
echo "======================================================================"
