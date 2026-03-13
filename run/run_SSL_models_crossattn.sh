#!/bin/bash
cd "$(dirname "$0")/.."
# Cross-attention fusion: 8 combinations
# (detailed/undetailed) x (transform/no_transform) x (debiased/no)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

CONFIGS=(
    # "configs/config_fusion_resnet_crossattn_detailed_transform.yaml"
    # "configs/config_fusion_resnet_crossattn_detailed_transform_db.yaml"
    # "configs/config_fusion_resnet_crossattn_detailed_no_transform.yaml"
    # "configs/config_fusion_resnet_crossattn_detailed_no_transform_db.yaml"
    # "configs/config_fusion_resnet_crossattn_undetailed_transform.yaml"
    "configs/config_fusion_resnet_crossattn_undetailed_transform_db.yaml"
    "configs/config_fusion_resnet_crossattn_undetailed_no_transform.yaml"
    "configs/config_fusion_resnet_crossattn_undetailed_no_transform_db.yaml"
)

DIR_NAMES=(
    # "fusion_resnet_crossattn_detailed_transform_0228"
    # "fusion_resnet_crossattn_detailed_transform_db_0228"
    # "fusion_resnet_crossattn_detailed_no_transform_0228"
    # "fusion_resnet_crossattn_detailed_no_transform_db_0228"
    # "fusion_resnet_crossattn_undetailed_transform_0228"
    "fusion_resnet_crossattn_undetailed_transform_db_0228"
    "fusion_resnet_crossattn_undetailed_no_transform_0228"
    "fusion_resnet_crossattn_undetailed_no_transform_db_0228"
)

FILES_NAMES=(
    "SSL_1" "SSL_2" "SSL_4" "SSL_8" "SSL_10" "SSL_20" "SSL_30" "SSL_40"
    "SSL_50" "SSL_60" "SSL_70" "SSL_80" "SSL_90" "SSL_100"
)

DATA_USED=(
    "0.01" "0.02" "0.04" "0.08" "0.10" "0.20" "0.30" "0.40"
    "0.50" "0.60" "0.70" "0.80" "0.90" "1.00"
)

PYTHON_CMD="python3"
GPU_DEVICE=1

j=0
for config in "${CONFIGS[@]}"; do
    if [ ! -f "$config" ]; then
        echo -e "${RED}Error: Config file not found: $config${NC}"
        j=$((j + 1))
        continue
    fi

    dir_name="${DIR_NAMES[$j]}"
    j=$((j + 1))

    i=0
    for file_name in "${FILES_NAMES[@]}"; do
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}Running: main.py | $config${NC}"
        echo -e "${GREEN}  dir_name: $dir_name | model: $file_name | ratio: ${DATA_USED[$i]}${NC}"
        echo -e "${BLUE}========================================${NC}"

        $PYTHON_CMD main.py --config "$config" --gpu_device $GPU_DEVICE \
            --dir_name "$dir_name" --fine_trained_model_name "$file_name" \
            --fine_tuning_data_ratio "${DATA_USED[$i]}"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Done: $dir_name / $file_name / ${DATA_USED[$i]}${NC}"
        else
            echo -e "${RED}✗ Failed: $dir_name / $file_name / ${DATA_USED[$i]}${NC}"
        fi
        i=$((i + 1))
        echo ""
    done
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All cross-attention experiments completed.${NC}"
echo -e "${BLUE}========================================${NC}"
