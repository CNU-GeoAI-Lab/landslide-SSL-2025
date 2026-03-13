#!/bin/bash
cd "$(dirname "$0")/.."
# ============================================================
# Ablation Study 지도학습 실험 실행 스크립트
#
# 새로 학습해야 하는 실험 2개:
#   1) single_resnet: 20 raw features + 단순 ResNet (임베딩/퓨전 없음)
#   2) multi_fusion:  embedding + multi (5-path) fusion encoder (SSL 없이 지도학습)
#
# 각 실험에서 14개 data ratio (1,2,4,8,10,20,30,40,50,60,70,80,90,100%)
# 별로 모델을 학습하여 저장.
#
# 사용법:
#   ./run_ablation.sh          # 두 실험 순차 실행
#   ./run_ablation.sh single   # single_resnet만 실행
#   ./run_ablation.sh fusion   # multi_fusion만 실행
# ============================================================

set -e

GPU_DEVICE=1
CONDA_ENV="ls_simclr"

echo "============================================================"
echo " Ablation Study: Supervised Training"
echo " GPU: ${GPU_DEVICE}"
echo "============================================================"

run_single_resnet() {
    echo ""
    echo "============================================================"
    echo " [1/2] Single ResNet (20 raw features, no embedding/fusion)"
    echo "============================================================"
    conda run -n ${CONDA_ENV} --no-capture-output python train_supervised_ablation.py \
        --config configs/config_ablation_supervised_single20_resnet.yaml \
        --mode single_resnet \
        --gpu_device ${GPU_DEVICE}
    echo ""
    echo "[1/2] Single ResNet 학습 완료."
}

run_multi_fusion() {
    echo ""
    echo "============================================================"
    echo " [2/2] Multi Fusion (embedding + 5-path encoder, supervised)"
    echo "============================================================"
    conda run -n ${CONDA_ENV} --no-capture-output python train_supervised_ablation.py \
        --config configs/config_ablation_supervised_multi_fusion.yaml \
        --mode multi_fusion \
        --gpu_device ${GPU_DEVICE}
    echo ""
    echo "[2/2] Multi Fusion 학습 완료."
}

# 인자에 따라 실행할 실험 선택
case "${1}" in
    single)
        run_single_resnet
        ;;
    fusion)
        run_multi_fusion
        ;;
    *)
        run_single_resnet
        run_multi_fusion
        ;;
esac

echo ""
echo "============================================================"
echo " Ablation 학습 완료!"
echo " 결과 위치:"
echo "   trained_supervised/ablation_supervised_single20_resnet/"
echo "   trained_supervised/ablation_supervised_multi_fusion/"
echo ""
echo " 테스트 실행:"
echo "   conda run -n ${CONDA_ENV} python test_ablation.py"
echo "============================================================"
