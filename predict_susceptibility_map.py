"""
산사태 취약성 매핑 (Landslide Susceptibility Mapping)
학습된 모델로 연구지역 전체를 예측하고 취약성 지도를 생성합니다.

Usage:
    # Fusion 모델 (proposed, ablation 등)
    python predict_susceptibility_map.py --model_dir <dir> --ratio 50 --mode fusion
    # Supervised single ResNet (20 raw features)
    python predict_susceptibility_map.py --model_dir <dir> --ratio 50 --mode single
    # Traditional ML baseline
    python predict_susceptibility_map.py --ratio 50 --mode traditional_rf
    python predict_susceptibility_map.py --ratio 50 --mode traditional_xgb
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

from data.data_reader import patch_
from utils.utils import fill_nan_nearest_2d, Multi_data_scaler
from model.simclr_model import (
    build_multi_fusion_encoder,
    build_finetune_fusion_model,
    residual_block,
)
from train.train_supervised_ablation import build_simple_resnet_encoder, build_finetune_simple_model

FEATURE_NAMES = [
    "aspect", "curv_plf", "curv_prf", "curv_std", "elev",
    "forest_age", "forest_diameter", "forest_density", "forest_type",
    "geology", "landuse", "sca", "slope",
    "soil_drainage", "soil_series", "soil_sub_texture", "soil_thickness", "soil_texture",
    "spi", "twi",
]
CATEGORICAL_FEATURES = [
    "geology", "landuse", "soil_drainage", "soil_series", "soil_texture",
    "soil_thickness", "soil_sub_texture",
    "forest_age", "forest_diameter", "forest_density", "forest_type",
]
CATEGORICAL_FEATURE_INDICES = {fn: i for i, fn in enumerate(FEATURE_NAMES) if fn in CATEGORICAL_FEATURES}
CONTINUOUS_FEATURE_INDICES = [i for i in range(len(FEATURE_NAMES)) if FEATURE_NAMES[i] not in CATEGORICAL_FEATURES]

TARGET_SIZE = 28
CROP_SIZE = 18
BATCH_SIZE = 512


def apply_transforms(patches):
    patches = patches.copy()
    idx_sca = FEATURE_NAMES.index("sca")
    patches[:, :, :, idx_sca] = np.log1p(patches[:, :, :, idx_sca])
    if np.any(np.isnan(patches[:, :, :, idx_sca])):
        patches[:, :, :, idx_sca] = fill_nan_nearest_2d(patches[:, :, :, idx_sca])
    for name in ["curv_plf", "curv_prf", "curv_std"]:
        idx = FEATURE_NAMES.index(name)
        patches[:, :, :, idx] = np.arcsinh(patches[:, :, :, idx])
        if np.any(np.isnan(patches[:, :, :, idx])):
            patches[:, :, :, idx] = fill_nan_nearest_2d(patches[:, :, :, idx])
    return patches


def prepare_single_input(low_patches):
    """20개 raw features → resize → scale → crop (single ResNet 입력용)."""
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont = low_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat = low_patches[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    all20 = np.zeros((low_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        all20[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        all20[:, :, :, idx] = cat_resized[:, :, :, i]
    scaler = Multi_data_scaler(all20)
    scaled = scaler.multi_scale(all20)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        scaled = scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
    return scaled.astype(np.float32)


def prepare_ml_input(low_patches):
    """20개 raw features → resize → scale → crop → flatten (ML 입력용)."""
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont = low_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat = low_patches[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    all20 = np.zeros((low_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        all20[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        all20[:, :, :, idx] = cat_resized[:, :, :, i]
    scaler = Multi_data_scaler(all20)
    scaled = scaler.multi_scale(all20)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        scaled = scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
    # flatten: (N, 18, 18, 20) → (N, 6480)
    return scaled.reshape(scaled.shape[0], -1).astype(np.float32)


def prepare_fusion_input(low_patches, embedding_path):
    """Continuous + categorical embedding → resize → scale → crop (fusion 모델 입력용)."""
    cont = low_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
    scaler = Multi_data_scaler(cont_resized)
    cont_scaled = scaler.multi_scale(cont_resized)

    print(f"  Loading embeddings: {embedding_path}")
    cat_emb = np.load(embedding_path)
    print(f"  Embedding shape: {cat_emb.shape}")
    if cat_emb.shape[1] != TARGET_SIZE:
        with tf.device("/CPU:0"):
            cat_emb = np.array(tf.image.resize(
                tf.cast(cat_emb, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"
            ))

    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        cont_scaled = cont_scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
        cat_emb = cat_emb[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]

    return cont_scaled.astype(np.float32), cat_emb.astype(np.float32)


def predict_fusion(model_dir, ratio, cont_scaled, cat_emb):
    """Fusion 모델 로드 및 예측."""
    weight_path = os.path.join(model_dir, f"SSL_{ratio}_weight.h5")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    cont_shape = cont_scaled.shape[1:]
    cat_shape = cat_emb.shape[1:]

    tf.keras.backend.clear_session()
    encoder = build_multi_fusion_encoder(cont_shape, cat_shape, layer_name="fusion_encoder")
    model = build_finetune_fusion_model(cont_shape, cat_shape, encoder, num_classes=2, training=False)
    model([tf.zeros((1, *cont_shape)), tf.zeros((1, *cat_shape))])
    model.load_weights(weight_path)
    print(f"  Loaded: {weight_path}")

    n_samples = cont_scaled.shape[0]
    all_preds = []
    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        preds = model.predict([cont_scaled[start:end], cat_emb[start:end]], verbose=0)
        all_preds.append(preds)
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {end}/{n_samples} ({100 * end / n_samples:.1f}%)")

    del model
    return np.concatenate(all_preds, axis=0)[:, 1]


def predict_single(model_dir, ratio, single_input):
    """Single ResNet 모델 로드 및 예측."""
    weight_path = os.path.join(model_dir, f"SSL_{ratio}_weight.h5")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    input_shape = single_input.shape[1:]

    tf.keras.backend.clear_session()
    encoder = build_simple_resnet_encoder(input_shape)
    model = build_finetune_simple_model(input_shape, encoder, num_classes=2, training=False)
    model(tf.zeros((1, *input_shape)))
    model.load_weights(weight_path)
    print(f"  Loaded: {weight_path}")

    n_samples = single_input.shape[0]
    all_preds = []
    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)
        preds = model.predict(single_input[start:end], verbose=0)
        all_preds.append(preds)
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {end}/{n_samples} ({100 * end / n_samples:.1f}%)")

    del model
    return np.concatenate(all_preds, axis=0)[:, 1]


def predict_ml(mode, ratio, ml_input, train_data=None, train_labels=None):
    """Random Forest 또는 XGBoost 모델 학습 및 예측."""
    print(f"  Training {mode.upper()} on ratio={ratio}%...")

    if mode == "traditional_rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, n_jobs=-1, random_state=118,
        )
    else:  # traditional_xgb
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=118, verbosity=0,
        )

    model.fit(train_data, train_labels)
    print(f"  Predicting with {mode.upper()}...")
    pred_prob = model.predict_proba(ml_input)[:, 1]
    return pred_prob


def plot_maps(susceptibility, cx, cy, ls_coor, ls_label, model_name, ratio, output_dir, dpi):
    """연속 컬러맵 + 5등급 분류맵 생성."""
    os.makedirs(output_dir, exist_ok=True)

    cmap_discrete = ListedColormap(["darkgreen", "lawngreen", "palegoldenrod", "coral", "firebrick"])
    caution_labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    legend_elements = [mpatches.Patch(color=cmap_discrete.colors[i], label=caution_labels[i]) for i in range(5)]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", label="Landslide",
               markerfacecolor="k", markersize=4)
    )

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    # 연속 컬러맵
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(cx, cy, s=5, c=susceptibility, cmap="RdYlGn_r", vmin=0, vmax=1,
                    marker=".", linewidths=0, edgecolors="none")
    ax.scatter(ls_coor[:, 0], ls_coor[:, 1], s=3, c=ls_label[:, 0], cmap="gray",
               marker=".", linewidths=0, edgecolors="none")
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    plt.colorbar(sc, ax=ax, label="Landslide Probability", shrink=0.7, pad=0.02)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(f"Landslide Susceptibility Map\n({model_name}, ratio={ratio}%)")
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"susceptibility_ratio{ratio}.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    # 5등급 분류 맵
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    class_indices = np.clip(np.digitize(susceptibility, boundaries) - 1, 0, 4)
    colors = [cmap_discrete.colors[c] for c in class_indices]
    ax2.scatter(cx, cy, s=5, c=colors, marker=".", linewidths=0, edgecolors="none")
    ax2.scatter(ls_coor[:, 0], ls_coor[:, 1], s=3, c=ls_label[:, 0], cmap="gray",
                marker=".", linewidths=0, edgecolors="none")
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    ax2.set_xlabel("Easting")
    ax2.set_ylabel("Northing")
    ax2.set_title(f"Landslide Susceptibility Classification\n({model_name}, ratio={ratio}%)")
    ax2.set_aspect("equal")
    plt.tight_layout()

    out_path2 = os.path.join(output_dir, f"susceptibility_class_ratio{ratio}.png")
    plt.savefig(out_path2, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {out_path}, {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Landslide Susceptibility Mapping")
    parser.add_argument("--model_dir", type=str, default="",
                        help="모델 가중치 디렉토리 경로 (fusion/single 모드 필수)")
    parser.add_argument("--ratio", type=int, default=50,
                        help="사용할 data ratio (모델 가중치 선택용)")
    parser.add_argument("--embedding_path", type=str, default="",
                        help="categorical embedding 파일 경로 (fusion 모드에서 필요)")
    parser.add_argument("--mode", type=str, default="fusion",
                        choices=["fusion", "single", "traditional_rf", "traditional_xgb"],
                        help="fusion: multi fusion encoder, single: 20 raw features + simple ResNet, traditional_rf/xgb: sklearn baselines")
    parser.add_argument("--gpu_device", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./images/susceptibility_maps")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Traditional ML 아닌 경우 model_dir 필수
    if not args.mode.startswith("traditional"):
        if not args.model_dir:
            raise ValueError("--model_dir is required for fusion and single modes")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 70)
    print("Landslide Susceptibility Mapping")
    if args.model_dir:
        print(f"  Model: {args.model_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  Ratio: {args.ratio}%")
    print("=" * 70)

    # 1. 연구지역 패치 생성
    print("\n[1/4] Loading study area patches...")
    tif_img = np.load("./dataset/tif_img.npy")
    ls_img = np.load("./dataset/ls_img.npy")
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)
    low_patches, _, _, _, _, coor_list = patch_(
        [554270.0, 562860.0, 859], [331270.0, 342070.0, 1080], tif_img, 6, 3
    )
    print(f"  Patches: {low_patches.shape}")

    # 2. 전처리
    print("\n[2/4] Preprocessing...")
    low_patches = apply_transforms(low_patches)

    # 3. 예측
    print("\n[3/4] Predicting...")
    if args.mode in ["traditional_rf", "traditional_xgb"]:
        # Traditional ML: 학습 데이터 로드 필요
        print("  Loading training data for traditional ML...")
        non_ls_patches = np.load("./dataset/0926/non_ls_patches_60dist.npy")
        non_ls_labels = np.load("./dataset/0926/non_ls_labels_60dist.npy")
        ls_patches = np.load("./dataset/0926/ls_patches_60dist.npy")
        ls_labels = np.load("./dataset/0926/ls_labels_60dist.npy")

        # Train/valid 분리
        np.random.seed(118)
        patch_size = 6
        valid_non_ls_ind = np.random.choice(
            non_ls_patches.shape[0], int(ls_labels.shape[0] * 0.1), replace=False
        )
        valid_ls_ind = np.random.choice(
            ls_patches.shape[0], int(ls_labels.shape[0] * 0.1), replace=False
        )

        train_non_ls = np.delete(non_ls_patches, valid_non_ls_ind, axis=0)
        train_non_ls_label = np.delete(non_ls_labels, valid_non_ls_ind, axis=0)
        train_ls = np.delete(ls_patches, valid_ls_ind, axis=0)
        train_ls_label = np.delete(ls_labels, valid_ls_ind, axis=0)

        train_non_ls_ind_full = np.random.choice(
            train_non_ls.shape[0], int(train_ls_label.shape[0]), replace=False
        )
        train_ls_ind_full = np.random.choice(
            train_ls.shape[0], int(train_ls.shape[0]), replace=False
        )

        def center_labels(labels):
            out = []
            for i in range(labels.shape[0]):
                if np.sum(labels[i, patch_size - 1:patch_size + 1, patch_size - 1:patch_size + 1]) >= 1.0:
                    out.append(1)
                else:
                    out.append(0)
            return np.array(out, dtype=np.int32)

        full_train_patches = np.concatenate([
            train_ls[train_ls_ind_full], train_non_ls[train_non_ls_ind_full]
        ])
        full_train_labels = np.concatenate([
            train_ls_label[train_ls_ind_full], train_non_ls_label[train_non_ls_ind_full]
        ])
        full_train_labels_ = center_labels(full_train_labels)

        # 학습 데이터 전처리
        print("  Preprocessing training data...")
        full_train_flat = prepare_ml_input(full_train_patches)
        print(f"  Train flat: {full_train_flat.shape}")

        # 연구지역 데이터 전처리
        ml_input = prepare_ml_input(low_patches)
        print(f"  Study area flat: {ml_input.shape}")

        # 비율별 학습
        n_ls_full = len(train_ls_ind_full)
        n_non_ls_full = len(train_non_ls_ind_full)
        n_ls = max(1, int(n_ls_full * args.ratio / 100))
        n_n = max(1, int(n_non_ls_full * args.ratio / 100))
        train_X = np.concatenate([full_train_flat[:n_ls], full_train_flat[n_ls_full:n_ls_full + n_n]])
        train_y = np.concatenate([full_train_labels_[:n_ls], full_train_labels_[n_ls_full:n_ls_full + n_n]])

        susceptibility = predict_ml(args.mode, args.ratio, ml_input, train_X, train_y)

    elif args.mode == "fusion":
        cont_scaled, cat_emb = prepare_fusion_input(low_patches, args.embedding_path)
        print(f"  Continuous: {cont_scaled.shape}, Categorical: {cat_emb.shape}")
        susceptibility = predict_fusion(args.model_dir, args.ratio, cont_scaled, cat_emb)
    else:
        single_input = prepare_single_input(low_patches)
        print(f"  Single input: {single_input.shape}")
        susceptibility = predict_single(args.model_dir, args.ratio, single_input)

    print(f"  Prob range: [{susceptibility.min():.4f}, {susceptibility.max():.4f}]")

    # 4. 시각화
    print("\n[4/4] Generating maps...")
    ls_coor = np.load("./dataset/ls_coor.npy")
    ls_label = np.load("./dataset/ls_labels.npy")
    cx = coor_list[:, 5, 5, 0]
    cy = coor_list[:, 5, 5, 1]

    model_name = os.path.basename(args.model_dir)
    plot_maps(susceptibility, cx, cy, ls_coor, ls_label, model_name, args.ratio, args.output_dir, args.dpi)

    pred_path = os.path.join(args.output_dir, f"predictions_ratio{args.ratio}.npy")
    np.save(pred_path, susceptibility)
    print(f"  Predictions saved: {pred_path}")
    print("Done.")


if __name__ == "__main__":
    main()
