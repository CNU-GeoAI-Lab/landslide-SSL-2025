"""
Traditional ML baseline: Random Forest, XGBoost
동일한 seed(118), 동일한 train/valid/test 분할, 동일한 전처리.
패치를 flatten하여 ML 입력으로 사용.
결과를 ablation_results_0311.csv에 추가.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from data.data_reader import patch_
from data.transform import *
from utils.utils import fill_nan_nearest_2d, Multi_data_scaler

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
DATA_SEED = 118
DATA_RATIOS = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def apply_transforms(patches, apply_log=True, apply_arcsinh=True):
    patches = patches.copy()
    if apply_log:
        idx = FEATURE_NAMES.index("sca")
        patches[:, :, :, idx] = np.log1p(patches[:, :, :, idx])
        if np.any(np.isnan(patches[:, :, :, idx])):
            patches[:, :, :, idx] = fill_nan_nearest_2d(patches[:, :, :, idx])
    if apply_arcsinh:
        for name in ["curv_plf", "curv_prf", "curv_std"]:
            idx = FEATURE_NAMES.index(name)
            patches[:, :, :, idx] = np.arcsinh(patches[:, :, :, idx])
            if np.any(np.isnan(patches[:, :, :, idx])):
                patches[:, :, :, idx] = fill_nan_nearest_2d(patches[:, :, :, idx])
    return patches


def extract_center_labels(labels, patch_size=6):
    if len(labels.shape) == 4:
        center = labels[:, patch_size - 1:patch_size + 1, patch_size - 1:patch_size + 1]
        return (np.sum(center, axis=(1, 2)) >= 1.0).astype(np.int32)
    else:
        return (labels[:, patch_size, patch_size] >= 1.0).astype(np.int32)


def process_patches_to_flat(patches, scaler, apply_log=True, apply_arcsinh=True):
    """패치를 전처리 → resize → scale → crop → flatten."""
    patches = apply_transforms(patches, apply_log, apply_arcsinh)
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont = patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat = patches[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    all20 = np.zeros((patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        all20[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        all20[:, :, :, idx] = cat_resized[:, :, :, i]
    scaled = scaler.multi_scale(all20)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        scaled = scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
    # flatten: (N, 18, 18, 20) → (N, 6480)
    return scaled.reshape(scaled.shape[0], -1).astype(np.float32)


def evaluate(pred_prob, labels):
    pred_binary = (pred_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(labels, pred_prob)
    auc_score = auc(fpr, tpr)
    return {
        "accuracy": accuracy_score(labels, pred_binary),
        "precision": precision_score(labels, pred_binary, zero_division=0),
        "recall": recall_score(labels, pred_binary, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(labels, pred_binary, zero_division=0),
        "kappa": cohen_kappa_score(labels, pred_binary),
        "auc": auc_score,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("=" * 70)
    print("Traditional ML Baseline: Random Forest & XGBoost")
    print("=" * 70)

    # ===== 1. 데이터 로드 =====
    print("\n[1/5] Loading data...")
    non_ls_patches = np.load("./dataset/0926/non_ls_patches_60dist.npy")
    non_ls_labels = np.load("./dataset/0926/non_ls_labels_60dist.npy")
    ls_patches = np.load("./dataset/0926/ls_patches_60dist.npy")
    ls_labels = np.load("./dataset/0926/ls_labels_60dist.npy")
    test_patches = np.load("./dataset/1013/test_85patches.npy")
    test_labels = np.load("./dataset/1013/test_85labels.npy")
    test_labels[test_labels == 0.001] = 0.0
    test_label_center = extract_center_labels(test_labels)
    print(f"  LS: {ls_patches.shape}, Non-LS: {non_ls_patches.shape}, Test: {test_patches.shape}")

    # ===== 2. Train/Valid 분리 (seed=118, 기존과 동일) =====
    print("\n[2/5] Splitting train/valid (seed=118)...")
    np.random.seed(DATA_SEED)
    patch_size = 6

    valid_non_ls_ind = np.random.choice(
        non_ls_patches.shape[0], int(ls_labels.shape[0] * 0.1), replace=False
    )
    valid_ls_ind = np.random.choice(
        ls_patches.shape[0], int(ls_labels.shape[0] * 0.1), replace=False
    )

    valid_patches = np.concatenate([ls_patches[valid_ls_ind], non_ls_patches[valid_non_ls_ind]])
    valid_labels = np.concatenate([ls_labels[valid_ls_ind], non_ls_labels[valid_non_ls_ind]])

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

    valid_labels_ = center_labels(valid_labels)
    n_ls_full = len(train_ls_ind_full)
    n_non_ls_full = len(train_non_ls_ind_full)
    print(f"  Train LS: {n_ls_full}, Train Non-LS: {n_non_ls_full}, Valid: {len(valid_patches)}")

    # ===== 3. Scaler 구성 =====
    print("\n[3/5] Building scaler...")
    tif_img = np.load("./dataset/tif_img.npy")
    ls_img = np.load("./dataset/ls_img.npy")
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)
    real_patches, _, _, _, _, _ = patch_(
        [554270.0, 562860.0, 859], [331270.0, 342070.0, 1080], tif_img, 6, 3
    )
    real_transformed = apply_transforms(real_patches, apply_log=True, apply_arcsinh=True)
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont_real = real_transformed[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat_real = real_transformed[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont_real, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat_real, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    real_all20 = np.zeros((real_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        real_all20[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        real_all20[:, :, :, idx] = cat_resized[:, :, :, i]
    scaler = Multi_data_scaler(real_all20)

    # ===== 4. 전처리: train, valid, test → flatten =====
    print("\n[4/5] Preprocessing & flattening patches...")

    # 전체 학습 데이터 전처리
    full_train_patches = np.concatenate([
        train_ls[train_ls_ind_full], train_non_ls[train_non_ls_ind_full]
    ])
    full_train_labels = np.concatenate([
        train_ls_label[train_ls_ind_full], train_non_ls_label[train_non_ls_ind_full]
    ])
    full_train_labels_ = center_labels(full_train_labels)

    full_train_flat = process_patches_to_flat(full_train_patches, scaler)
    valid_flat = process_patches_to_flat(valid_patches, scaler)
    test_flat = process_patches_to_flat(test_patches, scaler)

    print(f"  Train flat: {full_train_flat.shape}, Valid flat: {valid_flat.shape}, Test flat: {test_flat.shape}")

    # ===== 5. 학습 & 평가 =====
    print("\n[5/5] Training & evaluating...")
    all_results = []

    models_config = [
        {
            "name": "random_forest",
            "ablation_role": "baseline_random_forest",
            "build_fn": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                min_samples_leaf=2, n_jobs=-1, random_state=DATA_SEED,
            ),
        },
        {
            "name": "xgboost",
            "ablation_role": "baseline_xgboost",
            "build_fn": lambda: XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=DATA_SEED,
                verbosity=0,
            ),
        },
    ]

    for mc in models_config:
        print(f"\n{'=' * 70}")
        print(f"[{mc['name'].upper()}]")
        print(f"{'=' * 70}")

        for ratio in DATA_RATIOS:
            n_ls = max(1, int(n_ls_full * ratio / 100))
            n_n = max(1, int(n_non_ls_full * ratio / 100))
            train_X = np.concatenate([full_train_flat[:n_ls], full_train_flat[n_ls_full:n_ls_full + n_n]])
            train_y = np.concatenate([full_train_labels_[:n_ls], full_train_labels_[n_ls_full:n_ls_full + n_n]])

            clf = mc["build_fn"]()

            if mc["name"] == "xgboost":
                clf.fit(train_X, train_y, eval_set=[(valid_flat, valid_labels_)], verbose=False)
            else:
                clf.fit(train_X, train_y)

            pred_prob = clf.predict_proba(test_flat)[:, 1]
            metrics = evaluate(pred_prob, test_label_center)

            row = {
                "experiment": mc["name"],
                "ablation_role": mc["ablation_role"],
                "encoder_type": mc["name"],
                "detailed": False,
                "transform": True,
                "debiased": False,
                "embedding_dim": 0,
                "data_ratio": ratio,
                **metrics,
            }
            all_results.append(row)
            print(f"  ratio={ratio:>3d}%: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")

    # ===== 결과 저장 =====
    if all_results:
        new_df = pd.DataFrame(all_results)
        csv_path = "./test_results/ablation_results_0311.csv"
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            existing = existing[~existing["ablation_role"].isin([
                "baseline_random_forest", "baseline_xgboost"
            ])]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(f"Total rows: {len(combined)}")

        print(f"\n{'=' * 70}")
        print("Best Accuracy per ablation role:")
        print("-" * 90)
        for role in combined["ablation_role"].unique():
            sub = combined[combined["ablation_role"] == role]
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  {role:45s}: Acc={best['accuracy']:.4f}  AUC={best['auc']:.4f}  ratio={int(best['data_ratio'])}%")

        print(f"\n{'=' * 70}")
        print("Best AUC per ablation role:")
        print("-" * 90)
        for role in combined["ablation_role"].unique():
            sub = combined[combined["ablation_role"] == role]
            best = sub.loc[sub["auc"].idxmax()]
            print(f"  {role:45s}: AUC={best['auc']:.4f}  Acc={best['accuracy']:.4f}  ratio={int(best['data_ratio'])}%")


if __name__ == "__main__":
    main()
