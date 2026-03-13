"""
Ablation study: 지도학습 baseline 모델 테스트.
1) single_resnet: 20 raw features + plain ResNet
2) multi_fusion: embedding + multi (5-path) encoder, supervised only

결과를 ablation_results_0311.csv에 추가.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score,
    confusion_matrix,
)
from data.data_reader import patch_
from data.transform import *
from utils.utils import fill_nan_nearest_2d, Multi_data_scaler
from model.simclr_model import (
    build_multi_fusion_encoder,
    build_finetune_fusion_model,
    residual_block,
)
from train.train_supervised_ablation import build_simple_resnet_encoder, build_finetune_simple_model
from embeddings.openai_embedding import create_categorical_embeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FEATURE_NAMES = [
    "aspect", "curv_plf", "curv_prf", "curv_std", "elev",
    "forest_age", "forest_diameter", "forest_density", "forest_type",
    "geology", "landuse", "sca", "slope",
    "soil_drainage", "soil_series", "soil_sub_texture", "soil_thickness", "soil_texture",
    "spi", "twi",
]
FEATURE_TO_LABEL_MAPPING = {
    "geology": ("Geology", "geology"),
    "landuse": ("Land_use", "landuse"),
    "soil_drainage": ("Soil_summary", "soil_drainage"),
    "soil_series": ("Soil_summary", "soil_series"),
    "soil_texture": ("Soil_summary", "soil_texture"),
    "soil_thickness": ("Soil_summary", "soil_thickness"),
    "soil_sub_texture": ("Soil_summary", "soil_sub_texture"),
    "forest_age": ("Forest_summary", "forest_age"),
    "forest_diameter": ("Forest_summary", "forest_diameter"),
    "forest_density": ("Forest_summary", "forest_density"),
    "forest_type": ("Forest_summary", "forest_type"),
}
CATEGORICAL_FEATURES = list(FEATURE_TO_LABEL_MAPPING.keys())
CATEGORICAL_FEATURE_INDICES = {fn: i for i, fn in enumerate(FEATURE_NAMES) if fn in CATEGORICAL_FEATURES}
CONTINUOUS_FEATURE_INDICES = [i for i in range(len(FEATURE_NAMES)) if FEATURE_NAMES[i] not in CATEGORICAL_FEATURES]

TARGET_SIZE = 28
CROP_SIZE = 18
EMBEDDING_DIM = 64
DATA_RATIOS = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def extract_center_labels(test_labels, patch_size=6):
    if len(test_labels.shape) == 4:
        center = test_labels[:, patch_size - 1:patch_size + 1, patch_size - 1:patch_size + 1]
        return (np.sum(center, axis=(1, 2)) >= 1.0).astype(np.int32)
    else:
        return (test_labels[:, patch_size, patch_size] >= 1.0).astype(np.int32)


def apply_transforms(patches, apply_log, apply_arcsinh):
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


def evaluate(predictions, labels):
    pred_prob = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 70)
    print("Ablation: Supervised Baseline Test")
    print("=" * 70)

    # Load test data
    print("\n[1/4] Loading test data...")
    test_patches = np.load("./dataset/1013/test_85patches.npy")
    test_labels = np.load("./dataset/1013/test_85labels.npy")
    test_labels[test_labels == 0.001] = 0.0
    test_label_center = extract_center_labels(test_labels)
    print(f"  Test: {test_patches.shape}, Labels: {test_label_center.shape}")

    # Build scalers
    print("\n[2/4] Building scalers...")
    tif_img = np.load("./dataset/tif_img.npy")
    ls_img = np.load("./dataset/ls_img.npy")
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)
    real_patches, _, _, _, _, _ = patch_(
        [554270.0, 562860.0, 859], [331270.0, 342070.0, 1080], tif_img, 6, 3
    )

    # Scaler for single 20ch
    real_transformed = apply_transforms(real_patches, apply_log=True, apply_arcsinh=True)
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont_real = real_transformed[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat_real = real_transformed[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont_real, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized_f = np.array(tf.cast(
            tf.image.resize(tf.cast(cat_real, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    real_all20 = np.zeros((real_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        real_all20[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        real_all20[:, :, :, idx] = cat_resized_f[:, :, :, i]
    scaler_all20 = Multi_data_scaler(real_all20)

    # Scaler for continuous only (fusion)
    scaler_cont = Multi_data_scaler(cont_resized)

    # Prepare test inputs
    print("\n[3/4] Preparing test inputs...")

    # Single 20ch input
    test_transformed = apply_transforms(test_patches, apply_log=True, apply_arcsinh=True)
    test_cont = test_transformed[:, :, :, CONTINUOUS_FEATURE_INDICES]
    test_cat = test_transformed[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        test_cont_resized = np.array(tf.image.resize(test_cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        test_cat_resized_f = np.array(tf.cast(
            tf.image.resize(tf.cast(test_cat, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32))
    test_all20 = np.zeros((test_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        test_all20[:, :, :, idx] = test_cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        test_all20[:, :, :, idx] = test_cat_resized_f[:, :, :, i]
    test_single = scaler_all20.multi_scale(test_all20)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        test_single = test_single[:, s:s+CROP_SIZE, s:s+CROP_SIZE, :]
    print(f"  Single 20ch: {test_single.shape}")

    # Fusion input (cont + embedding)
    test_cont_scaled = scaler_cont.multi_scale(test_cont_resized)
    cache_path = "./test_cache/test_cat_emb_detailed_transform_64d.npy"
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings: {cache_path}")
        test_cat_emb = np.load(cache_path)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        with tf.device("/CPU:0"):
            cat_resized_int = np.array(tf.cast(
                tf.image.resize(tf.cast(test_cat, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
                tf.int32))
        temp = np.zeros((*cat_resized_int.shape[:3], len(FEATURE_NAMES)))
        for idx, feat in enumerate(CATEGORICAL_FEATURES):
            temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat]] = cat_resized_int[:, :, :, idx]
        test_cat_emb, _ = create_categorical_embeddings(
            temp, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
            labels_path="description/detailed_labels.json", feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
            client=client, model="text-embedding-3-small", dimensions=EMBEDDING_DIM, batch_size=100, verbose=True,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, test_cat_emb)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        test_cont_scaled = test_cont_scaled[:, s:s+CROP_SIZE, s:s+CROP_SIZE, :]
        test_cat_emb = test_cat_emb[:, s:s+CROP_SIZE, s:s+CROP_SIZE, :]
    print(f"  Fusion cont: {test_cont_scaled.shape}, cat: {test_cat_emb.shape}")

    # Run tests
    print("\n[4/4] Running tests...")
    all_results = []

    experiments = [
        {
            "dir": "trained_supervised/ablation_supervised_single20_resnet",
            "ablation_role": "supervised_single20_resnet",
            "mode": "single",
        },
        {
            "dir": "trained_supervised/ablation_supervised_multi_fusion",
            "ablation_role": "supervised_multi_fusion",
            "mode": "fusion",
        },
    ]

    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"[{exp['ablation_role'].upper()}] {exp['dir']}")
        print(f"{'='*70}")

        for ratio in DATA_RATIOS:
            weight_path = os.path.join(exp["dir"], f"SSL_{ratio}_weight.h5")
            if not os.path.exists(weight_path):
                print(f"  SSL_{ratio:>3d}: weight not found")
                continue

            tf.keras.backend.clear_session()

            if exp["mode"] == "single":
                input_shape = test_single.shape[1:]  # (18, 18, 20)
                encoder = build_simple_resnet_encoder(input_shape)
                model = build_finetune_simple_model(input_shape, encoder, num_classes=2, training=False)
                model(tf.zeros((1, *input_shape)))
                model.load_weights(weight_path)
                with tf.device("/CPU:0"):
                    preds = model.predict(test_single, verbose=0)
                row = {
                    "experiment": "ablation_supervised_single20_resnet",
                    "ablation_role": exp["ablation_role"],
                    "encoder_type": "simple_resnet",
                    "detailed": False, "transform": True, "debiased": False,
                    "embedding_dim": 0,
                    "data_ratio": ratio,
                    **evaluate(preds, test_label_center),
                }
            else:
                cont_shape = test_cont_scaled.shape[1:]
                cat_shape = test_cat_emb.shape[1:]
                encoder = build_multi_fusion_encoder(cont_shape, cat_shape, layer_name="fusion_encoder")
                model = build_finetune_fusion_model(cont_shape, cat_shape, encoder, num_classes=2, training=False)
                model([tf.zeros((1, *cont_shape)), tf.zeros((1, *cat_shape))])
                model.load_weights(weight_path)
                with tf.device("/CPU:0"):
                    preds = model.predict([test_cont_scaled, test_cat_emb], verbose=0)
                row = {
                    "experiment": "ablation_supervised_multi_fusion",
                    "ablation_role": exp["ablation_role"],
                    "encoder_type": "multi",
                    "detailed": True, "transform": True, "debiased": False,
                    "embedding_dim": 64,
                    "data_ratio": ratio,
                    **evaluate(preds, test_label_center),
                }

            all_results.append(row)
            print(f"  SSL_{ratio:>3d}: acc={row['accuracy']:.4f}  f1={row['f1']:.4f}  auc={row['auc']:.4f}")
            del model

    # Append to existing ablation CSV
    if all_results:
        new_df = pd.DataFrame(all_results)
        csv_path = "./test_results/ablation_results_0311.csv"
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            # Remove old supervised rows if any
            existing = existing[~existing["ablation_role"].isin([
                "supervised_single20_resnet", "supervised_multi_fusion"
            ])]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(csv_path, index=False)
        print(f"\nResults appended to {csv_path}")
        print(f"Total rows: {len(combined)}")

        # Summary
        print(f"\n{'='*70}")
        print("Best Accuracy per ablation role:")
        print("-" * 90)
        for role in combined["ablation_role"].unique():
            sub = combined[combined["ablation_role"] == role]
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  {role:45s}: Acc={best['accuracy']:.4f}  AUC={best['auc']:.4f}  ratio={int(best['data_ratio'])}%")


if __name__ == "__main__":
    main()
