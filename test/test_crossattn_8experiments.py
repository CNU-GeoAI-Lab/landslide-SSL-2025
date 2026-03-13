"""
Test script for 8 cross-attention fusion experiments.
Combinations: detailed/undetailed × transform/no_transform × debiased/non-debiased.

- Caches test embeddings to ./test_cache/ to avoid repeated OpenAI API calls.
- Saves per-experiment CSV + overall summary CSV to ./test_results/.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
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
    build_cross_attention_fusion_encoder,
    build_finetune_fusion_model,
    residual_block,
    PositionalEmbedding,
)
from embeddings.openai_embedding import create_categorical_embeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EXPERIMENTS = [
    {
        "dir_name": "fusion_resnet_crossattn_detailed_transform_0228",
        "detailed": True, "transform": True, "debiased": False,
    },
    {
        "dir_name": "fusion_resnet_crossattn_detailed_transform_db_0228",
        "detailed": True, "transform": True, "debiased": True,
    },
    {
        "dir_name": "fusion_resnet_crossattn_detailed_no_transform_0228",
        "detailed": True, "transform": False, "debiased": False,
    },
    {
        "dir_name": "fusion_resnet_crossattn_detailed_no_transform_db_0228",
        "detailed": True, "transform": False, "debiased": True,
    },
    {
        "dir_name": "fusion_resnet_crossattn_undetailed_transform_0228",
        "detailed": False, "transform": True, "debiased": False,
    },
    {
        "dir_name": "fusion_resnet_crossattn_undetailed_transform_db_0228",
        "detailed": False, "transform": True, "debiased": True,
    },
    {
        "dir_name": "fusion_resnet_crossattn_undetailed_no_transform_0228",
        "detailed": False, "transform": False, "debiased": False,
    },
    {
        "dir_name": "fusion_resnet_crossattn_undetailed_no_transform_db_0228",
        "detailed": False, "transform": False, "debiased": True,
    },
]

DATA_RATIOS = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

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
CATEGORICAL_FEATURE_INDICES = {
    fn: i for i, fn in enumerate(FEATURE_NAMES) if fn in CATEGORICAL_FEATURES
}
CONTINUOUS_FEATURE_INDICES = [
    i for i in range(len(FEATURE_NAMES)) if FEATURE_NAMES[i] not in CATEGORICAL_FEATURES
]

EMBEDDING_DIM = 64
TARGET_SIZE = 28
CROP_SIZE = 18


def extract_center_labels(test_labels, patch_size=6):
    if len(test_labels.shape) == 4:
        center = test_labels[:, patch_size - 1:patch_size + 1, patch_size - 1:patch_size + 1]
        return (np.sum(center, axis=(1, 2)) >= 1.0).astype(np.int32)
    else:
        return (test_labels[:, patch_size, patch_size] >= 1.0).astype(np.int32)


def apply_transforms(patches, apply_log, apply_arcsinh):
    """Apply log/arcsinh transforms in-place on patches copy."""
    patches = patches.copy()
    if apply_log:
        sca_idx = FEATURE_NAMES.index("sca")
        patches[:, :, :, sca_idx] = np.log1p(patches[:, :, :, sca_idx])
        if np.any(np.isnan(patches[:, :, :, sca_idx])):
            patches[:, :, :, sca_idx] = fill_nan_nearest_2d(patches[:, :, :, sca_idx])
    if apply_arcsinh:
        for curv in ["curv_plf", "curv_prf", "curv_std"]:
            idx = FEATURE_NAMES.index(curv)
            patches[:, :, :, idx] = np.arcsinh(patches[:, :, :, idx])
            if np.any(np.isnan(patches[:, :, :, idx])):
                patches[:, :, :, idx] = fill_nan_nearest_2d(patches[:, :, :, idx])
    return patches


def prepare_test_inputs(patches, scaler, client, labels_path, cache_path,
                        apply_log, apply_arcsinh):
    """Prepare continuous + categorical test inputs with disk caching for embeddings."""
    patches = apply_transforms(patches, apply_log, apply_arcsinh)

    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont_features = patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat_features = patches[:, :, :, cat_idx_list]

    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont_features, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat_features, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.int32,
        ))

    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings: {cache_path}")
        cat_embeddings = np.load(cache_path)
    else:
        print(f"  Generating embeddings via API (will cache to {cache_path})...")
        temp = np.zeros((*cat_resized.shape[:3], len(FEATURE_NAMES)))
        for idx, feat in enumerate(CATEGORICAL_FEATURES):
            temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat]] = cat_resized[:, :, :, idx]

        cat_embeddings, _ = create_categorical_embeddings(
            temp, FEATURE_NAMES, CATEGORICAL_FEATURES,
            CATEGORICAL_FEATURE_INDICES, labels_path=labels_path,
            feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
            client=client, model="text-embedding-3-small",
            dimensions=EMBEDDING_DIM, batch_size=100, verbose=True,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, cat_embeddings)
        print(f"  Embeddings cached: {cache_path}")

    cont_scaled = scaler.multi_scale(cont_resized)

    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        e = s + CROP_SIZE
        cont_scaled = cont_scaled[:, s:e, s:e, :]
        cat_embeddings = cat_embeddings[:, s:e, s:e, :]

    return cont_scaled, cat_embeddings


def build_scaler(real_patches, apply_log, apply_arcsinh):
    """Fit scaler on real patches with optional transforms."""
    transformed = apply_transforms(real_patches, apply_log, apply_arcsinh)
    cont = transformed[:, :, :, CONTINUOUS_FEATURE_INDICES]
    with tf.device("/CPU:0"):
        cont = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
    return Multi_data_scaler(cont)


def load_model(weight_path, cont_shape, cat_shape):
    """Build cross-attention model and load weights.
    
    SSL_1 was saved with encoder.trainable=True, so its weight ordering
    lists BN trainable params (gamma, beta) separately from non-trainable
    (moving_mean, moving_variance). SSL_2+ were saved with encoder.trainable=False,
    giving a different weight ordering. We try both orderings.
    """
    c_h, c_w, c_c = map(int, cont_shape)
    k_h, k_w, k_c = map(int, cat_shape)

    # First try: encoder trainable (matches SSL_1 weight ordering)
    tf.keras.backend.clear_session()
    encoder = build_cross_attention_fusion_encoder(
        (c_h, c_w, c_c), (k_h, k_w, k_c), layer_name="fusion_encoder"
    )
    model = build_finetune_fusion_model(
        (c_h, c_w, c_c), (k_h, k_w, k_c), encoder, num_classes=2, training=False
    )
    model([tf.zeros((1, c_h, c_w, c_c)), tf.zeros((1, k_h, k_w, k_c))])
    try:
        model.load_weights(weight_path)
        return model
    except (ValueError, KeyError):
        pass

    # Second try: encoder frozen (matches SSL_2+ weight ordering)
    tf.keras.backend.clear_session()
    encoder = build_cross_attention_fusion_encoder(
        (c_h, c_w, c_c), (k_h, k_w, k_c), layer_name="fusion_encoder"
    )
    encoder.trainable = False
    model = build_finetune_fusion_model(
        (c_h, c_w, c_c), (k_h, k_w, k_c), encoder, num_classes=2, training=False
    )
    model([tf.zeros((1, c_h, c_w, c_c)), tf.zeros((1, k_h, k_w, k_c))])
    model.load_weights(weight_path)
    return model


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
    parser = argparse.ArgumentParser(description="Test 8 cross-attention fusion experiments")
    parser.add_argument("--test_patches", type=str, default="./dataset/1013/test_85patches.npy")
    parser.add_argument("--test_labels", type=str, default="./dataset/1013/test_85labels.npy")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--cache_dir", type=str, default="./test_cache")
    parser.add_argument("--cpu_only", action="store_true", help="Run inference on CPU only")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 70)
    print("Cross-Attention Fusion Model Test (8 Experiments × 14 Data Ratios)")
    print("=" * 70)

    # --- Load test data ---
    print("\n[1/4] Loading test data...")
    test_patches = np.load(args.test_patches)
    test_labels = np.load(args.test_labels)
    test_labels[test_labels == 0.001] = 0.0
    test_label_center = extract_center_labels(test_labels)
    print(f"  Test patches: {test_patches.shape}, Labels: {test_label_center.shape}")
    print(f"  Positive: {np.sum(test_label_center)}, Negative: {len(test_label_center) - np.sum(test_label_center)}")

    # --- Load real patches for scaler ---
    print("\n[2/4] Building scalers...")
    tif_img = np.load("./dataset/tif_img.npy")
    ls_img = np.load("./dataset/ls_img.npy")
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)
    real_patches, _, _, _, _, _ = patch_(
        [554270.0, 562860.0, 859], [331270.0, 342070.0, 1080], tif_img, 6, 3
    )

    scaler_transform = build_scaler(real_patches, apply_log=True, apply_arcsinh=True)
    scaler_no_transform = build_scaler(real_patches, apply_log=False, apply_arcsinh=False)
    print("  Scalers ready (transform + no_transform)")

    # --- Prepare 4 test input variants with caching ---
    print("\n[3/4] Preparing test inputs (4 variants)...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    client = OpenAI(api_key=api_key)

    test_inputs = {}
    variants = [
        ("detailed_transform",    True,  True,  "description/detailed_labels.json",   scaler_transform),
        ("detailed_no_transform", True,  False, "description/detailed_labels.json",   scaler_no_transform),
        ("undetailed_transform",  False, True,  "description/undetailed_labels.json", scaler_transform),
        ("undetailed_no_transform", False, False, "description/undetailed_labels.json", scaler_no_transform),
    ]

    for name, detailed, use_transform, labels_path, scaler in variants:
        cache_file = os.path.join(args.cache_dir, f"test_cat_emb_{name}_{EMBEDDING_DIM}d.npy")
        print(f"\n  Variant: {name}")
        cont, cat = prepare_test_inputs(
            test_patches, scaler, client, labels_path, cache_file,
            apply_log=use_transform, apply_arcsinh=use_transform,
        )
        test_inputs[name] = (cont, cat)
        print(f"    cont: {cont.shape}, cat: {cat.shape}")

    # --- Run tests ---
    print("\n[4/4] Running evaluation...")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for exp in EXPERIMENTS:
        dir_name = exp["dir_name"]
        detailed = exp["detailed"]
        use_transform = exp["transform"]
        debiased = exp["debiased"]

        variant_key = ("detailed" if detailed else "undetailed") + \
                      ("_transform" if use_transform else "_no_transform")
        test_cont, test_cat = test_inputs[variant_key]

        model_dir = os.path.join("./finetuned_models", dir_name)
        if not os.path.isdir(model_dir):
            print(f"\nSkip (dir not found): {model_dir}")
            continue

        print("\n" + "=" * 70)
        print(f"Experiment: {dir_name}")
        print(f"  detailed={detailed}, transform={use_transform}, debiased={debiased}")
        print("=" * 70)

        exp_results = []
        cont_shape = test_cont.shape[1:]
        cat_shape = test_cat.shape[1:]

        for ratio in DATA_RATIOS:
            weight_path = os.path.join(model_dir, f"SSL_{ratio}_weight.h5")
            savedmodel_path = os.path.join(model_dir, f"SSL_{ratio}")

            load_path = None
            if os.path.exists(weight_path):
                load_path = weight_path
            elif os.path.isdir(savedmodel_path):
                load_path = savedmodel_path

            if load_path is None:
                print(f"  SSL_{ratio:>3d}: not found")
                continue

            try:
                model = load_model(load_path, cont_shape, cat_shape)
            except Exception as e:
                print(f"  SSL_{ratio:>3d}: load error - {e}")
                continue

            with tf.device("/CPU:0"):
                preds = model.predict([test_cont, test_cat], verbose=0)

            metrics = evaluate(preds, test_label_center)
            row = {
                "experiment": dir_name,
                "detailed": detailed,
                "transform": use_transform,
                "debiased": debiased,
                "data_ratio": ratio,
                **metrics,
            }
            exp_results.append(row)
            all_results.append(row)

            print(f"  SSL_{ratio:>3d}: "
                  f"acc={metrics['accuracy']:.4f}  "
                  f"prec={metrics['precision']:.4f}  "
                  f"rec={metrics['recall']:.4f}  "
                  f"spec={metrics['specificity']:.4f}  "
                  f"f1={metrics['f1']:.4f}  "
                  f"kappa={metrics['kappa']:.4f}  "
                  f"auc={metrics['auc']:.4f}")

            tf.keras.backend.clear_session()

        if exp_results:
            exp_df = pd.DataFrame(exp_results)
            exp_csv = os.path.join(args.output_dir, f"{dir_name}_results.csv")
            exp_df.to_csv(exp_csv, index=False)
            print(f"\n  -> Saved: {exp_csv}")

    # --- Summary ---
    if not all_results:
        print("\nNo results collected.")
        return

    df_all = pd.DataFrame(all_results)
    summary_csv = os.path.join(args.output_dir, "crossattn_0228_all_results.csv")
    df_all.to_csv(summary_csv, index=False)
    print(f"\n{'=' * 70}")
    print(f"All results saved: {summary_csv}")
    print(f"Total evaluations: {len(all_results)}")

    # Best result per experiment (by AUC)
    print(f"\n{'=' * 70}")
    print("Best AUC per experiment:")
    print(f"{'=' * 70}")
    for exp in EXPERIMENTS:
        d = exp["dir_name"]
        exp_df = df_all[df_all["experiment"] == d]
        if exp_df.empty:
            continue
        best = exp_df.loc[exp_df["auc"].idxmax()]
        print(f"  {d}")
        print(f"    SSL_{int(best['data_ratio']):>3d}: "
              f"acc={best['accuracy']:.4f}  f1={best['f1']:.4f}  auc={best['auc']:.4f}")

    # Summary table: mean metrics across all ratios per experiment
    print(f"\n{'=' * 70}")
    print("Mean metrics per experiment (across all data ratios):")
    print(f"{'=' * 70}")
    metric_cols = ["accuracy", "precision", "recall", "specificity", "f1", "kappa", "auc"]
    summary_mean = df_all.groupby("experiment")[metric_cols].mean()
    print(summary_mean.to_string(float_format="%.4f"))

    summary_mean_csv = os.path.join(args.output_dir, "crossattn_0228_mean_summary.csv")
    summary_mean.to_csv(summary_mean_csv)
    print(f"\nMean summary saved: {summary_mean_csv}")


if __name__ == "__main__":
    main()
