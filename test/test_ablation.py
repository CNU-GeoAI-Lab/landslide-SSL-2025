"""
Ablation study test script.
Tests ablation models against the proposed best model to verify encoder compatibility
and evaluate performance. Results saved to ablation_results_0311.csv in an extensible format.

Proposed model: fusion_resnet_transform_db_detailed_dim64_0122 (multi 5-path encoder)

Ablation variables:
  - no_transform: remove data transformation (arcsinh/log)
  - undetailed: use undetailed categorical embedding
  - no_debiased: remove debiased contrastive loss
  - dim32: reduce embedding dimension from 64 to 32
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
    PositionalEmbedding,
)
from embeddings.openai_embedding import create_categorical_embeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ablation experiment definitions
# Each entry: dir_name, ablation_role, detailed, transform, debiased, embedding_dim
ABLATION_EXPERIMENTS = [
    {
        "dir_name": "fusion_resnet_transform_db_detailed_dim64_0122",
        "ablation_role": "proposed",
        "detailed": True, "transform": True, "debiased": True, "embedding_dim": 64,
    },
    {
        "dir_name": "fusion_resnet_detailed_no_transform_db_0205",
        "ablation_role": "ablation_no_transform",
        "detailed": True, "transform": False, "debiased": True, "embedding_dim": 64,
    },
    {
        "dir_name": "fusion_resnet_undetailed_transform_db_0205",
        "ablation_role": "ablation_undetailed",
        "detailed": False, "transform": True, "debiased": True, "embedding_dim": 64,
    },
    {
        "dir_name": "fusion_resnet_transform_detailed_dim64_0122",
        "ablation_role": "ablation_no_debiased",
        "detailed": True, "transform": True, "debiased": False, "embedding_dim": 64,
    },
    {
        "dir_name": "fusion_resnet_transform_db_detailed_dim32_0127",
        "ablation_role": "ablation_dim32",
        "detailed": True, "transform": True, "debiased": True, "embedding_dim": 32,
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

TARGET_SIZE = 28
CROP_SIZE = 18


def extract_center_labels(test_labels, patch_size=6):
    if len(test_labels.shape) == 4:
        center = test_labels[:, patch_size - 1:patch_size + 1, patch_size - 1:patch_size + 1]
        return (np.sum(center, axis=(1, 2)) >= 1.0).astype(np.int32)
    else:
        return (test_labels[:, patch_size, patch_size] >= 1.0).astype(np.int32)


def apply_transforms(patches, apply_log, apply_arcsinh):
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
                        apply_log, apply_arcsinh, embedding_dim):
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
        print(f"    Loading cached embeddings: {cache_path}")
        cat_embeddings = np.load(cache_path)
    else:
        print(f"    Generating embeddings (dim={embedding_dim}), caching to {cache_path}...")
        temp = np.zeros((*cat_resized.shape[:3], len(FEATURE_NAMES)))
        for idx, feat in enumerate(CATEGORICAL_FEATURES):
            temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat]] = cat_resized[:, :, :, idx]

        cat_embeddings, _ = create_categorical_embeddings(
            temp, FEATURE_NAMES, CATEGORICAL_FEATURES,
            CATEGORICAL_FEATURE_INDICES, labels_path=labels_path,
            feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
            client=client, model="text-embedding-3-small",
            dimensions=embedding_dim, batch_size=100, verbose=True,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, cat_embeddings)
        print(f"    Embeddings cached: {cache_path}")

    cont_scaled = scaler.multi_scale(cont_resized)

    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        e = s + CROP_SIZE
        cont_scaled = cont_scaled[:, s:e, s:e, :]
        cat_embeddings = cat_embeddings[:, s:e, s:e, :]

    return cont_scaled, cat_embeddings


def build_scaler(real_patches, apply_log, apply_arcsinh):
    transformed = apply_transforms(real_patches, apply_log, apply_arcsinh)
    cont = transformed[:, :, :, CONTINUOUS_FEATURE_INDICES]
    with tf.device("/CPU:0"):
        cont = np.array(tf.image.resize(cont, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
    return Multi_data_scaler(cont)


def try_load_model(weight_path, cont_shape, cat_shape):
    """Try to build multi encoder and load weights."""
    c_h, c_w, c_c = map(int, cont_shape)
    k_h, k_w, k_c = map(int, cat_shape)

    tf.keras.backend.clear_session()
    encoder = build_multi_fusion_encoder(
        (c_h, c_w, c_c), (k_h, k_w, k_c), layer_name="fusion_encoder"
    )
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 70)
    print("Ablation Study Test")
    print("=" * 70)

    # Load test data
    print("\n[1/4] Loading test data...")
    test_patches = np.load("./dataset/1013/test_85patches.npy")
    test_labels = np.load("./dataset/1013/test_85labels.npy")
    test_labels[test_labels == 0.001] = 0.0
    test_label_center = extract_center_labels(test_labels)
    print(f"  Test patches: {test_patches.shape}, Labels: {test_label_center.shape}")
    print(f"  Positive: {np.sum(test_label_center)}, Negative: {len(test_label_center) - np.sum(test_label_center)}")

    # Build scalers
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
    print("  Scalers ready")

    # Prepare test inputs for each unique (detailed, transform, embedding_dim) combination
    print("\n[3/4] Preparing test inputs...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    cache_dir = "./test_cache"
    test_inputs_cache = {}

    def get_test_inputs(detailed, use_transform, embedding_dim):
        key = (detailed, use_transform, embedding_dim)
        if key in test_inputs_cache:
            return test_inputs_cache[key]

        detail_str = "detailed" if detailed else "undetailed"
        transform_str = "transform" if use_transform else "no_transform"
        labels_path = "description/detailed_labels.json" if detailed else "description/undetailed_labels.json"
        scaler = scaler_transform if use_transform else scaler_no_transform
        cache_file = os.path.join(cache_dir, f"test_cat_emb_{detail_str}_{transform_str}_{embedding_dim}d.npy")

        print(f"\n  Preparing: {detail_str}_{transform_str}_dim{embedding_dim}")
        cont, cat = prepare_test_inputs(
            test_patches, scaler, client, labels_path, cache_file,
            apply_log=use_transform, apply_arcsinh=use_transform,
            embedding_dim=embedding_dim,
        )
        print(f"    cont: {cont.shape}, cat: {cat.shape}")
        test_inputs_cache[key] = (cont, cat)
        return cont, cat

    # Run ablation tests
    print("\n[4/4] Running ablation tests...")
    os.makedirs("./test_results", exist_ok=True)
    all_results = []

    for exp in ABLATION_EXPERIMENTS:
        dir_name = exp["dir_name"]
        ablation_role = exp["ablation_role"]
        detailed = exp["detailed"]
        use_transform = exp["transform"]
        debiased = exp["debiased"]
        embedding_dim = exp["embedding_dim"]

        model_dir = os.path.join("./finetuned_models", dir_name)
        if not os.path.isdir(model_dir):
            print(f"\n[SKIP] {dir_name} - directory not found")
            continue

        # Get test inputs for this configuration
        test_cont, test_cat = get_test_inputs(detailed, use_transform, embedding_dim)
        cont_shape = test_cont.shape[1:]
        cat_shape = test_cat.shape[1:]

        # Verify compatibility with multi encoder using first available weight
        test_weight = None
        for r in [100, 10, 1]:
            wp = os.path.join(model_dir, f"SSL_{r}_weight.h5")
            if os.path.exists(wp):
                test_weight = wp
                break

        if test_weight is None:
            print(f"\n[SKIP] {dir_name} - no weight files")
            continue

        try:
            tf.keras.backend.clear_session()
            model = try_load_model(test_weight, cont_shape, cat_shape)
            del model
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"\n[INCOMPATIBLE] {dir_name} - multi encoder load failed: {e}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{ablation_role.upper()}] {dir_name}")
        print(f"  detailed={detailed}, transform={use_transform}, debiased={debiased}, dim={embedding_dim}")
        print("=" * 70)

        # Evaluate all data ratios
        for ratio in DATA_RATIOS:
            weight_path = os.path.join(model_dir, f"SSL_{ratio}_weight.h5")
            if not os.path.exists(weight_path):
                continue

            try:
                tf.keras.backend.clear_session()
                model = try_load_model(weight_path, cont_shape, cat_shape)
            except Exception as e:
                print(f"  SSL_{ratio:>3d}: load error - {e}")
                continue

            with tf.device("/CPU:0"):
                preds = model.predict([test_cont, test_cat], verbose=0)

            metrics = evaluate(preds, test_label_center)
            row = {
                "experiment": dir_name,
                "ablation_role": ablation_role,
                "encoder_type": "multi",
                "detailed": detailed,
                "transform": use_transform,
                "debiased": debiased,
                "embedding_dim": embedding_dim,
                "data_ratio": ratio,
                **metrics,
            }
            all_results.append(row)

            print(f"  SSL_{ratio:>3d}: "
                  f"acc={metrics['accuracy']:.4f}  "
                  f"f1={metrics['f1']:.4f}  "
                  f"auc={metrics['auc']:.4f}")

            del model
            tf.keras.backend.clear_session()

    # Save results
    out_path = "./test_results/ablation_results_0311.csv"
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"Results saved: {out_path}")
        print(f"Total evaluations: {len(all_results)}")

        # Print summary: best AUC and best Accuracy per ablation role
        print(f"\n{'=' * 70}")
        print("Best AUC per ablation role:")
        print("-" * 70)
        for role in df["ablation_role"].unique():
            sub = df[df["ablation_role"] == role]
            best = sub.loc[sub["auc"].idxmax()]
            print(f"  {role:25s}: AUC={best['auc']:.4f}  Acc={best['accuracy']:.4f}  "
                  f"F1={best['f1']:.4f}  ratio={int(best['data_ratio'])}%")

        print(f"\nBest Accuracy per ablation role:")
        print("-" * 70)
        for role in df["ablation_role"].unique():
            sub = df[df["ablation_role"] == role]
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  {role:25s}: Acc={best['accuracy']:.4f}  AUC={best['auc']:.4f}  "
                  f"F1={best['f1']:.4f}  ratio={int(best['data_ratio'])}%")
    else:
        print("\nNo results to save.")

    print(f"\n{'=' * 70}")
    print("Ablation test complete.")
    print(f"To append future results, load {out_path} and pd.concat new rows.")


if __name__ == "__main__":
    main()
