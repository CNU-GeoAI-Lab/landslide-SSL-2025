"""
Ablation study 지도학습 스크립트.
새로 학습해야 하는 2개의 실험:

1) single_resnet: 20개 raw features → 단순 ResNet (임베딩/퓨전 없음)
2) multi_fusion: continuous + detailed_embedding → multi (5-path) fusion encoder

두 모드 모두 SSL 사전학습 없이 지도학습만 수행.
데이터 비율별(1~100%) 모델 학습 및 저장.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
import argparse
import tensorflow as tf
from data.data_reader import patch_
from data.transform import *
from utils.utils import fill_nan_nearest_2d, Multi_data_scaler
from model.simclr_model import (
    build_multi_fusion_encoder,
    build_finetune_fusion_model,
    residual_block,
)
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
DATA_SEED = 118
DATA_RATIOS = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


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


def process_patches_single_input(patches, multi_data_scale_all20, apply_log=True, apply_arcsinh=True):
    """20개 factor를 그대로 사용 (임베딩 없이). MinMax scaling 후 crop."""
    patches = apply_transforms(patches, apply_log, apply_arcsinh)
    cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
    cont_features = patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
    cat_features = patches[:, :, :, cat_idx_list]
    with tf.device("/CPU:0"):
        cont_resized = np.array(tf.image.resize(cont_features, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        cat_resized = np.array(tf.cast(
            tf.image.resize(tf.cast(cat_features, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
            tf.float32,
        ))
    all20_resized = np.zeros((patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
    for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
        all20_resized[:, :, :, idx] = cont_resized[:, :, :, i]
    for i, feat in enumerate(CATEGORICAL_FEATURES):
        idx = CATEGORICAL_FEATURE_INDICES[feat]
        all20_resized[:, :, :, idx] = cat_resized[:, :, :, i]
    scaled = multi_data_scale_all20.multi_scale(all20_resized)
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        scaled = scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
    return scaled.astype(np.float32)


def process_patches_fusion(patches, multi_data_scale_continuous, client, apply_log=True, apply_arcsinh=True,
                           labels_path="description/detailed_labels.json"):
    """Fusion: continuous (MinMax scaled) + categorical (OpenAI embedding)."""
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
    cont_scaled = multi_data_scale_continuous.multi_scale(cont_resized)
    temp = np.zeros((*cat_resized.shape[:3], len(FEATURE_NAMES)))
    for idx, feat in enumerate(CATEGORICAL_FEATURES):
        temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat]] = cat_resized[:, :, :, idx]
    cat_emb, _ = create_categorical_embeddings(
        temp, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
        labels_path=labels_path, feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
        client=client, model="text-embedding-3-small", dimensions=EMBEDDING_DIM,
        batch_size=100, verbose=True,
    )
    if CROP_SIZE < TARGET_SIZE:
        s = (TARGET_SIZE - CROP_SIZE) // 2
        cont_scaled = cont_scaled[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
        cat_emb = cat_emb[:, s:s + CROP_SIZE, s:s + CROP_SIZE, :]
    return cont_scaled.astype(np.float32), cat_emb.astype(np.float32)


# ============================================================
# 단순 ResNet 모델 (20 raw features, 퓨전 없음)
# ============================================================
def build_simple_resnet_encoder(input_shape, layer_name="simple_resnet"):
    """
    단순 ResNet encoder: 20채널 입력 → 4단계 residual block → GAP → Dense(128).
    퓨전 모듈 없이 단일 스트림 ResNet.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = residual_block(inputs, 64, stride=1, name=f"{layer_name}_stage1")
    x = residual_block(x, 64, stride=1, name=f"{layer_name}_stage2")
    x = residual_block(x, 128, stride=2, name=f"{layer_name}_stage3")
    x = residual_block(x, 128, stride=1, name=f"{layer_name}_stage4")
    x = tf.keras.layers.GlobalAveragePooling2D(name=f"{layer_name}_gap")(x)
    x = tf.keras.layers.Dense(128, name=f"{layer_name}_dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=layer_name)


def build_finetune_simple_model(input_shape, encoder, num_classes=2, training=True):
    """단순 ResNet encoder 위에 분류 head 추가."""
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs, training=training)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ============================================================
# 데이터 준비
# ============================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    args_dict = {}
    args_dict["gpu_device"] = config.get("device", {}).get("gpu_device", 0)
    args_dict["tif_img_path"] = config.get("data", {}).get("tif_img_path", "./dataset/tif_img.npy")
    args_dict["ls_img_path"] = config.get("data", {}).get("ls_img_path", "./dataset/ls_img.npy")
    args_dict["height"] = config.get("data", {}).get("height", [554270.0, 562860.0, 859])
    args_dict["width"] = config.get("data", {}).get("width", [331270.0, 342070.0, 1080])
    args_dict["non_ls_patches_path"] = config.get("data", {}).get("non_ls_patches_features_path", "./dataset/0926/non_ls_patches_60dist.npy")
    args_dict["non_ls_labels_path"] = config.get("data", {}).get("non_ls_patches_labels_path", "./dataset/0926/non_ls_labels_60dist.npy")
    args_dict["ls_patches_path"] = config.get("data", {}).get("ls_patches_features_path", "./dataset/0926/ls_patches_60dist.npy")
    args_dict["ls_labels_path"] = config.get("data", {}).get("ls_patches_labels_path", "./dataset/0926/ls_labels_60dist.npy")
    args_dict["apply_arcsinh"] = config.get("data", {}).get("apply_arcsinh", True)
    args_dict["apply_log"] = config.get("data", {}).get("apply_log", True)
    args_dict["patch_size"] = config.get("model", {}).get("patch_size", 6)
    args_dict["strides"] = config.get("model", {}).get("strides", 3)
    args_dict["learning_rate"] = config.get("train", {}).get("learning_rate", 0.0001)
    args_dict["batch_size"] = config.get("train", {}).get("batch_size", 32)
    args_dict["dir_name"] = config.get("model", {}).get("dir_name", "ablation_experiment")
    return argparse.Namespace(**args_dict)


def prepare_train_valid(args, seed=DATA_SEED):
    """main_embedding_fusion.py와 동일한 seed·로직으로 train/valid 분리."""
    np.random.seed(seed)
    non_ls_patches_features = np.load(args.non_ls_patches_path)
    non_ls_patches_labels = np.load(args.non_ls_labels_path)
    ls_patches_features = np.load(args.ls_patches_path)
    ls_patches_labels = np.load(args.ls_labels_path)

    valid_non_ls_ind = np.random.choice(
        non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0] * 0.1), replace=False
    )
    valid_ls_ind = np.random.choice(
        ls_patches_features.shape[0], int(ls_patches_labels.shape[0] * 0.1), replace=False
    )

    valid_non_ls = non_ls_patches_features[valid_non_ls_ind]
    valid_non_ls_label = non_ls_patches_labels[valid_non_ls_ind]
    valid_ls = ls_patches_features[valid_ls_ind]
    valid_ls_label = ls_patches_labels[valid_ls_ind]
    valid_patches = np.concatenate([valid_ls, valid_non_ls])
    valid_labels = np.concatenate([valid_ls_label, valid_non_ls_label])

    non_ls_patches_features = np.delete(non_ls_patches_features, valid_non_ls_ind, axis=0)
    non_ls_patches_labels = np.delete(non_ls_patches_labels, valid_non_ls_ind, axis=0)
    ls_patches_features = np.delete(ls_patches_features, valid_ls_ind, axis=0)
    ls_patches_labels = np.delete(ls_patches_labels, valid_ls_ind, axis=0)

    train_non_ls_ind_full = np.random.choice(
        non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]), replace=False
    )
    train_ls_ind_full = np.random.choice(
        ls_patches_features.shape[0], int(ls_patches_features.shape[0]), replace=False
    )

    patch_size = args.patch_size

    def center_labels(labels):
        out = []
        for i in range(labels.shape[0]):
            if np.sum(labels[i, patch_size - 1: patch_size + 1, patch_size - 1: patch_size + 1]) >= 1.0:
                out.append(1)
            else:
                out.append(0)
        return np.expand_dims(np.array(out, dtype=np.int32), axis=1)

    valid_labels_ = center_labels(valid_labels)
    return (
        valid_patches, valid_labels_,
        non_ls_patches_features, non_ls_patches_labels,
        ls_patches_features, ls_patches_labels,
        train_non_ls_ind_full, train_ls_ind_full,
        center_labels,
    )


def get_train_for_ratio(train_non_ls, train_non_ls_label, train_ls, train_ls_label,
                        train_non_ls_ind_full, train_ls_ind_full, ratio_pct, center_labels_fn):
    r = ratio_pct / 100.0
    n_non_ls = max(1, int(len(train_non_ls_ind_full) * r))
    n_ls = max(1, int(len(train_ls_ind_full) * r))
    ind_n = train_non_ls_ind_full[:n_non_ls]
    ind_l = train_ls_ind_full[:n_ls]
    train_patches = np.concatenate([train_ls[ind_l], train_non_ls[ind_n]])
    train_labels = np.concatenate([train_ls_label[ind_l], train_non_ls_label[ind_n]])
    train_labels_ = center_labels_fn(train_labels)
    return train_patches, train_labels_, len(ind_l), len(ind_n)


# ============================================================
# 학습 함수
# ============================================================
def train_single_resnet(args, train_input, train_labels_, valid_input, valid_labels_, ratio, save_dir):
    """단순 ResNet으로 지도학습 (20 raw features, 단일 입력)."""
    input_shape = train_input.shape[1:]  # (18, 18, 20)
    tf.keras.backend.clear_session()
    encoder = build_simple_resnet_encoder(input_shape, layer_name="simple_resnet")
    model = build_finetune_simple_model(input_shape, encoder, num_classes=2, training=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"SSL_{ratio}_weight.h5")
    if os.path.exists(weight_path):
        os.remove(weight_path)

    model.fit(
        train_input, train_labels_,
        validation_data=(valid_input, valid_labels_),
        epochs=200,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                weight_path, save_weights_only=True, save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10),
        ],
        verbose=1,
    )
    model.load_weights(weight_path)
    model.save_weights(weight_path)
    print(f"  Single ResNet saved: {weight_path}")


def train_multi_fusion(args, train_cont, train_cat, train_labels_, valid_cont, valid_cat, valid_labels_, ratio, save_dir):
    """Multi (5-path) fusion encoder로 지도학습 (continuous + embedding 입력)."""
    cont_shape = train_cont.shape[1:]
    cat_shape = train_cat.shape[1:]
    tf.keras.backend.clear_session()
    encoder = build_multi_fusion_encoder(
        cont_shape, cat_shape, layer_name="fusion_encoder"
    )
    model = build_finetune_fusion_model(
        cont_shape, cat_shape, encoder, num_classes=2, training=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, f"SSL_{ratio}_weight.h5")
    if os.path.exists(weight_path):
        os.remove(weight_path)

    model.fit(
        [train_cont, train_cat], train_labels_,
        validation_data=([valid_cont, valid_cat], valid_labels_),
        epochs=200,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                weight_path, save_weights_only=True, save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10),
        ],
        verbose=1,
    )
    model.load_weights(weight_path)
    model.save_weights(weight_path)
    print(f"  Multi fusion saved: {weight_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ablation supervised training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["single_resnet", "multi_fusion"],
                        help="single_resnet: 20 raw features + plain ResNet, "
                             "multi_fusion: embedding + multi (5-path) fusion encoder")
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated ratios (e.g. 1,10,100). Default: all 14 ratios")
    cmd = parser.parse_args()

    args = load_config(cmd.config)
    if cmd.gpu_device is not None:
        args.gpu_device = cmd.gpu_device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    if tf.config.experimental.list_physical_devices("GPU"):
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    ratios = DATA_RATIOS
    if cmd.ratios is not None:
        ratios = [int(x) for x in cmd.ratios.split(",")]

    print("=" * 70)
    print(f"Ablation Supervised Training: {cmd.mode}")
    print(f"Config: {cmd.config}")
    print(f"GPU: {args.gpu_device}, Ratios: {ratios}")
    print("=" * 70)

    # Prepare data
    (
        valid_patches, valid_labels_,
        train_non_ls, train_non_ls_label,
        train_ls, train_ls_label,
        train_non_ls_ind_full, train_ls_ind_full,
        center_labels_fn,
    ) = prepare_train_valid(args)

    n_ls_full = len(train_ls_ind_full)
    n_non_ls_full = len(train_non_ls_ind_full)
    print(f"Train LS: {n_ls_full}, Train Non-LS: {n_non_ls_full}, Valid: {len(valid_patches)}")

    # Load real_patches for scaler fitting
    tif_img = np.load(args.tif_img_path)
    ls_img = np.load(args.ls_img_path)
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)
    real_patches, _, _, _, _, _ = patch_(
        args.height, args.width, tif_img, args.patch_size, args.strides
    )
    real_patches = apply_transforms(real_patches, args.apply_log, args.apply_arcsinh)

    save_dir = os.path.join("trained_supervised", args.dir_name)

    if cmd.mode == "single_resnet":
        # Scaler for all 20 factors
        cat_idx_list = [CATEGORICAL_FEATURE_INDICES[f] for f in CATEGORICAL_FEATURES]
        cont_real = real_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
        cat_real = real_patches[:, :, :, cat_idx_list]
        with tf.device("/CPU:0"):
            cont_resized = np.array(tf.image.resize(cont_real, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
            cat_resized = np.array(tf.cast(
                tf.image.resize(tf.cast(cat_real, tf.float32), (TARGET_SIZE, TARGET_SIZE), method="nearest"),
                tf.float32,
            ))
        real_all20 = np.zeros((real_patches.shape[0], TARGET_SIZE, TARGET_SIZE, 20), dtype=np.float32)
        for i, idx in enumerate(CONTINUOUS_FEATURE_INDICES):
            real_all20[:, :, :, idx] = cont_resized[:, :, :, i]
        for i, feat in enumerate(CATEGORICAL_FEATURES):
            idx = CATEGORICAL_FEATURE_INDICES[feat]
            real_all20[:, :, :, idx] = cat_resized[:, :, :, i]
        multi_data_scale_all20 = Multi_data_scaler(real_all20)

        # Process full train and valid once
        full_train_patches = np.concatenate([
            train_ls[train_ls_ind_full], train_non_ls[train_non_ls_ind_full]
        ])
        full_train_labels = np.concatenate([
            train_ls_label[train_ls_ind_full], train_non_ls_label[train_non_ls_ind_full]
        ])
        full_train_labels_ = center_labels_fn(full_train_labels)

        print("Processing full train (single 20ch)...")
        full_train_single = process_patches_single_input(
            full_train_patches, multi_data_scale_all20,
            apply_log=args.apply_log, apply_arcsinh=args.apply_arcsinh,
        )
        print("Processing valid (single 20ch)...")
        valid_single = process_patches_single_input(
            valid_patches, multi_data_scale_all20,
            apply_log=args.apply_log, apply_arcsinh=args.apply_arcsinh,
        )
        print(f"  Train shape: {full_train_single.shape}, Valid shape: {valid_single.shape}")

        for ratio in ratios:
            n_ls = max(1, int(n_ls_full * ratio / 100))
            n_n = max(1, int(n_non_ls_full * ratio / 100))
            train_input = np.concatenate([full_train_single[:n_ls], full_train_single[n_ls_full:n_ls_full + n_n]])
            train_labels_r = np.concatenate([full_train_labels_[:n_ls], full_train_labels_[n_ls_full:n_ls_full + n_n]])
            print(f"\n{'='*70}")
            print(f"Single ResNet - ratio {ratio}% (train n={len(train_labels_r)})")
            print(f"{'='*70}")
            train_single_resnet(args, train_input, train_labels_r, valid_single, valid_labels_, ratio, save_dir)

    elif cmd.mode == "multi_fusion":
        # Scaler for continuous factors only
        cont_real = real_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
        with tf.device("/CPU:0"):
            cont_real_resized = np.array(tf.image.resize(cont_real, (TARGET_SIZE, TARGET_SIZE), method="bilinear"))
        multi_data_scale_continuous = Multi_data_scaler(cont_real_resized)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for multi_fusion mode")
        client = OpenAI(api_key=api_key)

        full_train_patches = np.concatenate([
            train_ls[train_ls_ind_full], train_non_ls[train_non_ls_ind_full]
        ])
        full_train_labels = np.concatenate([
            train_ls_label[train_ls_ind_full], train_non_ls_label[train_non_ls_ind_full]
        ])
        full_train_labels_ = center_labels_fn(full_train_labels)

        print("Processing full train (fusion: cont + embedding)...")
        full_train_cont, full_train_cat = process_patches_fusion(
            full_train_patches, multi_data_scale_continuous, client,
            apply_log=args.apply_log, apply_arcsinh=args.apply_arcsinh,
        )
        print("Processing valid (fusion: cont + embedding)...")
        valid_cont, valid_cat = process_patches_fusion(
            valid_patches, multi_data_scale_continuous, client,
            apply_log=args.apply_log, apply_arcsinh=args.apply_arcsinh,
        )
        print(f"  Train cont: {full_train_cont.shape}, cat: {full_train_cat.shape}")

        for ratio in ratios:
            n_ls = max(1, int(n_ls_full * ratio / 100))
            n_n = max(1, int(n_non_ls_full * ratio / 100))
            train_cont = np.concatenate([full_train_cont[:n_ls], full_train_cont[n_ls_full:n_ls_full + n_n]])
            train_cat = np.concatenate([full_train_cat[:n_ls], full_train_cat[n_ls_full:n_ls_full + n_n]])
            train_labels_r = np.concatenate([full_train_labels_[:n_ls], full_train_labels_[n_ls_full:n_ls_full + n_n]])
            print(f"\n{'='*70}")
            print(f"Multi Fusion - ratio {ratio}% (train n={len(train_labels_r)})")
            print(f"{'='*70}")
            train_multi_fusion(args, train_cont, train_cat, train_labels_r,
                               valid_cont, valid_cat, valid_labels_, ratio, save_dir)

    print(f"\n{'='*70}")
    print(f"All training complete. Models saved to: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
