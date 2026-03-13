import os
import numpy as np
import yaml
import argparse
import tensorflow as tf
from data.data_reader import *
from data.augmentation import *
from data.transform import *
from loss.loss import *
from train.train import *
from utils.utils import *
from model.simclr_model import *
from model.simclr_model import build_fusion_vit_encoder
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score
from embeddings.openai_embedding import create_categorical_embeddings
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

# Register custom objects for SavedModel loading
CUSTOM_OBJECTS = {
    "PositionalEmbedding": PositionalEmbedding,
    "residual_block": residual_block,
}
get_custom_objects().update(CUSTOM_OBJECTS)

# Feature definitions
FEATURE_NAMES = [
    'aspect', 'curv_plf', 'curv_prf', 'curv_std', 'elev',
    'forest_age', 'forest_diameter', 'forest_density', 'forest_type',
    'geology', 'landuse', 'sca', 'slope',
    'soil_drainage', 'soil_series', 'soil_sub_texture', 'soil_thickness', 'soil_texture',
    'spi', 'twi',
]
FEATURE_TO_LABEL_MAPPING = {
    'geology': ('Geology', 'geology'),
    'landuse': ('Land_use', 'landuse'),
    'soil_drainage': ('Soil_summary', 'soil_drainage'),
    'soil_series': ('Soil_summary', 'soil_series'),
    'soil_texture': ('Soil_summary', 'soil_texture'),
    'soil_thickness': ('Soil_summary', 'soil_thickness'),
    'soil_sub_texture': ('Soil_summary', 'soil_sub_texture'),
    'forest_age': ('Forest_summary', 'forest_age'),
    'forest_diameter': ('Forest_summary', 'forest_diameter'),
    'forest_density': ('Forest_summary', 'forest_density'),
    'forest_type': ('Forest_summary', 'forest_type'),
}
CATEGORICAL_FEATURES = list(FEATURE_TO_LABEL_MAPPING.keys())
CATEGORICAL_FEATURE_INDICES = {fn: i for i, fn in enumerate(FEATURE_NAMES) if fn in CATEGORICAL_FEATURES}
CONTINUOUS_FEATURE_INDICES = [i for i in range(len(FEATURE_NAMES)) if FEATURE_NAMES[i] not in CATEGORICAL_FEATURES]


class SafeModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ModelCheckpoint that deletes existing file/directory before saving to avoid HDF5 dataset name conflicts"""
    def _save_model(self, epoch, logs):
        if self.filepath:
            if os.path.exists(self.filepath):
                try:
                    if os.path.isdir(self.filepath):
                        import shutil
                        shutil.rmtree(self.filepath)
                    else:
                        os.remove(self.filepath)
                except Exception as e:
                    print(f"Warning: Could not remove {self.filepath}: {e}")
        super()._save_model(epoch, logs)


def process_patches_with_embedding(patches, feature_names, categorical_features, categorical_feature_indices,
                                   feature_to_label_mapping, multi_data_scale, client, embedding_dimension,
                                   apply_arcsinh=True, apply_log=True, target_size=28, crop_size=18,
                                   return_separate=False, embedding_file=None, labels_path=None):
    """
    Process patches with transformation, embedding, and scaling.

    Parameters:
    - return_separate: if True, return (continuous, categorical) separately for fusion model
    - embedding_file: path to .npy embedding file
    - labels_path: JSON file for label mapping

    Returns:
    - If return_separate=False: processed_patches (concatenated)
    - If return_separate=True: (continuous_features_scaled, categorical_embeddings)
    """
    patches = patches.copy()

    if apply_log:
        if 'sca' in feature_names:
            feat_idx = feature_names.index('sca')
            patches[:, :, :, feat_idx] = np.log1p(patches[:, :, :, feat_idx])
            if np.any(np.isnan(patches[:, :, :, feat_idx])):
                patches[:, :, :, feat_idx] = fill_nan_nearest_2d(patches[:, :, :, feat_idx])

    if apply_arcsinh:
        for curv_name in ['curv_plf', 'curv_prf', 'curv_std']:
            if curv_name in feature_names:
                feat_idx = feature_names.index(curv_name)
                patches[:, :, :, feat_idx] = np.arcsinh(patches[:, :, :, feat_idx])
                if np.any(np.isnan(patches[:, :, :, feat_idx])):
                    patches[:, :, :, feat_idx] = fill_nan_nearest_2d(patches[:, :, :, feat_idx])

    continuous_feature_indices = [i for i in range(len(feature_names)) if feature_names[i] not in categorical_features]
    categorical_feature_indices_list = [categorical_feature_indices[feat] for feat in categorical_features]

    continuous_features = patches[:, :, :, continuous_feature_indices]
    categorical_features_data = patches[:, :, :, categorical_feature_indices_list]

    with tf.device('/CPU:0'):
        continuous_features_resized = tf.image.resize(continuous_features, (target_size, target_size), method='bilinear')
        categorical_features_resized = tf.image.resize(
            tf.cast(categorical_features_data, tf.float32),
            (target_size, target_size), method='nearest'
        )
        categorical_features_resized = tf.cast(categorical_features_resized, tf.int32)

    continuous_features_resized = np.array(continuous_features_resized)
    categorical_features_resized = np.array(categorical_features_resized)

    if embedding_file is None:
        embedding_file = f"./embeddings/detailed_embedding{embedding_dimension}.npy"
    if labels_path is None:
        labels_path = 'description/detailed_labels.json'

    if os.path.exists(embedding_file):
        full_embeddings = np.load(embedding_file)
        print(f"Loaded embeddings from {embedding_file}, shape: {full_embeddings.shape}")

    temp_data_for_embedding = np.zeros((categorical_features_resized.shape[0],
                                       categorical_features_resized.shape[1],
                                       categorical_features_resized.shape[2],
                                       len(feature_names)))
    for idx, feat_name in enumerate(categorical_features):
        orig_idx = categorical_feature_indices[feat_name]
        temp_data_for_embedding[:, :, :, orig_idx] = categorical_features_resized[:, :, :, idx]

    categorical_embeddings, _ = create_categorical_embeddings(
        temp_data_for_embedding, feature_names, categorical_features,
        categorical_feature_indices, labels_path=labels_path,
        feature_to_label_mapping=feature_to_label_mapping,
        client=client, model="text-embedding-3-small",
        dimensions=embedding_dimension, batch_size=100, verbose=False
    )

    continuous_features_scaled = multi_data_scale.multi_scale(continuous_features_resized)

    if crop_size < target_size:
        crop_start = (target_size - crop_size) // 2
        crop_end = crop_start + crop_size
        continuous_features_scaled = continuous_features_scaled[:, crop_start:crop_end, crop_start:crop_end, :]
        categorical_embeddings = categorical_embeddings[:, crop_start:crop_end, crop_start:crop_end, :]

    if return_separate:
        return continuous_features_scaled, categorical_embeddings
    else:
        final_feature_dim = len(continuous_feature_indices) + categorical_embeddings.shape[3]
        processed_patches = np.zeros((continuous_features_scaled.shape[0],
                                     continuous_features_scaled.shape[1],
                                     continuous_features_scaled.shape[2],
                                     final_feature_dim))
        for idx, orig_idx in enumerate(continuous_feature_indices):
            processed_patches[:, :, :, idx] = continuous_features_scaled[:, :, :, idx]
        embedding_start_idx = len(continuous_feature_indices)
        processed_patches[:, :, :, embedding_start_idx:] = categorical_embeddings
        return processed_patches


def load_config(config_path='configs/default.yaml'):
    """Load configuration from YAML file and flatten it to argparse-like namespace"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    args_dict = {}
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('train', {})
    aug_config = config.get('aug', {})

    # Auto-detect mode from config structure
    if 'mode' in config:
        args_dict['mode'] = config['mode']
    elif 'fusion_encoder_type' in model_config:
        args_dict['mode'] = 'fusion'
    elif 'embedding_dimension' in data_config:
        args_dict['mode'] = 'embedding'
    else:
        args_dict['mode'] = 'plain'

    # Device
    args_dict['gpu_device'] = config.get('device', {}).get('gpu_device', 0)

    # Data
    args_dict['tif_img_path'] = data_config.get('tif_img_path', './dataset/tif_img.npy')
    args_dict['ls_img_path'] = data_config.get('ls_img_path', './dataset/ls_img.npy')
    args_dict['height'] = data_config.get('height', [554270.0, 562860.0, 859])
    args_dict['width'] = data_config.get('width', [331270.0, 342070.0, 1080])
    args_dict['non_ls_patches_path'] = data_config.get('non_ls_patches_features_path', './dataset/0926/non_ls_patches_60dist.npy')
    args_dict['non_ls_labels_path'] = data_config.get('non_ls_patches_labels_path', './dataset/0926/non_ls_labels_60dist.npy')
    args_dict['ls_patches_path'] = data_config.get('ls_patches_features_path', './dataset/0926/ls_patches_60dist.npy')
    args_dict['ls_labels_path'] = data_config.get('ls_patches_labels_path', './dataset/0926/ls_labels_60dist.npy')
    args_dict['apply_arcsinh'] = data_config.get('apply_arcsinh', False)
    args_dict['apply_log'] = data_config.get('apply_log', False)
    args_dict['embedding_dimension'] = data_config.get('embedding_dimension', 64)
    args_dict['detailed'] = data_config.get('detailed', True)

    # Model
    args_dict['model_type'] = model_config.get('model_type', 'ResNet')
    args_dict['ssl_type'] = model_config.get('ssl_type', 'SimCLR')
    args_dict['pre_trained_model_name'] = model_config.get('pre_trained_model_name', 'pre')
    args_dict['fine_trained_model_name'] = model_config.get('fine_trained_model_name', 'fine')
    args_dict['dir_name'] = model_config.get('dir_name', '250228')
    args_dict['patch_size'] = model_config.get('patch_size', 6)
    args_dict['strides'] = model_config.get('strides', 3)
    args_dict['temperature'] = model_config.get('temperature', 0.05)
    args_dict['tau_plus'] = model_config.get('tau_plus', 0.10)
    args_dict['debiased'] = model_config.get('debiased', False)
    args_dict['fusion_encoder_type'] = model_config.get('fusion_encoder_type', 'multi')
    args_dict['verify_saved_model'] = model_config.get('verify_saved_model', False)

    # Train
    args_dict['pre_batch_size'] = train_config.get('pre_batch_size', 256)
    args_dict['batch_size'] = train_config.get('batch_size', 32)
    args_dict['pre_epochs'] = train_config.get('pre_epochs', 200)
    args_dict['pre_learning_rate'] = train_config.get('pre_learning_rate', 0.0001)
    args_dict['learning_rate'] = train_config.get('learning_rate', 0.00001)
    args_dict['fine_tuning_areas'] = train_config.get('fine_tuning_areas', 10)
    args_dict['fine_tuning_data_ratio'] = train_config.get('fine_tuning_data_ratio', 1.0)
    args_dict['fixed_valid'] = train_config.get('fixed_valid', True)

    # Augmentation
    args_dict['random_aug'] = aug_config.get('random_aug', False)
    args_dict['aug_1'] = aug_config.get('aug_1', [1, 1, 1, 1, 1, 1])
    args_dict['aug_2'] = aug_config.get('aug_2', [1, 1, 1, 1, 1, 1])
    args_dict['random_aug_thresholds'] = aug_config.get('random_aug_thresholds', [0.5, 0.5, 0.5, 0.6, 0.4])

    args = argparse.Namespace(**args_dict)
    return args


def load_and_concat_npz(npz_path, fine_tuning_areas):
    """npz 파일에서 최대 fine_tuning_areas개 array를 불러와 concat."""
    data = np.load(npz_path)
    keys = ['arr_7', 'arr_1', 'arr_5', 'arr_6', 'arr_9', 'arr_0', 'arr_2', 'arr_4', 'arr_8', 'arr_3']
    keys.reverse()
    keys = keys[:fine_tuning_areas]
    arrays = [data[k] for k in keys]
    return np.concatenate(arrays, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train a SimCLR on Landslide dataset")
    parser.add_argument("--config", type=str, default='configs/default.yaml', help="Path to config YAML file")
    parser.add_argument("--mode", type=str, default=None, choices=['plain', 'embedding', 'fusion'],
                        help="Training mode: plain (20ch raw), embedding (concat), fusion (dual-input)")
    parser.add_argument("--dir_name", type=str, default=None)
    parser.add_argument("--fine_trained_model_name", type=str, default=None)
    parser.add_argument("--fine_tuning_data_ratio", type=float, default=None)
    parser.add_argument("--gpu_device", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tau_plus", type=float, default=None)
    parser.add_argument("--embedding_dimension", type=int, default=None)
    parser.add_argument("--detailed", action="store_true", default=None)
    parser.add_argument("--undetailed", action="store_true", default=False)
    parser.add_argument("--fusion_encoder_type", type=str, default=None,
                        help="Fusion encoder type: multi, cross_attention, legacy")
    parser.add_argument("--verify_saved_model", action="store_true", default=None)
    parser.add_argument("--no_verify_saved_model", action="store_true", default=False)
    cmd_args = parser.parse_args()

    # Load config and apply overrides
    args = load_config(cmd_args.config)

    if cmd_args.mode is not None:
        args.mode = cmd_args.mode
    if cmd_args.dir_name is not None:
        args.dir_name = cmd_args.dir_name
    if cmd_args.fine_trained_model_name is not None:
        args.fine_trained_model_name = cmd_args.fine_trained_model_name
    if cmd_args.fine_tuning_data_ratio is not None:
        args.fine_tuning_data_ratio = cmd_args.fine_tuning_data_ratio
    if cmd_args.gpu_device is not None:
        args.gpu_device = cmd_args.gpu_device
    if cmd_args.temperature is not None:
        args.temperature = cmd_args.temperature
    if cmd_args.tau_plus is not None:
        args.tau_plus = cmd_args.tau_plus
    if cmd_args.embedding_dimension is not None:
        args.embedding_dimension = cmd_args.embedding_dimension
    if cmd_args.undetailed:
        args.detailed = False
    elif cmd_args.detailed is not None:
        args.detailed = True
    if cmd_args.fusion_encoder_type is not None:
        args.fusion_encoder_type = cmd_args.fusion_encoder_type
    if cmd_args.no_verify_saved_model:
        args.verify_saved_model = False
    elif cmd_args.verify_saved_model is not None:
        args.verify_saved_model = True

    mode = args.mode
    print(f"Mode: {mode}")
    print(f"Model: {args.model_type}, SSL: {args.ssl_type}")
    if mode == 'fusion':
        print(f"Fusion encoder: {args.fusion_encoder_type}")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_device}"

    # ================================================================
    # Phase 1: Load base data
    # ================================================================
    tif_img = np.load(args.tif_img_path)
    ls_img = np.load(args.ls_img_path)
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)

    real_patches, _, _, _, _, coor_list = patch_(
        args.height, args.width, tif_img, args.patch_size, args.strides
    )

    _patch_size = 28

    # ================================================================
    # Phase 2: Prepare pretraining data (mode-specific)
    # ================================================================
    if mode == 'plain':
        # Plain mode: resize all 20 channels together, scale
        real_patches = tf.image.resize(real_patches, (_patch_size, _patch_size))
        real_patches = np.array(real_patches)
        multi_data_scale = Multi_data_scaler(real_patches)
        train_scaled = multi_data_scale.multi_scale(real_patches)

    elif mode == 'embedding':
        # Embedding mode: transform → separate → resize → embed → concat → scale
        if args.apply_log:
            feat_idx = FEATURE_NAMES.index('sca')
            real_patches[:, :, :, feat_idx] = np.log1p(real_patches[:, :, :, feat_idx])
            if np.any(np.isnan(real_patches[:, :, :, feat_idx])):
                real_patches[:, :, :, feat_idx] = fill_nan_nearest_2d(real_patches[:, :, :, feat_idx])

        if args.apply_arcsinh:
            for curv_name in ['curv_plf', 'curv_prf', 'curv_std']:
                feat_idx = FEATURE_NAMES.index(curv_name)
                real_patches[:, :, :, feat_idx] = np.arcsinh(real_patches[:, :, :, feat_idx])
                if np.any(np.isnan(real_patches[:, :, :, feat_idx])):
                    real_patches[:, :, :, feat_idx] = fill_nan_nearest_2d(real_patches[:, :, :, feat_idx])

        continuous_features = real_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
        cat_idx_list = [CATEGORICAL_FEATURE_INDICES[feat] for feat in CATEGORICAL_FEATURES]
        categorical_features_data = real_patches[:, :, :, cat_idx_list]

        with tf.device('/CPU:0'):
            continuous_features_resized = np.array(
                tf.image.resize(continuous_features, (_patch_size, _patch_size), method='bilinear'))
            categorical_features_resized = np.array(tf.cast(
                tf.image.resize(tf.cast(categorical_features_data, tf.float32),
                               (_patch_size, _patch_size), method='nearest'), tf.int32))

        # Load or generate embeddings
        embedding_dimension = args.embedding_dimension
        use_detailed = args.detailed
        embedding_basename = "detailed_embedding" if use_detailed else "undetailed_embedding"
        labels_path = "description/detailed_labels.json" if use_detailed else "description/undetailed_labels.json"
        embedding_file = f"./embeddings/{embedding_basename}{embedding_dimension}.npy"

        if os.path.exists(embedding_file):
            print(f"Loading saved embeddings from {embedding_file}...")
            categorical_embeddings = np.load(embedding_file)
            if categorical_embeddings.shape[1] == 12:
                with tf.device('/CPU:0'):
                    categorical_embeddings = np.array(
                        tf.image.resize(tf.cast(categorical_embeddings, tf.float32),
                                       (_patch_size, _patch_size), method='nearest'))
        else:
            print(f"Generating embeddings...")
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found")
            client = OpenAI(api_key=api_key)
            temp = np.zeros((*categorical_features_resized.shape[:3], len(FEATURE_NAMES)))
            for idx, feat_name in enumerate(CATEGORICAL_FEATURES):
                temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat_name]] = categorical_features_resized[:, :, :, idx]
            categorical_embeddings, _ = create_categorical_embeddings(
                temp, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
                labels_path=labels_path, feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
                client=client, model="text-embedding-3-small",
                dimensions=embedding_dimension, batch_size=100, verbose=True
            )
            os.makedirs('./embeddings', exist_ok=True)
            np.save(embedding_file, categorical_embeddings)

        multi_data_scale = Multi_data_scaler(continuous_features_resized)
        continuous_features_scaled = multi_data_scale.multi_scale(continuous_features_resized)

        final_feature_dim = len(CONTINUOUS_FEATURE_INDICES) + categorical_embeddings.shape[3]
        train_scaled = np.zeros((continuous_features_scaled.shape[0], _patch_size, _patch_size, final_feature_dim))
        for idx in range(len(CONTINUOUS_FEATURE_INDICES)):
            train_scaled[:, :, :, idx] = continuous_features_scaled[:, :, :, idx]
        train_scaled[:, :, :, len(CONTINUOUS_FEATURE_INDICES):] = categorical_embeddings

    elif mode == 'fusion':
        # Fusion mode: transform → separate → resize → embed → keep separate
        if args.apply_log:
            feat_idx = FEATURE_NAMES.index('sca')
            real_patches[:, :, :, feat_idx] = np.log1p(real_patches[:, :, :, feat_idx])
            if np.any(np.isnan(real_patches[:, :, :, feat_idx])):
                real_patches[:, :, :, feat_idx] = fill_nan_nearest_2d(real_patches[:, :, :, feat_idx])

        if args.apply_arcsinh:
            for curv_name in ['curv_plf', 'curv_prf', 'curv_std']:
                feat_idx = FEATURE_NAMES.index(curv_name)
                real_patches[:, :, :, feat_idx] = np.arcsinh(real_patches[:, :, :, feat_idx])
                if np.any(np.isnan(real_patches[:, :, :, feat_idx])):
                    real_patches[:, :, :, feat_idx] = fill_nan_nearest_2d(real_patches[:, :, :, feat_idx])

        continuous_features = real_patches[:, :, :, CONTINUOUS_FEATURE_INDICES]
        cat_idx_list = [CATEGORICAL_FEATURE_INDICES[feat] for feat in CATEGORICAL_FEATURES]
        categorical_features_data = real_patches[:, :, :, cat_idx_list]

        with tf.device('/CPU:0'):
            continuous_features_resized = np.array(
                tf.image.resize(continuous_features, (_patch_size, _patch_size), method='bilinear'))
            categorical_features_resized = np.array(tf.cast(
                tf.image.resize(tf.cast(categorical_features_data, tf.float32),
                               (_patch_size, _patch_size), method='nearest'), tf.int32))

        embedding_dimension = args.embedding_dimension
        use_detailed = args.detailed
        embedding_basename = "detailed_embedding" if use_detailed else "undetailed_embedding"
        labels_path = "description/detailed_labels.json" if use_detailed else "description/undetailed_labels.json"
        embedding_file = f"./embeddings/{embedding_basename}{embedding_dimension}.npy"

        if os.path.exists(embedding_file):
            print(f"Loading saved embeddings from {embedding_file}...")
            categorical_embeddings = np.load(embedding_file)
            if categorical_embeddings.shape[1] == 12:
                with tf.device('/CPU:0'):
                    categorical_embeddings = np.array(
                        tf.image.resize(tf.cast(categorical_embeddings, tf.float32),
                                       (_patch_size, _patch_size), method='nearest'))
        else:
            print(f"Generating embeddings...")
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found")
            client = OpenAI(api_key=api_key)
            temp = np.zeros((*categorical_features_resized.shape[:3], len(FEATURE_NAMES)))
            for idx, feat_name in enumerate(CATEGORICAL_FEATURES):
                temp[:, :, :, CATEGORICAL_FEATURE_INDICES[feat_name]] = categorical_features_resized[:, :, :, idx]
            categorical_embeddings, _ = create_categorical_embeddings(
                temp, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
                labels_path=labels_path, feature_to_label_mapping=FEATURE_TO_LABEL_MAPPING,
                client=client, model="text-embedding-3-small",
                dimensions=embedding_dimension, batch_size=100, verbose=True
            )
            os.makedirs('./embeddings', exist_ok=True)
            np.save(embedding_file, categorical_embeddings)

        multi_data_scale = Multi_data_scaler(continuous_features_resized)
        continuous_features_scaled = multi_data_scale.multi_scale(continuous_features_resized)

        # Keep separate for fusion pretraining
        train_continuous = continuous_features_scaled
        train_categorical = categorical_embeddings

    # Initialize OpenAI client for train/valid processing (embedding/fusion modes)
    if mode in ('embedding', 'fusion'):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OPENAI_API_KEY not found")
        client = OpenAI(api_key=api_key)

    np.random.seed(118)

    # ================================================================
    # Phase 3: Pretraining (if ssl_type != no_pretrain)
    # ================================================================
    if mode == 'fusion':
        # Fusion mode input shapes (18x18 after augmentation crop)
        num_continuous = train_continuous.shape[3]
        num_categorical = train_categorical.shape[3]
        continuous_input_shape = (18, 18, num_continuous)
        categorical_input_shape = (18, 18, num_categorical)

    if args.ssl_type != "no_pretrain":
        weights_loaded = False
        extracted_encoder = None
        finetuned_model_path = "./finetuned_models/%s/SSL_1_weight.h5" % args.dir_name

        if os.path.exists(finetuned_model_path):
            # Load encoder from previously fine-tuned model
            try:
                if mode == 'fusion':
                    if args.model_type == 'ResNet':
                        if args.fusion_encoder_type == 'cross_attention':
                            extracted_encoder = build_cross_attention_fusion_encoder(
                                continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                        elif args.fusion_encoder_type == 'legacy':
                            extracted_encoder = build_fusion_encoder(
                                continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                        else:
                            extracted_encoder = build_multi_fusion_encoder(
                                continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                    elif args.model_type == 'ViT':
                        extracted_encoder = build_fusion_vit_encoder(
                            continuous_input_shape, categorical_input_shape, layer_name="fusion_vit_encoder")
                    else:
                        raise ValueError(f"Fusion mode only supports ResNet and ViT, got {args.model_type}")

                    temp_model = build_finetune_fusion_model(
                        continuous_input_shape, categorical_input_shape,
                        extracted_encoder, num_classes=2, training=False)
                    temp_model.load_weights(finetuned_model_path)
                    del temp_model
                else:
                    # plain/embedding mode
                    if mode == 'plain':
                        input_shape = (18, 18, real_patches.shape[3])
                    else:
                        final_feature_dim = len(CONTINUOUS_FEATURE_INDICES) + categorical_embeddings.shape[3]
                        input_shape = (18, 18, final_feature_dim)

                    if args.model_type == 'ResNet':
                        extracted_encoder = build_encoder(input_shape, layer_name="encoder")
                    elif args.model_type == 'CNN':
                        extracted_encoder = lsm_CNN(input_shape)
                    elif args.model_type == 'ViT':
                        extracted_encoder = vit_model(input_shape)

                    finetuned_model_temp = build_finetune_model(input_shape, extracted_encoder, num_classes=2, training=False)
                    finetuned_model_temp.load_weights(finetuned_model_path)

                    encoder_layer = None
                    for i, layer in enumerate(finetuned_model_temp.layers):
                        if i > 0 and hasattr(layer, 'layers') and len(layer.layers) > 0:
                            encoder_layer = layer
                            break

                    if encoder_layer is not None:
                        encoder_weights = encoder_layer.get_weights()
                        if args.model_type == 'ResNet':
                            extracted_encoder = build_encoder(input_shape, layer_name="encoder")
                        elif args.model_type == 'CNN':
                            extracted_encoder = lsm_CNN(input_shape)
                        elif args.model_type == 'ViT':
                            extracted_encoder = vit_model(input_shape)
                        extracted_encoder.set_weights(encoder_weights)

                    del finetuned_model_temp

                extracted_encoder.trainable = False
                weights_loaded = True
                print(f"Encoder loaded and FROZEN from: {finetuned_model_path}")

            except (KeyError, ValueError, OSError, Exception) as e:
                print(f"Warning: Could not load weights from {finetuned_model_path}: {e}")
                print("Proceeding with pretraining from scratch.")
                weights_loaded = False
                extracted_encoder = None

        if not weights_loaded:
            # Full pretraining from scratch
            if mode == 'fusion':
                data_1_cont, data_1_cat, data_2_cont, data_2_cat = [], [], [], []
                for i in range(len(train_continuous)):
                    cont_v1, cont_v2 = apply_simclr_augmentation(
                        train_continuous[i], args.aug_1, args.aug_2, args.random_aug, args.random_aug_thresholds)
                    cat_v1, cat_v2 = apply_simclr_augmentation(
                        train_categorical[i], args.aug_1, args.aug_2, args.random_aug, args.random_aug_thresholds)
                    data_1_cont.append(np.expand_dims(cont_v1, 0))
                    data_1_cat.append(np.expand_dims(cat_v1, 0))
                    data_2_cont.append(np.expand_dims(cont_v2, 0))
                    data_2_cat.append(np.expand_dims(cat_v2, 0))
                data_1_cont = np.concatenate(data_1_cont)
                data_1_cat = np.concatenate(data_1_cat)
                data_2_cont = np.concatenate(data_2_cont)
                data_2_cat = np.concatenate(data_2_cat)

                os.makedirs('./pretrained_model/%s' % args.dir_name, exist_ok=True)

                with tf.device(f'/GPU:{args.gpu_device}'):
                    pretrain_model = pre_train_simclr_fusion(
                        data_1_cont, data_1_cat, data_2_cont, data_2_cat,
                        batch_size=args.pre_batch_size, pre_model=args.model_type,
                        debiased=args.debiased, tau_plus=args.tau_plus,
                        temperature=args.temperature, epochs=args.pre_epochs,
                        learning_rate=args.pre_learning_rate,
                        fusion_encoder_type=args.fusion_encoder_type)
                del data_1_cont, data_1_cat, data_2_cont, data_2_cat

            else:
                # plain/embedding: single-input pretraining
                data_1, data_2 = [], []
                for i in range(len(train_scaled)):
                    view1, view2 = apply_simclr_augmentation(
                        train_scaled[i], args.aug_1, args.aug_2, args.random_aug, args.random_aug_thresholds)
                    data_1.append(np.expand_dims(view1, 0))
                    data_2.append(np.expand_dims(view2, 0))
                data_1 = np.concatenate(data_1)
                data_2 = np.concatenate(data_2)

                os.makedirs('./pretrained_model/%s' % args.dir_name, exist_ok=True)

                if args.ssl_type == "SimCLR":
                    with tf.device(f'/GPU:{args.gpu_device}'):
                        pretrain_model = pre_train_simclr(
                            data_1, data_2, batch_size=args.pre_batch_size,
                            pre_model=args.model_type, debiased=args.debiased,
                            tau_plus=args.tau_plus, temperature=args.temperature,
                            epochs=args.pre_epochs, learning_rate=args.pre_learning_rate)
                del data_1, data_2

            try:
                pretrain_model.save_weights(
                    "./pretrained_model/%s/%s.h5" % (args.dir_name, args.pre_trained_model_name))
            except:
                pass

    # Clean up pretraining data
    if mode == 'fusion':
        del train_continuous, train_categorical
    elif mode in ('plain', 'embedding'):
        del train_scaled

    # ================================================================
    # Phase 4: Load and prepare fine-tuning data
    # ================================================================
    non_ls_patches_features = np.load(args.non_ls_patches_path)
    non_ls_patches_labels = np.load(args.non_ls_labels_path)
    ls_patches_features = np.load(args.ls_patches_path)
    ls_patches_labels = np.load(args.ls_labels_path)

    valid_non_ls_ind = np.random.choice(non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]*0.1))
    valid_ls_ind = np.random.choice(ls_patches_features.shape[0], int(ls_patches_features.shape[0]*0.1))

    valid_non_ls = non_ls_patches_features[valid_non_ls_ind]
    valid_non_ls_label = non_ls_patches_labels[valid_non_ls_ind]
    valid_ls = ls_patches_features[valid_ls_ind]
    valid_ls_label = ls_patches_labels[valid_ls_ind]

    non_ls_patches_features = np.delete(non_ls_patches_features, valid_non_ls_ind, axis=0)
    non_ls_patches_labels = np.delete(non_ls_patches_labels, valid_non_ls_ind, axis=0)
    ls_patches_features = np.delete(ls_patches_features, valid_ls_ind, axis=0)
    ls_patches_labels = np.delete(ls_patches_labels, valid_ls_ind, axis=0)

    valid_patches = np.concatenate([valid_ls, valid_non_ls])
    valid_labels = np.concatenate([valid_ls_label, valid_non_ls_label])

    train_non_ls_ind = np.random.choice(non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]))
    train_ls_ind = np.random.choice(ls_patches_features.shape[0], int(ls_patches_features.shape[0]))

    train_non_ls_ind = train_non_ls_ind[:int(len(train_non_ls_ind)*args.fine_tuning_data_ratio)]
    train_ls_ind = train_ls_ind[:int(len(train_ls_ind)*args.fine_tuning_data_ratio)]

    train_non_ls = non_ls_patches_features[train_non_ls_ind]
    train_non_ls_label = non_ls_patches_labels[train_non_ls_ind]
    train_ls = ls_patches_features[train_ls_ind]
    train_ls_label = ls_patches_labels[train_ls_ind]

    train_patches = np.concatenate([train_ls, train_non_ls])
    train_labels = np.concatenate([train_ls_label, train_non_ls_label])

    # Extract center labels
    train_labels_ = []
    for i in range(train_labels.shape[0]):
        if np.sum(train_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >= 1.:
            train_labels_.append(1)
        else:
            train_labels_.append(0)

    valid_labels_ = []
    for i in range(valid_labels.shape[0]):
        if np.sum(valid_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >= 1.:
            valid_labels_.append(1)
        else:
            valid_labels_.append(0)

    train_labels_ = np.expand_dims(train_labels_, axis=1)
    valid_labels_ = np.expand_dims(valid_labels_, axis=1)

    # ================================================================
    # Phase 5: Process fine-tuning inputs (mode-specific)
    # ================================================================
    if mode == 'plain':
        train_patches = tf.image.resize(train_patches, (_patch_size, _patch_size))
        valid_patches = tf.image.resize(valid_patches, (_patch_size, _patch_size))
        train_patches = np.array(train_patches)
        valid_patches = np.array(valid_patches)
        train_patches_scaled = multi_data_scale.multi_scale(train_patches)
        valid_patches_scaled = multi_data_scale.multi_scale(valid_patches)
        train_patches_scaled = train_patches_scaled[:, 5:-5, 5:-5, :]
        valid_patches_scaled = valid_patches_scaled[:, 5:-5, 5:-5, :]
        input_shape = train_patches_scaled.shape[1:]

    elif mode == 'embedding':
        print("Processing train patches with embedding...")
        train_patches_scaled = process_patches_with_embedding(
            train_patches, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
            FEATURE_TO_LABEL_MAPPING, multi_data_scale, client, embedding_dimension,
            apply_arcsinh=args.apply_arcsinh, apply_log=args.apply_log,
            target_size=_patch_size, crop_size=18, return_separate=False,
            embedding_file=embedding_file, labels_path=labels_path)
        print("Processing valid patches with embedding...")
        valid_patches_scaled = process_patches_with_embedding(
            valid_patches, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
            FEATURE_TO_LABEL_MAPPING, multi_data_scale, client, embedding_dimension,
            apply_arcsinh=args.apply_arcsinh, apply_log=args.apply_log,
            target_size=_patch_size, crop_size=18, return_separate=False,
            embedding_file=embedding_file, labels_path=labels_path)
        input_shape = train_patches_scaled.shape[1:]

    elif mode == 'fusion':
        print("Processing train patches with embedding (fusion)...")
        train_continuous_scaled, train_categorical_emb = process_patches_with_embedding(
            train_patches, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
            FEATURE_TO_LABEL_MAPPING, multi_data_scale, client, embedding_dimension,
            apply_arcsinh=args.apply_arcsinh, apply_log=args.apply_log,
            target_size=_patch_size, crop_size=18, return_separate=True,
            embedding_file=embedding_file, labels_path=labels_path)
        print("Processing valid patches with embedding (fusion)...")
        valid_continuous_scaled, valid_categorical_emb = process_patches_with_embedding(
            valid_patches, FEATURE_NAMES, CATEGORICAL_FEATURES, CATEGORICAL_FEATURE_INDICES,
            FEATURE_TO_LABEL_MAPPING, multi_data_scale, client, embedding_dimension,
            apply_arcsinh=args.apply_arcsinh, apply_log=args.apply_log,
            target_size=_patch_size, crop_size=18, return_separate=True,
            embedding_file=embedding_file, labels_path=labels_path)
        continuous_input_shape = train_continuous_scaled.shape[1:]
        categorical_input_shape = train_categorical_emb.shape[1:]

    # ================================================================
    # Phase 6: Build model
    # ================================================================
    if args.ssl_type != "no_pretrain":
        if weights_loaded:
            encoder = extracted_encoder
            print(f"Using frozen encoder from SSL_1_weight.h5")
        else:
            if mode == 'fusion':
                if args.model_type == 'ResNet':
                    encoder = pretrain_model.get_layer("fusion_encoder")
                elif args.model_type == 'ViT':
                    encoder = pretrain_model.get_layer("fusion_vit_encoder")
                else:
                    raise ValueError(f"Fusion mode only supports ResNet and ViT, got {args.model_type}")
            else:
                if args.ssl_type == "SimCLR":
                    encoder = pretrain_model.get_layer("encoder")

        if mode == 'fusion':
            finetune_model = build_finetune_fusion_model(
                continuous_input_shape, categorical_input_shape, encoder, num_classes=2, training=False)
        else:
            finetune_model = build_finetune_model(input_shape, encoder, num_classes=2, training=False)

        finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

        trainable_count = sum(tf.keras.backend.count_params(w) for w in finetune_model.trainable_weights)
        non_trainable_count = sum(tf.keras.backend.count_params(w) for w in finetune_model.non_trainable_weights)
        print(f"Trainable params: {trainable_count:,} | Non-trainable params: {non_trainable_count:,}")
    else:
        # No pretraining: build fresh model
        if mode == 'fusion':
            if args.model_type == "ResNet":
                if args.fusion_encoder_type == 'cross_attention':
                    encoder = build_cross_attention_fusion_encoder(
                        continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                elif args.fusion_encoder_type == 'legacy':
                    encoder = build_fusion_encoder(
                        continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                else:
                    encoder = build_multi_fusion_encoder(
                        continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
            elif args.model_type == "ViT":
                encoder = build_fusion_vit_encoder(
                    continuous_input_shape, categorical_input_shape, layer_name="fusion_vit_encoder")
            else:
                raise ValueError(f"Fusion mode only supports ResNet and ViT, got {args.model_type}")

            finetune_model = build_finetune_fusion_model(
                continuous_input_shape, categorical_input_shape, encoder, num_classes=2, training=True)
        else:
            if args.model_type == "ResNet":
                encoder = build_encoder(input_shape)
                finetune_model = build_finetune_model(input_shape, encoder)
            elif args.model_type == "CNN":
                finetune_model = build_cnn_model(input_shape)
            elif args.model_type == "ViT":
                finetune_model = build_vit_model(input_shape)

        finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=['accuracy'])

    # ================================================================
    # Phase 7: Train
    # ================================================================
    if args.ssl_type != "no_pretrain":
        model_name = 'finetuned_models'
    else:
        if args.model_type == "ResNet":
            model_name = "trained_resnet_models"
        elif args.model_type == "CNN":
            model_name = "trained_cnn_models"
        else:
            model_name = "trained_vit_models"

    os.makedirs('./%s/%s' % (model_name, args.dir_name), exist_ok=True)

    if mode == 'fusion':
        checkpoint_path = "./%s/%s/%s" % (model_name, args.dir_name, args.fine_trained_model_name)
        checkpoint_weight_path = "./%s/%s/%s_weight.h5" % (model_name, args.dir_name, args.fine_trained_model_name)

        # Clean existing checkpoints
        if os.path.exists(checkpoint_path):
            try:
                if os.path.isdir(checkpoint_path):
                    import shutil
                    shutil.rmtree(checkpoint_path)
                else:
                    os.remove(checkpoint_path)
            except:
                pass
        if os.path.exists(checkpoint_weight_path):
            try:
                os.remove(checkpoint_weight_path)
            except:
                pass

        checkpoint_cb = SafeModelCheckpoint(
            checkpoint_path, save_best_only=True, overwrite=True, save_format='tf')
        checkpoint_cb2 = SafeModelCheckpoint(
            checkpoint_weight_path, save_weights_only=True, overwrite=True)
    else:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "./%s/%s/%s.h5" % (model_name, args.dir_name, args.fine_trained_model_name), save_best_only=True)
        checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint(
            "./%s/%s/%s_weight.h5" % (model_name, args.dir_name, args.fine_trained_model_name), save_weight_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    reduce_lr_plateu_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    with tf.device(f'/GPU:{args.gpu_device}'):
        if mode == 'fusion':
            history_2 = finetune_model.fit(
                [train_continuous_scaled, train_categorical_emb], train_labels_,
                validation_data=([valid_continuous_scaled, valid_categorical_emb], valid_labels_),
                epochs=200, batch_size=32,
                callbacks=[checkpoint_cb, checkpoint_cb2, early_stopping_cb, reduce_lr_plateu_cb])
        else:
            history_2 = finetune_model.fit(
                train_patches_scaled, train_labels_,
                validation_data=(valid_patches_scaled, valid_labels_),
                epochs=200, batch_size=32,
                callbacks=[checkpoint_cb, checkpoint_cb2, early_stopping_cb, reduce_lr_plateu_cb])

    # Verify saved model (fusion only)
    if mode == 'fusion' and args.verify_saved_model:
        print("\nVerifying SavedModel/weights load...")
        sample_cont = train_continuous_scaled[:1]
        sample_cat = train_categorical_emb[:1]

        def build_selected_fusion_encoder():
            if args.model_type == "ResNet":
                if args.fusion_encoder_type == 'cross_attention':
                    return build_cross_attention_fusion_encoder(
                        continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                elif args.fusion_encoder_type == 'legacy':
                    return build_fusion_encoder(
                        continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
                return build_multi_fusion_encoder(
                    continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder")
            if args.model_type == "ViT":
                return build_fusion_vit_encoder(
                    continuous_input_shape, categorical_input_shape, layer_name="fusion_vit_encoder")
            raise ValueError(f"Fusion mode only supports ResNet and ViT")

        try:
            loaded = tf.keras.models.load_model(checkpoint_path, custom_objects=CUSTOM_OBJECTS)
            _ = loaded([sample_cont, sample_cat], training=False)
            print(f"SavedModel load OK: {checkpoint_path}")
        except Exception as e:
            print(f"SavedModel load failed: {checkpoint_path} -> {e}")

        try:
            verify_encoder = build_selected_fusion_encoder()
            verify_model = build_finetune_fusion_model(
                continuous_input_shape, categorical_input_shape, verify_encoder, num_classes=2, training=False)
            _ = verify_model([sample_cont, sample_cat], training=False)
            verify_model.load_weights(checkpoint_weight_path)
            print(f"Weights load OK: {checkpoint_weight_path}")
        except Exception as e:
            print(f"Weights load failed: {checkpoint_weight_path} -> {e}")

    # ================================================================
    # Phase 8: Save loss curve
    # ================================================================
    os.makedirs('./loss_curve/%s' % args.dir_name, exist_ok=True)

    plt.rc('font', size=15)
    plt.plot(history_2.history['val_loss'], label='valid_loss')
    plt.plot(history_2.history['loss'], label='train_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('Categorical_Cross_Entropy')
    plt.title('Loss curve')
    plt.ylim(0.001, 1.)
    plt.tight_layout()
    plt.savefig('./loss_curve/%s/%s' % (args.dir_name, args.fine_trained_model_name))


if __name__ == "__main__":
    main()
