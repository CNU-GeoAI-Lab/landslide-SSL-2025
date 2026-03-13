import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model

        self.pos_emb = self.add_weight(
            "pos_emb_T",
            shape=(1, self.num_patches + 1, self.d_model),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
        )
        self.class_emb = self.add_weight(
            "class_emb_T",
            shape=(1, 1, self.d_model),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # broadcast -> 1, 1, 64 의 class emb 가중치를 [batch_size, 1, d_model]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        return tf.concat([class_emb, x], axis=1) + self.pos_emb

    # 객체의 현재 구성을 딕셔너리로 반환
    # 사용 이유: 직렬화(저장) 할 때, 객체를 재구성하는 데 필요한 매개변수 정보를 저장
    # custom layer 나 subclass model 의 경우 저장하는 데 불안정하기 때문에 이 함수를 적용
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "d_model": self.d_model,
        })
        return config

    # 저장된 config 딕셔너리를 사용하여 객체를 복원할 때 사용
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def residual_block(x, filters, kernel_size=3, stride=1, activation='relu', name=None):
    # 첫 번째 컨볼루션 레이어
    conv_name = f"{name}_conv1" if name else None
    bn_name = f"{name}_bn1" if name else None
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=conv_name)(x)
    y = tf.keras.layers.BatchNormalization(name=bn_name)(y)
    y = tf.keras.layers.Activation(activation)(y)
    
    # 두 번째 컨볼루션 레이어
    conv_name2 = f"{name}_conv2" if name else None
    bn_name2 = f"{name}_bn2" if name else None
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', name=conv_name2)(y)
    y = tf.keras.layers.BatchNormalization(name=bn_name2)(y)
    
    # 입력과 출력 크기 조정 (stride로 인해 다를 경우)
    if stride > 1 or x.shape[-1] != filters:
        shortcut_name = f"{name}_shortcut" if name else None
        x = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', name=shortcut_name)(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_shortcut_bn" if name else None)(x)
    
    # Residual 연결
    out = tf.keras.layers.Add(name=f"{name}_add" if name else None)([x, y])
    out = tf.keras.layers.Activation(activation, name=f"{name}_activation" if name else None)(out)
    return out

# Feature Fusion Module: Helper function to create a fusion module
def feature_fusion_module(input_features, filters, stride, name_prefix):
    """
    Feature Fusion Module: Fuses features, then splits back into continuous and categorical branches.
    
    This module implements the feature fusion pattern where:
    1. Categorical encoder and continuous encoder process the fused input separately
    2. Their outputs are fused together
    3. The fused result is returned for the next iteration
    
    Args:
        input_features: Input features (fused features from previous step)
        filters: Number of filters for each encoder branch output
        stride: Stride for residual blocks
        name_prefix: Prefix for layer names
    
    Returns:
        Fused features after processing through continuous and categorical encoders
    """
    # Split fused features into two branches (continuous and categorical encoders)
    # Each encoder processes the same fused input independently
    cont_features = residual_block(input_features, filters=filters, stride=stride, name=f"{name_prefix}_cont_encoder")
    cat_features = residual_block(input_features, filters=filters, stride=stride, name=f"{name_prefix}_cat_encoder")
    
    # Fusion: Concatenate continuous and categorical features
    fused = tf.keras.layers.Concatenate(axis=-1, name=f"{name_prefix}_fusion_concat")([cont_features, cat_features])
    
    # Fusion convolution to merge the concatenated features
    # Output channels = filters * 2 (since we concatenated two branches with 'filters' channels each)
    fused = tf.keras.layers.Conv2D(filters * 2, 3, strides=1, padding='same', activation='relu', name=f"{name_prefix}_fusion_conv")(fused)
    fused = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_fusion_bn")(fused)
    
    return fused

# Cross-Attention Fusion Module: bidirectional attention between continuous and categorical streams
def cross_attention_fusion_module(cont_features, cat_features, filters, num_heads, name_prefix):
    """
    Cross-attention fusion module.

    Args:
        cont_features: continuous feature map (B, H, W, C)
        cat_features: categorical feature map (B, H, W, C)
        filters: channel dimension for attention projections
        num_heads: number of attention heads
        name_prefix: prefix for layer names

    Returns:
        cont_out: updated continuous feature map
        cat_out: updated categorical feature map
        fused: fused feature map (concat + 1x1 conv)
    """
    # Project to same channel dimension
    cont_proj = tf.keras.layers.Conv2D(filters, 1, padding='same', name=f"{name_prefix}_cont_proj")(cont_features)
    cat_proj = tf.keras.layers.Conv2D(filters, 1, padding='same', name=f"{name_prefix}_cat_proj")(cat_features)

    # Flatten spatial dims to tokens (keep channel dim static)
    cont_tokens = tf.keras.layers.Reshape((-1, filters), name=f"{name_prefix}_cont_tokens")(cont_proj)
    cat_tokens = tf.keras.layers.Reshape((-1, filters), name=f"{name_prefix}_cat_tokens")(cat_proj)

    key_dim = max(1, filters // num_heads)
    cont_att = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name_prefix}_cont_attn"
    )(cont_tokens, cat_tokens, cat_tokens)
    cat_att = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name_prefix}_cat_attn"
    )(cat_tokens, cont_tokens, cont_tokens)

    # Residual + normalization
    cont_tokens = tf.keras.layers.LayerNormalization(name=f"{name_prefix}_cont_ln")(cont_tokens + cont_att)
    cat_tokens = tf.keras.layers.LayerNormalization(name=f"{name_prefix}_cat_ln")(cat_tokens + cat_att)

    # Tokens -> spatial maps using static shape from projected tensors
    h, w = int(cont_proj.shape[1]), int(cont_proj.shape[2])
    cont_out = tf.keras.layers.Reshape((h, w, filters), name=f"{name_prefix}_cont_spatial")(cont_tokens)
    cat_out = tf.keras.layers.Reshape((h, w, filters), name=f"{name_prefix}_cat_spatial")(cat_tokens)

    # Fuse both streams
    fused = tf.keras.layers.Concatenate(axis=-1, name=f"{name_prefix}_fusion_concat")([cont_out, cat_out])
    fused = tf.keras.layers.Conv2D(filters * 2, 1, padding='same', activation='relu', name=f"{name_prefix}_fusion_proj")(fused)
    fused = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_fusion_bn")(fused)
    return cont_out, cat_out, fused


def build_cross_attention_fusion_encoder(continuous_input_shape, categorical_input_shape,
                                         layer_name="cross_attention_fusion_encoder",
                                         num_heads=4):
    """
    Build fusion encoder that uses cross-attention to mix continuous and categorical features
    at multiple stages.
    """
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")

    # Initial conv for each stream
    cont_x = tf.keras.layers.Dropout(0.2, name="cont_dropout")(continuous_input)
    cont_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cont_conv1")(cont_x)
    cont_x = tf.keras.layers.BatchNormalization(name="cont_bn1")(cont_x)

    cat_x = tf.keras.layers.Dropout(0.2, name="cat_dropout")(categorical_input)
    cat_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cat_conv1")(cat_x)
    cat_x = tf.keras.layers.BatchNormalization(name="cat_bn1")(cat_x)

    # Stage 1
    cont_x = residual_block(cont_x, filters=64, stride=2, name="ca_res1_cont")
    cat_x = residual_block(cat_x, filters=64, stride=2, name="ca_res1_cat")
    cont_x, cat_x, fused = cross_attention_fusion_module(cont_x, cat_x, filters=64, num_heads=num_heads, name_prefix="ca_stage1")

    # Stage 2
    cont_x = residual_block(cont_x, filters=128, stride=3, name="ca_res2_cont")
    cat_x = residual_block(cat_x, filters=128, stride=3, name="ca_res2_cat")
    cont_x, cat_x, fused = cross_attention_fusion_module(cont_x, cat_x, filters=128, num_heads=num_heads, name_prefix="ca_stage2")

    # Stage 3
    cont_x = residual_block(cont_x, filters=256, stride=3, name="ca_res3_cont")
    cat_x = residual_block(cat_x, filters=256, stride=3, name="ca_res3_cat")
    cont_x, cat_x, fused = cross_attention_fusion_module(cont_x, cat_x, filters=256, num_heads=num_heads, name_prefix="ca_stage3")

    # Stage 4
    cont_x = residual_block(cont_x, filters=512, stride=1, name="ca_res4_cont")
    cat_x = residual_block(cat_x, filters=512, stride=1, name="ca_res4_cat")
    cont_x, cat_x, fused = cross_attention_fusion_module(cont_x, cat_x, filters=512, num_heads=num_heads, name_prefix="ca_stage4")

    x = tf.keras.layers.GlobalAveragePooling2D(name="ca_gap")(fused)
    x = tf.keras.layers.Dense(128, activation='relu', name="ca_dense")(x)
    return tf.keras.Model([continuous_input, categorical_input], x, name=layer_name)

# Multi-Path Feature Fusion Encoder: 5-path architecture with intermediate fusions
def build_multi_fusion_encoder(continuous_input_shape, categorical_input_shape, layer_name="multi_fusion_encoder"):
    """
    Build new feature fusion encoder with multi-path architecture.
    
    Architecture:
    1. Continuous encoder: Full ResNet (Conv2D -> 4 residual blocks)
    2. Embedding encoder: Full ResNet (Conv2D -> 4 residual blocks)
    3. Fusion 1: Concatenate 1st residual block outputs -> ResNet from 2nd block (3 blocks)
    4. Fusion 2: Concatenate 2nd residual block outputs -> ResNet from 3rd block (2 blocks)
    5. Fusion 3: Concatenate 3rd residual block outputs -> Single residual block
    6. Concatenate all 5 outputs -> GlobalAveragePooling2D -> Dense(128)
    
    Args:
        continuous_input_shape: Shape of continuous features (H, W, C_cont)
        categorical_input_shape: Shape of categorical embeddings (H, W, C_cat)
        layer_name: Name for the encoder
    
    Returns:
        Model with two inputs (continuous, categorical) and one output
    """
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")
    
    # ============================================================================
    # 1. Continuous Encoder: Full ResNet process
    # ============================================================================
    cont_x = tf.keras.layers.Dropout(0.2, name="cont_dropout")(continuous_input)
    cont_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cont_conv1")(cont_x)
    cont_x = tf.keras.layers.BatchNormalization(name="cont_bn1")(cont_x)
    
    # Store intermediate outputs for fusion
    cont_res1 = residual_block(cont_x, filters=64, stride=2, name="cont_res1")
    cont_res2 = residual_block(cont_res1, filters=128, stride=3, name="cont_res2")
    cont_res3 = residual_block(cont_res2, filters=256, stride=3, name="cont_res3")
    cont_final = residual_block(cont_res3, filters=512, stride=1, name="cont_res4")
    
    # ============================================================================
    # 2. Embedding Encoder: Full ResNet process
    # ============================================================================
    cat_x = tf.keras.layers.Dropout(0.2, name="cat_dropout")(categorical_input)
    cat_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cat_conv1")(cat_x)
    cat_x = tf.keras.layers.BatchNormalization(name="cat_bn1")(cat_x)
    
    # Store intermediate outputs for fusion
    cat_res1 = residual_block(cat_x, filters=64, stride=2, name="cat_res1")
    cat_res2 = residual_block(cat_res1, filters=128, stride=3, name="cat_res2")
    cat_res3 = residual_block(cat_res2, filters=256, stride=3, name="cat_res3")
    cat_final = residual_block(cat_res3, filters=512, stride=1, name="cat_res4")
    
    # ============================================================================
    # 3. Fusion 1: Concatenate 1st residual block outputs -> ResNet from 2nd block
    # ============================================================================
    fusion1_input = tf.keras.layers.Concatenate(axis=-1, name="fusion1_concat")([cont_res1, cat_res1])
    # Project to 128 channels (since concatenating 64+64=128)
    fusion1_input = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name="fusion1_proj")(fusion1_input)
    fusion1_input = tf.keras.layers.BatchNormalization(name="fusion1_bn")(fusion1_input)
    
    # ResNet from 2nd block (3 blocks: 128->256->512)
    fusion1_res2 = residual_block(fusion1_input, filters=128, stride=3, name="fusion1_res2")
    fusion1_res3 = residual_block(fusion1_res2, filters=256, stride=3, name="fusion1_res3")
    fusion1_final = residual_block(fusion1_res3, filters=512, stride=1, name="fusion1_res4")
    
    # ============================================================================
    # 4. Fusion 2: Concatenate 2nd residual block outputs -> ResNet from 3rd block (2 blocks only)
    # ============================================================================
    fusion2_input = tf.keras.layers.Concatenate(axis=-1, name="fusion2_concat")([cont_res2, cat_res2])
    # Project to 256 channels (since concatenating 128+128=256)
    fusion2_input = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name="fusion2_proj")(fusion2_input)
    fusion2_input = tf.keras.layers.BatchNormalization(name="fusion2_bn")(fusion2_input)
    
    # ResNet from 3rd block (2 blocks only: 256->512)
    fusion2_res3 = residual_block(fusion2_input, filters=256, stride=3, name="fusion2_res3")
    fusion2_final = residual_block(fusion2_res3, filters=512, stride=1, name="fusion2_res4")
    
    # ============================================================================
    # 5. Fusion 3: Concatenate 3rd residual block outputs -> Single residual block
    # ============================================================================
    fusion3_input = tf.keras.layers.Concatenate(axis=-1, name="fusion3_concat")([cont_res3, cat_res3])
    # Project to 512 channels (since concatenating 256+256=512)
    fusion3_input = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name="fusion3_proj")(fusion3_input)
    fusion3_input = tf.keras.layers.BatchNormalization(name="fusion3_bn")(fusion3_input)
    
    # Single residual block
    fusion3_final = residual_block(fusion3_input, filters=512, stride=1, name="fusion3_res4")
    
    # ============================================================================
    # 6. Concatenate all 5 outputs -> GlobalAveragePooling2D -> Dense(128)
    # ============================================================================
    # All 5 outputs should have shape (batch, H, W, 512) at this point
    # Concatenate along channel dimension
    all_outputs = tf.keras.layers.Concatenate(axis=-1, name="final_concat")([
        cont_final,    # (batch, H, W, 512)
        cat_final,     # (batch, H, W, 512)
        fusion1_final, # (batch, H, W, 512)
        fusion2_final, # (batch, H, W, 512)
        fusion3_final  # (batch, H, W, 512)
    ])
    # Result: (batch, H, W, 2560) = 5 * 512
    
    # GlobalAveragePooling2D
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(all_outputs)
    # Result: (batch, 2560)
    
    # Dense(128)
    x = tf.keras.layers.Dense(128, activation='relu', name="final_dense")(x)
    # Result: (batch, 128)
    
    return tf.keras.Model([continuous_input, categorical_input], x, name=layer_name)

# Feature Fusion Encoder: Continuous와 Categorical을 분리 입력받아 각각 처리 후 모든 층에서 fusion
def build_fusion_encoder(continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder"):
    """
    Build encoder with feature fusion module pattern applied at every layer.
    
    Feature Fusion Module Pattern:
    - Categorical encoder and continuous encoder are used separately
    - Their outputs are fused together
    - The fused result goes back into each encoder again
    - This process repeats at every layer
    
    Args:
        continuous_input_shape: Shape of continuous features (H, W, C_cont)
        categorical_input_shape: Shape of categorical embeddings (H, W, C_cat)
        layer_name: Name for the encoder
    
    Returns:
        Model with two inputs (continuous, categorical) and one output
    """
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")
    
    # Initial continuous feature extraction branch
    cont_x = tf.keras.layers.Dropout(0.2)(continuous_input)
    cont_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cont_conv1")(cont_x)
    cont_x = tf.keras.layers.BatchNormalization(name="cont_bn1")(cont_x)
    
    # Initial categorical feature extraction branch
    cat_x = tf.keras.layers.Dropout(0.2)(categorical_input)
    cat_x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name="cat_conv1")(cat_x)
    cat_x = tf.keras.layers.BatchNormalization(name="cat_bn1")(cat_x)
    
    # Initial fusion: First fusion of continuous and categorical features
    fused = tf.keras.layers.Concatenate(axis=-1, name="fusion_initial")([cont_x, cat_x])
    fused = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name="fusion_initial_conv")(fused)
    fused = tf.keras.layers.BatchNormalization(name="fusion_initial_bn")(fused)
    
    # Apply feature fusion module at every layer (64 -> 128 -> 256 -> 512 channels)
    # Feature Fusion Module 1: 128 channels -> 256 channels
    fused = feature_fusion_module(fused, filters=128, stride=2, name_prefix="ffm1")
    
    # Feature Fusion Module 2: 256 channels -> 512 channels
    fused = feature_fusion_module(fused, filters=256, stride=3, name_prefix="ffm2")
    
    # Feature Fusion Module 3: 512 channels -> 1024 channels, then reduce to 512
    fused = feature_fusion_module(fused, filters=512, stride=3, name_prefix="ffm3")
    
    # Final feature fusion module: 1024 channels -> 512 channels
    fused = feature_fusion_module(fused, filters=512, stride=1, name_prefix="ffm4")
    
    x = fused
    
    # Final processing after all fusion modules
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    return tf.keras.Model([continuous_input, categorical_input], x, name=layer_name)

# SimCLR 모델 정의
def build_encoder(input_shape, layer_name="encoder"):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Dropout(0.2)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = residual_block(x, filters=64, stride=2)
    x = residual_block(x, filters=128, stride=3)           # 12x12x64
    x = residual_block(x, filters=256, stride=3) # 6x6x128
    x = residual_block(x, filters=512)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (batch_size, 128)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return tf.keras.Model(inputs, x, name = layer_name)

# 프로젝션 헤드 모델
def build_projection_head(input_dim=128, embedding_dim=128):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(embedding_dim)(x)
    return tf.keras.Model(inputs, outputs, name="projection_head")

# SimCLR 모델 조합
def build_simclr_model(input_shape, embedding_dim=128, pre_model = 'ResNet'):
    """pre_model : ['ResNet', 'CNN', 'ViT']"""

    if pre_model=='ResNet':
        encoder = build_encoder(input_shape, layer_name="encoder")
    elif pre_model=='CNN':
        encoder = lsm_CNN(input_shape)
    elif pre_model=='ViT':
        encoder = vit_model(input_shape)
    else:
        raise ValueError("pre_model %s is not exist"%pre_model)
    
    projection_head = build_projection_head(encoder.output_shape[-1], embedding_dim)
    
    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    projections = projection_head(features)
    
    return tf.keras.Model(inputs, projections, name="simclr_model")

# SimCLR Fusion 모델: 두 입력을 받아 fusion encoder 사용
def build_simclr_fusion_model(continuous_input_shape, categorical_input_shape, embedding_dim=128,
                              pre_model='ResNet', fusion_encoder_type='multi'):
    """
    Build SimCLR model with feature fusion for continuous and categorical features.
    All layers now use feature fusion module pattern by default.
    
    Args:
        continuous_input_shape: Shape of continuous features (H, W, C_cont)
        categorical_input_shape: Shape of categorical embeddings (H, W, C_cat)
        embedding_dim: Dimension of projection output
        pre_model: Base model type ('ResNet' or 'ViT' supported for fusion)
    Returns:
        Model with two inputs (continuous, categorical) and projection output
    """
    if pre_model == 'ResNet':
        if fusion_encoder_type == 'cross_attention':
            encoder = build_cross_attention_fusion_encoder(
                continuous_input_shape, categorical_input_shape, layer_name="fusion_encoder"
            )
        elif fusion_encoder_type == 'legacy':
            encoder = build_fusion_encoder(continuous_input_shape, categorical_input_shape, 
                                          layer_name="fusion_encoder")
        else:
            # Default: multi-fusion encoder
            encoder = build_multi_fusion_encoder(continuous_input_shape, categorical_input_shape, 
                                          layer_name="fusion_encoder")
    elif pre_model == 'ViT':
        encoder = build_fusion_vit_encoder(continuous_input_shape, categorical_input_shape, 
                                          layer_name="fusion_vit_encoder")
    else:
        raise ValueError(f"Fusion model supports 'ResNet' and 'ViT', got {pre_model}")
    
    projection_head = build_projection_head(encoder.output_shape[-1], embedding_dim)
    
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")
    
    features = encoder([continuous_input, categorical_input])
    projections = projection_head(features)
    
    return tf.keras.Model([continuous_input, categorical_input], projections, name="simclr_fusion_model")

def build_finetune_model(input_shape, encoder, num_classes=2, training=False):
    inputs = tf.keras.Input(shape=(None, None, None))
    x = encoder(inputs, training=training)  # Freeze된 Encoder 사용
    x = tf.keras.layers.Dense(128, activation='relu')(x)  # 새로운 Fully Connected Laye
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # 최종 분류 Layer
    return tf.keras.Model(inputs, x, name="FineTuned_Model")

def build_finetune_fusion_model(continuous_input_shape, categorical_input_shape, encoder, num_classes=2, training=False):
    """
    Build fine-tune model with feature fusion encoder.
    
    Args:
        continuous_input_shape: Shape of continuous features (H, W, C_cont)
        categorical_input_shape: Shape of categorical embeddings (H, W, C_cat)
        encoder: Fusion encoder model
        num_classes: Number of output classes
        training: Whether encoder is in training mode
    
    Returns:
        Model with two inputs (continuous, categorical) and classification output
    """
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")
    
    x = encoder([continuous_input, categorical_input], training=training)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model([continuous_input, categorical_input], x, name="FineTuned_Fusion_Model")

def lsm_CNN(input_size = (None, None, 20), layer_name="encoder"):  
    inputs = tf.keras.Input(shape=input_size)
    
    layer_0 = Dropout(0.2)(inputs)

    layer_0 = Conv2D(64, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = Conv2D(64, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = MaxPool2D((2, 2))(layer_0)

    layer_0 = Conv2D(128, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = Conv2D(128, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = MaxPool2D((2, 2))(layer_0)

    layer_0 = Conv2D(256, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = Conv2D(256, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = MaxPool2D((2, 2))(layer_0)

    layer_0 = Conv2D(512, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = Conv2D(512, 3, activation = 'relu', padding='same', data_format="channels_last")(layer_0)
    layer_0 = BatchNormalization()(layer_0)

    layer_0 = GlobalAveragePooling2D()(layer_0)
    

    emb = Dense(128, activation='relu')(layer_0)

    model = Model(inputs=inputs, outputs = emb, name = layer_name)

    return model

def final_layer(input_dim=(None,)):
    inputs = tf.keras.Input(shape=input_dim)
    layer_0 = Dense(16, activation = 'relu')(inputs)
    output = Dense(2, activation='softmax')(layer_0)

    return Model(inputs = inputs, outputs = output, name='final')
    
def build_cnn_model(input_shape = (None, None, None)):
    encoder = lsm_CNN(input_shape)
    final_head = final_layer(encoder.output_shape[-1])

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = final_head(features)
    
    return tf.keras.Model(inputs, outputs, name="cnn_model")

def multi_head_self_attention(inputs, embed_dim, num_heads):
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
        )
    projection_dim = embed_dim // num_heads

    def separate_heads(x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    query = Dense(embed_dim)(inputs)
    key = Dense(embed_dim)(inputs)
    value = Dense(embed_dim)(inputs)

    query = separate_heads(query)
    key = separate_heads(key)
    value = separate_heads(value)

    attention_output = attention(query, key, value)
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(
        attention_output, (tf.shape(inputs)[0], -1, embed_dim)
    )
    return Dense(embed_dim)(concat_attention)

def transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout=0.2):
    attn_output = multi_head_self_attention(inputs, embed_dim, num_heads)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-4)(Add()([inputs, attn_output]))

    ffn = tf.keras.Sequential(
        [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
    )
    ffn_output = ffn(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-4)(Add()([out1, ffn_output]))

def vit_model(input_shape, embed_dim=64, num_heads=4, mlp_dim=128, num_transformer_blocks=4, layer_name="encoder"):
    """
    Vision Transformer (ViT) model for self-supervised learning
    
    Args:
        input_shape: Input shape tuple (height, width, channels)
        embed_dim: Embedding dimension for transformer
        num_heads: Number of attention heads
        mlp_dim: MLP dimension in transformer blocks
        num_transformer_blocks: Number of transformer blocks
        layer_name: Name for the model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Calculate number of patches from input shape
    # Handle both static and dynamic shapes
    if input_shape[0] is not None and input_shape[1] is not None:
        # Static shape: calculate directly
        num_patches = input_shape[0] * input_shape[1]
    else:
        # For dynamic shapes, use a reasonable default or calculate at build time
        # Common input size is 18x18, so default to 324 patches
        # This will be adjusted when model is built with actual data
        num_patches = 18 * 18  # Default, will work for most cases
    
    # Patch Embedding: Convert image patches to embeddings
    # [batch, H, W, C] -> [batch, H, W, embed_dim]
    x = layers.Dropout(0.2)(inputs)
    x = layers.Conv2D(
        filters=embed_dim, 
        kernel_size=3, 
        strides=1, 
        padding="same", 
        data_format='channels_last',
        name="patch_embedding"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape to Sequence: [batch, H, W, embed_dim] -> [batch, num_patches, embed_dim]
    # Each spatial location becomes a patch token
    x = layers.Reshape((-1, embed_dim), name="patch_reshape")(x)
    
    # Add Positional Encoding and Class Token
    # Use the PositionalEmbedding class defined at the top of the file
    pos_emb_layer = PositionalEmbedding(num_patches=num_patches, d_model=embed_dim)
    x = pos_emb_layer(x)

    # Transformer Blocks: Apply multiple transformer blocks
    for i in range(num_transformer_blocks):
        x = transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1)

    # Extract class token (first token) for final representation
    # [batch, num_patches+1, embed_dim] -> [batch, embed_dim]
    class_token = layers.Lambda(lambda x: x[:, 0, :], name="extract_class_token")(x)
    
    # Final projection layers
    x = Dense(512, activation='relu', name="fc1")(class_token)
    x = layers.Dropout(0.1)(x)
    x = Dense(mlp_dim, activation="gelu", name="fc2")(x)
    
    model = Model(inputs=inputs, outputs=x, name=layer_name)
    return model

def build_vit_model(input_shape = (None, None, 20)):
    encoder = vit_model(input_shape)
    final_head = final_layer(encoder.output_shape[-1])

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = final_head(features)
    
    return tf.keras.Model(inputs, outputs, name="vit_model")

# ViT Feature Fusion Module: Helper function for ViT fusion
def vit_feature_fusion_module(input_features, embed_dim, num_heads, mlp_dim, num_blocks, name_prefix):
    """
    ViT Feature Fusion Module: Fuses features, then splits back into continuous and categorical branches.
    
    Args:
        input_features: Input features (fused features from previous step)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_dim: MLP dimension
        num_blocks: Number of transformer blocks to apply in each branch
        name_prefix: Prefix for layer names
    
    Returns:
        Fused features after processing through continuous and categorical encoders
    """
    # Split fused features into two branches (continuous and categorical encoders)
    # Each encoder processes the same fused input independently
    cont_features = input_features
    cat_features = input_features
    
    # Apply transformer blocks to each branch
    for i in range(num_blocks):
        cont_features = transformer_block(cont_features, embed_dim, num_heads, mlp_dim, dropout=0.1)
        cat_features = transformer_block(cat_features, embed_dim, num_heads, mlp_dim, dropout=0.1)
    
    # Fusion: Concatenate continuous and categorical features along feature dimension
    fused = tf.keras.layers.Concatenate(axis=-1, name=f"{name_prefix}_fusion_concat")([cont_features, cat_features])
    
    # Fusion projection to merge the concatenated features back to embed_dim
    fused = Dense(embed_dim, name=f"{name_prefix}_fusion_proj")(fused)
    
    return fused

# ViT Fusion Encoder: Continuous와 Categorical을 분리 입력받아 각각 처리 후 모든 층에서 fusion
def build_fusion_vit_encoder(continuous_input_shape, categorical_input_shape, layer_name="fusion_vit_encoder", 
                             embed_dim=64, num_heads=4, mlp_dim=128, num_transformer_blocks=4):
    """
    Build ViT encoder with feature fusion module pattern applied at every layer.
    
    Feature Fusion Module Pattern:
    - Categorical encoder and continuous encoder are used separately
    - Their outputs are fused together
    - The fused result goes back into each encoder again
    - This process repeats at every layer
    
    Args:
        continuous_input_shape: Shape of continuous features (H, W, C_cont)
        categorical_input_shape: Shape of categorical embeddings (H, W, C_cat)
        layer_name: Name for the encoder
        embed_dim: Embedding dimension for transformer
        num_heads: Number of attention heads
        mlp_dim: MLP dimension in transformer blocks
        num_transformer_blocks: Number of transformer blocks
    
    Returns:
        Model with two inputs (continuous, categorical) and one output
    """
    continuous_input = tf.keras.Input(shape=continuous_input_shape, name="continuous_input")
    categorical_input = tf.keras.Input(shape=categorical_input_shape, name="categorical_input")
    
    # Calculate number of patches
    if continuous_input_shape[0] is not None and continuous_input_shape[1] is not None:
        num_patches = continuous_input_shape[0] * continuous_input_shape[1]
    else:
        num_patches = 18 * 18  # Default
    
    # Continuous feature patch embedding
    cont_x = layers.Dropout(0.2)(continuous_input)
    cont_x = layers.Conv2D(
        filters=embed_dim, 
        kernel_size=3, 
        strides=1, 
        padding="same", 
        data_format='channels_last',
        name="cont_patch_embedding"
    )(cont_x)
    cont_x = layers.BatchNormalization(name="cont_bn")(cont_x)
    cont_x = layers.Activation("relu")(cont_x)
    cont_x = layers.Reshape((-1, embed_dim), name="cont_patch_reshape")(cont_x)
    
    # Categorical feature patch embedding
    cat_x = layers.Dropout(0.2)(categorical_input)
    cat_x = layers.Conv2D(
        filters=embed_dim, 
        kernel_size=3, 
        strides=1, 
        padding="same", 
        data_format='channels_last',
        name="cat_patch_embedding"
    )(cat_x)
    cat_x = layers.BatchNormalization(name="cat_bn")(cat_x)
    cat_x = layers.Activation("relu")(cat_x)
    cat_x = layers.Reshape((-1, embed_dim), name="cat_patch_reshape")(cat_x)
    
    # Initial fusion: First fusion of continuous and categorical patch embeddings
    fused = tf.keras.layers.Concatenate(axis=-1, name="fusion_initial")([cont_x, cat_x])
    fused = Dense(embed_dim, name="fusion_initial_proj")(fused)
    
    # Add positional embedding and class token to fused features
    pos_emb = PositionalEmbedding(num_patches=num_patches, d_model=embed_dim)
    fused = pos_emb(fused)
    
    # Apply feature fusion modules at every layer
    # Distribute transformer blocks across fusion modules
    blocks_per_module = max(1, num_transformer_blocks // 4)  # Divide blocks across ~4 fusion modules
    
    # Feature Fusion Module 1
    fused = vit_feature_fusion_module(fused, embed_dim, num_heads, mlp_dim, blocks_per_module, "ffm1")
    
    # Feature Fusion Module 2
    fused = vit_feature_fusion_module(fused, embed_dim, num_heads, mlp_dim, blocks_per_module, "ffm2")
    
    # Feature Fusion Module 3
    fused = vit_feature_fusion_module(fused, embed_dim, num_heads, mlp_dim, blocks_per_module, "ffm3")
    
    # Feature Fusion Module 4 (remaining blocks)
    remaining_blocks = num_transformer_blocks - (blocks_per_module * 3)
    if remaining_blocks > 0:
        fused = vit_feature_fusion_module(fused, embed_dim, num_heads, mlp_dim, remaining_blocks, "ffm4")
    
    # Extract class token
    class_token = layers.Lambda(lambda x: x[:, 0, :], name="extract_class_token")(fused)
    x = class_token
    
    # Final projection layers
    x = Dense(512, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.1)(x)
    x = Dense(mlp_dim, activation="gelu", name="fc2")(x)
    
    return tf.keras.Model([continuous_input, categorical_input], x, name=layer_name)

