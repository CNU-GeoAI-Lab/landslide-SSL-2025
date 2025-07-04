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

def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
    # 첫 번째 컨볼루션 레이어
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation(activation)(y)
    
    # 두 번째 컨볼루션 레이어
    y = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # 입력과 출력 크기 조정 (stride로 인해 다를 경우)
    if stride > 1 or x.shape[-1] != filters:
        x = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    # Residual 연결
    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.Activation(activation)(out)
    return out

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
def build_simclr_model(input_shape=(None, None, 20), embedding_dim=128, pre_model = 'ResNet'):
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

def build_finetune_model(input_shape, encoder, num_classes=2, training=False):
    inputs = tf.keras.Input(shape=(None, None, None))
    x = encoder(inputs, training=training)  # Freeze된 Encoder 사용
    x = tf.keras.layers.Dense(128, activation='relu')(x)  # 새로운 Fully Connected Laye
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # 최종 분류 Layer
    return tf.keras.Model(inputs, x, name="FineTuned_Model")

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

def vit_model(input_shape=(None, None, 20), embed_dim=64, num_heads=4, mlp_dim = 128, num_transformer_blocks=4, layer_name="encoder"):
    inputs = layers.Input(shape=input_shape)
    
        # Patch Embedding
    # [batch, 12, 12, 20] -> [batch, 12, 12, 64]
    #x = tf.image.resize(inputs, (12, 12))
    #x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=embed_dim, kernel_size=3, strides=1, padding="same", data_format = 'channels_last')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Reshape to Sequence
    # [batch, 12, 12, 64] - [batch, 144, 64]
    x = layers.Reshape((-1, embed_dim))(x)
        
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

    # Add Positional Encoding
    pos_emb_layer = PositionalEmbedding(num_patches=18**2, d_model=embed_dim)
    x = pos_emb_layer(x)

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, embed_dim, num_heads, mlp_dim, 0.1)

    x = layers.Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(mlp_dim, activation="gelu")(x)
    model = Model(inputs=inputs, outputs = x, name = layer_name)

    return model

def build_vit_model(input_shape = (None, None, 20)):
    encoder = vit_model(input_shape)
    final_head = final_layer(encoder.output_shape[-1])

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = final_head(features)
    
    return tf.keras.Model(inputs, outputs, name="vit_model")

