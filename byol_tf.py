import tensorflow as tf
import numpy as np

# L2 normalize
def l2_normalize(x, axis=-1, epsilon=1e-12):
    return tf.math.l2_normalize(x, axis=axis, epsilon=epsilon)

# BYOL loss
def byol_loss(x, y):
    x = l2_normalize(x, axis=-1)
    y = l2_normalize(y, axis=-1)
    return 2 - 2 * tf.reduce_sum(x * y, axis=-1)

# MLP projector/predictor
def build_mlp(input_dim, projection_size, hidden_size=1024):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(projection_size)
    ])

# EMA updater
class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return self.beta * old + (1 - self.beta) * new

def update_moving_average(ema, target_model, online_model):
    for target_w, online_w in zip(target_model.weights, online_model.weights):
        target_w.assign(ema.update_average(target_w, online_w))

# NetWrapper: extracts representation from a given layer
class NetWrapper(tf.keras.Model):
    def __init__(self, base_encoder, projection_size, projection_hidden_size, layer_name=None):
        super().__init__()
        self.encoder = base_encoder
        self.layer_name = layer_name
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

    def get_representation(self, x):
        if self.layer_name is None:
            return self.encoder(x)
        intermediate_layer_model = tf.keras.Model(
            inputs=self.encoder.input,
            outputs=self.encoder.get_layer(self.layer_name).output
        )
        return intermediate_layer_model(x)

    def build_projector(self, hidden):
        if self.projector is None:
            input_dim = hidden.shape[-1]
            self.projector = build_mlp(input_dim, self.projection_size, self.projection_hidden_size)

    def call(self, x, return_projection=True):
        hidden = self.get_representation(x)
        self.build_projector(hidden)
        if not return_projection:
            return hidden
        projection = self.projector(hidden)
        return projection, hidden

# BYOL main class
class BYOL(tf.keras.Model):
    def __init__(
        self,
        base_encoder,
        image_size,
        # hidden_layer=None,
        projection_size=128,
        projection_hidden_size=1024,
        moving_average_decay=0.99
    ):
        super().__init__()
        self.online_encoder = NetWrapper(
            base_encoder,
            projection_size,
            projection_hidden_size,
            layer_name=None
        )
        self.target_encoder = NetWrapper(
            tf.keras.models.clone_model(base_encoder),
            projection_size,
            projection_hidden_size,
            layer_name=None
        )
        self.ema_updater = EMA(moving_average_decay)
        self.online_predictor = build_mlp(projection_size, projection_size, projection_hidden_size)

    def update_moving_average(self):
        update_moving_average(self.ema_updater, self.target_encoder, self.online_encoder)

    def call(self, x1, x2, training=True):
        # x1, x2: two augmented views of the same batch
        online_proj1, _ = self.online_encoder(x1, return_projection=True)
        online_proj2, _ = self.online_encoder(x2, return_projection=True)
        online_pred1 = self.online_predictor(online_proj1)
        online_pred2 = self.online_predictor(online_proj2)

        # target encoder: stop gradient
        target_proj1, _ = self.target_encoder(x1, return_projection=True)
        target_proj2, _ = self.target_encoder(x2, return_projection=True)
        target_proj1 = tf.stop_gradient(target_proj1)
        target_proj2 = tf.stop_gradient(target_proj2)

        loss1 = byol_loss(online_pred1, target_proj2)
        loss2 = byol_loss(online_pred2, target_proj1)
        loss = tf.reduce_mean(loss1 + loss2)
        return loss

# 사용 예시
# base_encoder = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(224,224,3))
# byol = BYOL(base_encoder, image_size=224)
# loss = byol(x1, x2)