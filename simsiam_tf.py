import tensorflow as tf
import numpy as np

# L2 normalize
def l2_normalize(x, axis=-1, epsilon=1e-12):
    return tf.math.l2_normalize(x, axis=axis, epsilon=epsilon)

# SimSiam loss (negative cosine similarity)
def simsiam_loss(p, z):
    p = l2_normalize(p, axis=-1)
    z = l2_normalize(z, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=-1))

# MLP for projector and predictor
def build_mlp(input_dim, projection_size, hidden_size=1024):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(projection_size, kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    ])

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

# SimSiam main class
class SimSiam(tf.keras.Model):
    def __init__(
        self,
        base_encoder,
        image_size,
        hidden_layer=None,
        projection_size=128,
        projection_hidden_size=1024,
        predictor_hidden_size=512
    ):
        super().__init__()
        self.encoder = NetWrapper(
            base_encoder,
            projection_size,
            projection_hidden_size,
            layer_name=hidden_layer
        )
        self.predictor = build_mlp(projection_size, projection_size, predictor_hidden_size)

    def call(self, x1, x2, training=True):
        # x1, x2: two augmented views of the same batch
        z1, _ = self.encoder(x1, return_projection=True)
        z2, _ = self.encoder(x2, return_projection=True)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # Stop gradient for targets
        z1 = tf.stop_gradient(z1)
        z2 = tf.stop_gradient(z2)
        loss1 = simsiam_loss(p1, z2)
        loss2 = simsiam_loss(p2, z1)
        loss = 0.5 * (loss1 + loss2)
        return loss

# 사용 예시
# base_encoder = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(224,224,3))
# simsiam = SimSiam(base_encoder, image_size=224)
# loss = simsiam(x1, x2)