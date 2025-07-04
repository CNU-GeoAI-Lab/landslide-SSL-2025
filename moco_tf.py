import tensorflow as tf
import numpy as np

def l2_normalize(x, axis=-1, epsilon=1e-12):
    return tf.math.l2_normalize(x, axis=axis, epsilon=epsilon)

def moco_loss(q, k, queue, temperature=0.07):
    # q: (batch, dim), k: (batch, dim), queue: (K, dim)
    # Positive logits: Nx1
    l_pos = tf.reduce_sum(q * k, axis=-1, keepdims=True)  # (N, 1)
    # Negative logits: NxK
    l_neg = tf.matmul(q, queue, transpose_b=True)  # (N, K)
    logits = tf.concat([l_pos, l_neg], axis=1)  # (N, 1+K)
    logits /= temperature
    labels = tf.zeros(tf.shape(logits)[0], dtype=tf.int32)  # positives are 0-th
    return logits, labels

def build_mlp(input_dim, projection_size, hidden_size=1024, kernel_regularizer=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, input_shape=(input_dim,), kernel_regularizer=kernel_regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(projection_size, kernel_regularizer=kernel_regularizer)
    ])

class MoCo(tf.keras.Model):
    def __init__(
        self,
        base_encoder,
        feature_dim=128,
        K=4096,
        m=0.999,
        temperature=0.07,
        hidden_size=1024,
        kernel_regularizer=None
    ):
        super().__init__()
        self.K = K
        self.m = m
        self.temperature = temperature
        self.feature_dim = feature_dim

        # Query encoder
        self.encoder_q = base_encoder
        self.proj_q = build_mlp(self.encoder_q.output_shape[-1], feature_dim, hidden_size, kernel_regularizer)

        # Key encoder (momentum encoder)
        self.encoder_k = tf.keras.models.clone_model(base_encoder)
        self.proj_k = build_mlp(self.encoder_k.output_shape[-1], feature_dim, hidden_size, kernel_regularizer)

        # Initialize key encoder weights to query encoder
        self.encoder_k.set_weights(self.encoder_q.get_weights())
        self.proj_k.set_weights(self.proj_q.get_weights())

        # Create the queue
        self.queue = tf.Variable(
            l2_normalize(tf.random.normal([K, feature_dim]), axis=1),
            trainable=False,
            dtype=tf.float32
        )
        self.queue_ptr = tf.Variable(0, dtype=tf.int64, trainable=False)

    @tf.function
    def momentum_update_key_encoder(self):
        # Momentum update for key encoder
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            kw.assign(self.m * kw + (1 - self.m) * qw)
        for qw, kw in zip(self.proj_q.weights, self.proj_k.weights):
            kw.assign(self.m * kw + (1 - self.m) * qw)

    def dequeue_and_enqueue(self, keys):
        batch_size = tf.shape(keys)[0]
        ptr = self.queue_ptr.numpy()
        assert self.K % batch_size.numpy() == 0  # for simplicity

        # Replace the keys at ptr (dequeue and enqueue)
        indices = tf.range(ptr, ptr + batch_size) % self.K
        updated_queue = tf.tensor_scatter_nd_update(self.queue, tf.expand_dims(indices, 1), keys)
        self.queue.assign(updated_queue)
        ptr = (ptr + batch_size.numpy()) % self.K
        self.queue_ptr.assign(ptr)

    def call(self, x_q, x_k, training=True):
        # Compute query features
        q = self.encoder_q(x_q, training=training)
        q = self.proj_q(q, training=training)
        q = l2_normalize(q, axis=1)

        # Compute key features
        k = self.encoder_k(x_k, training=training)
        k = self.proj_k(k, training=training)
        k = l2_normalize(k, axis=1)

        # Compute logits and labels
        logits, labels = moco_loss(q, k, self.queue, self.temperature)

        # Update queue
        self.dequeue_and_enqueue(k)

        return logits, labels

# 사용 예시
# base_encoder = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(224,224,3))
# moco = MoCo(base_encoder, feature_dim=128, K=4096, m=0.999)
# logits, labels = moco(x_q, x_k)