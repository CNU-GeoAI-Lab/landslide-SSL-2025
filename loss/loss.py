import tensorflow as tf

def euclidean_distance(coords1, coords2):
    # coords1, coords2: [N,1,3] and [1,N,3]
    return tf.sqrt(tf.reduce_sum((coords1 - coords2) ** 2, axis=-1))  # [N,N]

def geo_loss(z, coords, sigma=10.0):
    """
    z: [batch, dim] - embedding vector
    coords: [batch, 3] - (x, y, z) in meters
    sigma: distance scale factor (meters)
    """
    N = tf.shape(z)[0]
    z_exp1 = tf.expand_dims(z, 1)  # [N,1,D]
    z_exp2 = tf.expand_dims(z, 0)  # [1,N,D]
    dist_embed = tf.reduce_sum((z_exp1 - z_exp2) ** 2, axis=-1)  # [N,N]

    coords_exp1 = tf.expand_dims(coords, 1)  # [N,1,3]
    coords_exp2 = tf.expand_dims(coords, 0)  # [1,N,3]
    dist_geo = euclidean_distance(coords_exp1, coords_exp2)  # [N,N]

    # 가까울수록 가중치가 크도록 설정
    w = tf.exp(-dist_geo / sigma)

    return tf.reduce_mean(w * dist_embed)


def debiased_geo_contrastive_loss(z1, z2, c1, c2, temperature=0.5, tau_plus=0.1, sigma=10.0, lambda_geo=0.1):
    """
    Debiased Contrastive Loss (NeurIPS 2020) - TensorFlow version

    Args:
        z1: [batch_size, dim] 첫 번째 view embedding (normalized)
        z2: [batch_size, dim] 두 번째 view embedding (normalized)
        temperature: softmax temperature (τ)
        tau_plus: false negative 비율 추정값 (0~1)

    Returns:
        scalar loss
    """
    batch_size = tf.shape(z1)[0]

    z1 = tf.math.l2_normalize(z1, -1)
    z2 = tf.math.l2_normalize(z2, -1)

    # 두 view 합치기 (총 2N 개)
    z = tf.concat([z1, z2], axis=0)  # [2N, dim]
    z = tf.math.l2_normalize(z, axis=1)  # L2 정규화

    # 유사도 행렬 계산
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # [2N, 2N]

    # 자기 자신과의 유사도 제거
    mask = tf.eye(2 * batch_size)
    sim = sim - 1e9 * mask  # 자기 자신은 매우 작은 값으로

    # Positive pair mask 만들기
    pos_mask = tf.roll(tf.eye(2 * batch_size), shift=batch_size, axis=0)

    # Positive similarity
    sim_pos = tf.reduce_sum(sim * pos_mask, axis=1)  # [2N]

    # Negative similarity
    neg_mask = 1.0 - pos_mask - mask
    neg_sim = sim * neg_mask

    # Negative exponential 합
    exp_neg = tf.reduce_sum(tf.exp(neg_sim), axis=1)

    # Debias term 적용
    # 실제 논문 식: denom = exp_pos + max( exp_neg - tau_plus * exp_pos / (1 - tau_plus), eps )
    exp_pos = tf.exp(sim_pos)
    N = tf.cast(2 * batch_size - 2, tf.float32)  # negative 개수

    # false negative 비율 조정
    debiased_neg = tf.maximum(exp_neg - tau_plus * N * exp_pos / (1 - tau_plus), 1e-9)

    # 최종 loss 계산
    loss = -tf.math.log(exp_pos / (exp_pos + debiased_neg))
    loss = tf.reduce_mean(loss)

    # 좌표 보정 loss
    coords_all = tf.concat([c1, c2], axis=0)  # positive pair 좌표 포함
    loss_geo = geo_loss(z, coords_all, sigma)
    
    return loss + lambda_geo * loss_geo

def simclr_loss_with_geo(z1, z2, c1, c2, temperature=0.2, sigma=10.0, lambda_geo=0.1):
    # 기본 SimCLR loss
    z1 = tf.math.l2_normalize(z1, -1)
    z2 = tf.math.l2_normalize(z2, -1)
    z = tf.concat([z1, z2], axis=0)
    N = tf.shape(z1)[0]
    
    sim = tf.matmul(z, z, transpose_b=True) / temperature
    mask = ~tf.eye(2 * N, dtype=tf.bool)
    exp_sim = tf.exp(sim) * tf.cast(mask, tf.float32)
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
    labels = tf.concat([tf.range(N, 2 * N), tf.range(0, N)], axis=0)
    loss_ntxent = -tf.reduce_mean(tf.gather_nd(log_prob, tf.stack([tf.range(2 * N), labels], axis=1)))

    # 좌표 보정 loss
    coords_all = tf.concat([c1, c2], axis=0)  # positive pair 좌표 포함
    loss_geo = geo_loss(z, coords_all, sigma)

    return loss_ntxent + lambda_geo * loss_geo

def simclr_loss_with_geo_positive(z1, z2, c1, c2, r, temperature=0.2):
    """
    SimCLR NT-Xent loss with extra positives: coordinates within r meters are positives
    z1, z2: [batch, dim]
    coords: [batch, 2] (x, y in meters)
    r: 반경 (m)
    """
    # Normalize embeddings
    z1 = tf.math.l2_normalize(z1, -1)
    z2 = tf.math.l2_normalize(z2, -1)
    z = tf.concat([z1, z2], axis=0)  # [2N, D]
    N = tf.shape(z1)[0]

    # Similarity matrix
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # [2N, 2N]
    mask = ~tf.eye(2 * N, dtype=tf.bool)

    # -----------------------
    # Positive mask by coordinates
    # -----------------------
    coords_all = tf.concat([c1, c2], axis=0)  # [2N, 2]
    coords_exp1 = tf.expand_dims(coords_all, 1)  # [2N,1,2]
    coords_exp2 = tf.expand_dims(coords_all, 0)  # [1,2N,2]
    dist_geo = tf.sqrt(tf.reduce_sum((coords_exp1 - coords_exp2) ** 2, axis=-1))  # [2N,2N]
    geo_pos_mask = tf.cast(dist_geo <= r, tf.float32)  # 반경 r 이내 positive

    # 자기 자신 제외
    geo_pos_mask = geo_pos_mask * tf.cast(mask, tf.float32)

    # -----------------------
    # NT-Xent with multiple positives
    # -----------------------
    exp_sim = tf.exp(sim) * tf.cast(mask, tf.float32)
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))

    # positive pair 평균 (좌표 기반 positive 전부 포함)
    pos_count = tf.reduce_sum(geo_pos_mask, axis=1)
    pos_count = tf.where(pos_count > 0, pos_count, tf.ones_like(pos_count))  # 0 방지
    positive_log_prob = tf.reduce_sum(log_prob * geo_pos_mask, axis=1) / pos_count

    loss = -tf.reduce_mean(positive_log_prob)
    return loss

def nt_xent(z1, z2, temperature=0.05, zdim=128):
    """Implements normalized temperature-scaled cross-entropy loss

    Args:
        z1: normalized latent representation of first set of augmented images [N, D]
        z2: normalized latent representation of second set of augmented images [N, D]
        batch_size: number of images in batch
        temperature: temperature for softmax. set in config
        zdim: dimension for latent representation set in config
    Returns:
        loss: contrastive loss averaged over batch (2*N samples)
    """
    batch_size = tf.shape(z1)[0]
    
    # reshape so that the order is z1_1,z2_1,z1_2,z2_2,z1_3,z2_3
    z = tf.concat([z1, z2], axis=0)                       # [2N, D]
    z = tf.math.l2_normalize(z, axis=1)
    z_ = tf.reshape(tf.transpose(tf.reshape(z, [2, batch_size, zdim]), [1,0,2]), [batch_size*2, -1])
    
    # compute cosine similarity
    # a has order [z1_1*batch_size*2, z1_2*batch_size*2, ...]
    # b has order [z1_1, z1_2, z3_1 ...]
    a = tf.reshape(tf.transpose(tf.tile(tf.reshape(z_, [1, batch_size*2, zdim]), [batch_size*2 ,1, 1]), [1, 0, 2]), [batch_size * 2 * batch_size * 2, zdim])
    b = tf.tile(z_, [batch_size*2, 1])
    sim = cosine_similarity(a, b)
    sim = tf.expand_dims(sim, axis=1)/temperature
    sim = tf.reshape(sim, [batch_size*2, batch_size*2])
    sim = tf.math.exp(sim-tf.reduce_max(sim, axis=1, keepdims=True))  # for numerical stability

    pos_indices = tf.concat([tf.range(1, (2*batch_size)**2, (batch_size*4)+2), tf.range(batch_size*2, (2*batch_size)**2, (batch_size*4)+2)], axis=0)
    pos_indices = tf.expand_dims(pos_indices, axis=1)
    pos_mask = tf.zeros(((2*batch_size)**2, 1), dtype=tf.int32)
    pos_mask = tf.tensor_scatter_nd_add(pos_mask, pos_indices, tf.ones((batch_size*2, 1), dtype=tf.int32))
    pos_mask = tf.reshape(pos_mask, [batch_size*2, batch_size*2])
    neg_mask = tf.ones((batch_size*2, batch_size*2), dtype=tf.int32) - tf.eye(batch_size*2, dtype=tf.int32)

    # similarity between z11-z12, z12-z11, z21-22, z22-z21 etc. 
    pos_sim = tf.reduce_sum(sim*tf.cast(pos_mask, tf.float32), axis=1) 

    # negative similarity consists of all similarities except i=j
    neg_sim = tf.reduce_sum(sim*tf.cast(neg_mask, tf.float32), axis=1)
    loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(pos_sim/neg_sim, 1e-10, 1.0)))

    return loss

def debiased_nt_xent(z1, z2, temperature=0.05, zdim=128, tau_plus=0.2):
    """Implements debiased normalized temperature-scaled cross-entropy loss

    Args:
        z1: normalized latent representation of first set of augmented images [N, D]
        z2: normalized latent representation of second set of augmented images [N, D]
        batch_size: number of images in batch
        temperature: temperature for softmax. set in config
        zdim: dimension for latent representation set in config
        tau_plus: estimate of fraction of false negatives in negatives
    Returns:
        loss: contrastive loss averaged over batch (2*N samples)
    """
    batch_size = tf.shape(z1)[0]
    
    # reshape so that the order is z1_1,z2_1,z1_2,z2_2,z1_3,z2_3
    z = tf.concat([z1, z2], axis=0)                       # [2N, D]
    z = tf.math.l2_normalize(z, axis=1)
    z_ = tf.reshape(tf.transpose(tf.reshape(z, [2, batch_size, zdim]), [1,0,2]), [batch_size*2, -1])
    
    # compute cosine similarity
    # a has order [z1_1*batch_size*2, z1_2*batch_size*2, ...]
    # b has order [z1_1, z1_2, z3_1 ...]
    a = tf.reshape(tf.transpose(tf.tile(tf.reshape(z_, [1, batch_size*2, zdim]), [batch_size*2 ,1, 1]), [1, 0, 2]), [batch_size * 2 * batch_size * 2, zdim])
    b = tf.tile(z_, [batch_size*2, 1])
    sim = cosine_similarity(a, b)
    sim = tf.expand_dims(sim, axis=1)/temperature
    sim = tf.reshape(sim, [batch_size*2, batch_size*2])
    sim = tf.math.exp(sim-tf.reduce_max(sim, axis=1, keepdims=True))

    pos_indices = tf.concat([tf.range(1, (2*batch_size)**2, (batch_size*4)+2), tf.range(batch_size*2, (2*batch_size)**2, (batch_size*4)+2)], axis=0)
    pos_indices = tf.expand_dims(pos_indices, axis=1)
    pos_mask = tf.zeros(((2*batch_size)**2, 1), dtype=tf.int32)
    pos_mask = tf.tensor_scatter_nd_add(pos_mask, pos_indices, tf.ones((batch_size*2, 1), dtype=tf.int32))
    pos_mask = tf.reshape(pos_mask, [batch_size*2, batch_size*2])
    neg_mask = tf.ones((batch_size*2, batch_size*2), dtype=tf.int32) - tf.eye(batch_size*2, dtype=tf.int32)

    # similarity between z11-z12, z12-z11, z21-22, z22-z21 etc. 
    pos_sim = tf.reduce_sum(sim*tf.cast(pos_mask, tf.float32), axis=1) 

    # negative similarity consists of all similarities except i=j
    neg_sim = tf.reduce_sum(sim*tf.cast(neg_mask, tf.float32), axis=1)
    N = tf.cast(2*batch_size-2, tf.float32) # number of negatives
    neg_sim = tf.maximum(neg_sim - tau_plus * N * pos_sim/(1-tau_plus), 1e-9)
    loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value((pos_sim)/(pos_sim+neg_sim), 1e-10, 1.0)))

    return loss

def cosine_similarity(a, b):
    """Computes the cosine similarity between vectors a and b"""

    numerator = tf.reduce_sum(tf.multiply(a, b), axis=1)
    denominator = tf.multiply(tf.norm(a, axis=1), tf.norm(b, axis=1))
    cos_similarity = numerator/denominator
    return cos_similarity

