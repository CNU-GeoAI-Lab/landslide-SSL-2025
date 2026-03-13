import tensorflow as tf
from loss.loss import *
from model.simclr_model import *
import numpy as np

def pre_train_simclr_fusion(data_1_cont, data_1_cat, data_2_cont, data_2_cat, 
                             batch_size=128, epochs=10, learning_rate=0.0001, 
                             pre_model='ResNet', debiased=False, tau_plus=0.1, 
                             temperature=0.05, seed=42, strategy=None,
                             fusion_encoder_type='multi'):
    """
    Pre-train SimCLR fusion model with separate continuous and categorical inputs.
    
    Args:
        data_1_cont: First augmented view of continuous features
        data_1_cat: First augmented view of categorical embeddings
        data_2_cont: Second augmented view of continuous features
        data_2_cat: Second augmented view of categorical embeddings
        batch_size: Batch size for training (per replica if using MirroredStrategy)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        pre_model: Base model type ('ResNet' or 'ViT' supported for fusion)
        debiased: Whether to use debiased contrastive learning
        tau_plus: False negative ratio estimate for debiased loss
        temperature: Temperature parameter for contrastive loss
        seed: Random seed
        strategy: tf.distribute.Strategy for multi-GPU training (optional)
        fusion_encoder_type: Fusion encoder type ('multi', 'cross_attention', 'legacy')
    """
    # Data type conversion
    data_1_cont = tf.cast(data_1_cont, tf.float32)
    data_1_cat = tf.cast(data_1_cat, tf.float32)
    data_2_cont = tf.cast(data_2_cont, tf.float32)
    data_2_cat = tf.cast(data_2_cat, tf.float32)
    
    num_samples = data_1_cont.shape[0]
    continuous_input_shape = data_1_cont.shape[1:]
    categorical_input_shape = data_1_cat.shape[1:]
    
    # Adjust batch size for multi-GPU training
    if strategy is not None:
        global_batch_size = batch_size * strategy.num_replicas_in_sync
        steps_per_epoch = num_samples // global_batch_size
    else:
        global_batch_size = batch_size
        steps_per_epoch = num_samples // batch_size
    
    # Model initialization (within strategy.scope() if using multi-GPU)
    if strategy is not None:
        with strategy.scope():
            model = build_simclr_fusion_model(
                continuous_input_shape, categorical_input_shape, 
                embedding_dim=128, pre_model=pre_model,
                fusion_encoder_type=fusion_encoder_type
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        model = build_simclr_fusion_model(
            continuous_input_shape, categorical_input_shape, 
            embedding_dim=128, pre_model=pre_model,
            fusion_encoder_type=fusion_encoder_type
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Training step definition
    def train_step(cont_1, cat_1, cont_2, cat_2):
        with tf.GradientTape() as tape:
            z1 = model([cont_1, cat_1], training=True)
            z2 = model([cont_2, cat_2], training=True)
            
            # Debiased or standard contrastive loss
            if debiased:
                loss = debiased_nt_xent(z1, z2, temperature=temperature, zdim=128, tau_plus=tau_plus)
            else:
                loss = nt_xent(z1, z2, temperature=temperature, zdim=128)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    np.random.seed(seed)
    # Convert to numpy for easier shuffling
    data_1_cont_np = np.array(data_1_cont)
    data_1_cat_np = np.array(data_1_cat)
    data_2_cont_np = np.array(data_2_cont)
    data_2_cat_np = np.array(data_2_cat)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data using numpy
        indices = np.random.permutation(num_samples)
        data_1_cont_np = data_1_cont_np[indices]
        data_1_cat_np = data_1_cat_np[indices]
        data_2_cont_np = data_2_cont_np[indices]
        data_2_cat_np = data_2_cat_np[indices]
        
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        
        # Batch training
        for step in range(steps_per_epoch):
            start_idx = step * global_batch_size
            end_idx = start_idx + global_batch_size
            
            batch_cont_1 = tf.cast(data_1_cont_np[start_idx:end_idx], tf.float32)
            batch_cat_1 = tf.cast(data_1_cat_np[start_idx:end_idx], tf.float32)
            batch_cont_2 = tf.cast(data_2_cont_np[start_idx:end_idx], tf.float32)
            batch_cat_2 = tf.cast(data_2_cat_np[start_idx:end_idx], tf.float32)
            
            if strategy is not None:
                # Use strategy.run for multi-GPU training
                per_replica_losses = strategy.run(
                    train_step, 
                    args=(batch_cont_1, batch_cat_1, batch_cont_2, batch_cat_2)
                )
                loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / strategy.num_replicas_in_sync
            else:
                loss = train_step(batch_cont_1, batch_cat_1, batch_cont_2, batch_cat_2)
            
            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return model

def pre_train_simclr_triple(data_1, data_2, data_3, batch_size=128, epochs=10, learning_rate=0.0001, pre_model="ResNet", seed=42):
    data_1 = tf.cast(data_1, tf.float32)
    data_2 = tf.cast(data_2, tf.float32)
    data_3 = tf.cast(data_3, tf.float32)
    num_samples = data_1.shape[0]
    steps_per_epoch = num_samples // batch_size

    model = build_simclr_model(pre_model=pre_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(x1, x2, x3):
        with tf.GradientTape() as tape:
            z1 = model(x1, training=True)
            z2 = model(x2, training=True)
            z3 = model(x3, training=True)
            # 모든 쌍에 대해 loss 계산 (z1-z2, z1-z3, z2-z3)
            loss12 = nt_xent(z1, z2)
            loss13 = nt_xent(z1, z3)
            loss23 = nt_xent(z2, z3)
            loss = (loss12 + loss13 + loss23) / 3.0
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    np.random.seed(seed)
    for epoch in range(epochs):
        idx = np.random.permutation(num_samples)
        data_1, data_2, data_3 = data_1[idx], data_2[idx], data_3[idx]
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch_1 = data_1[start_idx:end_idx]
            batch_2 = data_2[start_idx:end_idx]
            batch_3 = data_3[start_idx:end_idx]
            loss = train_step(batch_1, batch_2, batch_3)
            total_loss += loss
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")

    return model

def pre_train_simclr(data_1, data_2, batch_size=128, epochs=10, learning_rate=0.0001, pre_model="ResNet", debiased=False, tau_plus=0.1, temperature=0.05, seed=42, strategy=None):
    """
    Pre-train SimCLR model
    
    Args:
        data_1: First augmented view of data
        data_2: Second augmented view of data
        batch_size: Batch size for training (per replica if using MirroredStrategy)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        pre_model: Base model type ['ResNet', 'CNN', 'ViT']
        debiased: Whether to use debiased contrastive learning
        tau_plus: False negative ratio estimate for debiased loss (used when debiased=True)
        temperature: Temperature parameter for contrastive loss
        seed: Random seed
        strategy: tf.distribute.Strategy for multi-GPU training (optional)
    """
    # 데이터 타입 변환 및 확인
    data_1 = tf.cast(data_1, tf.float32) 
    data_2 = tf.cast(data_2, tf.float32) 
    num_samples = data_1.shape[0]
    num_channels = data_1.shape[3]  # Get number of channels dynamically
    input_shape = data_1.shape[1:]
    # Adjust batch size for multi-GPU training
    if strategy is not None:
        global_batch_size = batch_size * strategy.num_replicas_in_sync
        steps_per_epoch = num_samples // global_batch_size
    else:
        global_batch_size = batch_size
        steps_per_epoch = num_samples // batch_size
    
    # 모델 초기화 (strategy.scope() 안에서 생성)
    if strategy is not None:
        with strategy.scope():
            model = build_simclr_model(input_shape=input_shape, pre_model=pre_model)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        model = build_simclr_model(input_shape=input_shape, pre_model=pre_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 학습 스텝 정의
    def train_step(pair_1, pair_2):
        with tf.GradientTape() as tape:
            z1 = model(pair_1, training=True)
            z2 = model(pair_2, training=True)
            
            # Debiased 또는 일반 contrastive loss 선택
            if debiased:
                # Debiased contrastive learning loss with tau_plus
                loss = debiased_nt_xent(z1, z2, temperature=temperature, zdim=128, tau_plus=tau_plus)
            else:
                # Standard contrastive learning loss
                loss = nt_xent(z1, z2, temperature=temperature, zdim=128)
                
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    data = np.concatenate((data_1, data_2), axis=3)

    np.random.seed(seed)
    # 학습 루프
    for epoch in range(epochs):
        np.random.shuffle(data)
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        # 배치 단위로 학습
        for step in range(steps_per_epoch):
            start_idx = step * global_batch_size
            end_idx = start_idx + global_batch_size
            batch_data = data[start_idx:end_idx]
            
            if strategy is not None:
                # Use strategy.run for multi-GPU training
                per_replica_losses = strategy.run(train_step, args=(batch_data[:,:,:,:num_channels], batch_data[:,:,:,num_channels:]))
                loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) / strategy.num_replicas_in_sync
            else:
                loss = train_step(batch_data[:,:,:,:num_channels], batch_data[:,:,:,num_channels:])
            
            total_loss += loss
        
        avg_loss = total_loss / steps_per_epoch
        print(f"Average Loss: {avg_loss:.4f}")
    
    return model

